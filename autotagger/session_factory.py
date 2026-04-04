"""Session construction and backend pooling utilities."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from autotagger.backends.base import Backend, ModelPlugin
from autotagger.devices import normalize_device_string
from autotagger.hf_downloader import get_auto_download_default
from autotagger.loader import FileMap, resolve_from_source_string
from autotagger.session import ModelSession, SessionError

logger = logging.getLogger(__name__)

_BACKEND_POOL_LOCK = threading.RLock()
_BACKEND_POOL: dict[tuple[Any, ...], tuple[Any, int]] = {}


def build_session(
    plugin_cls: type[ModelPlugin],
    source: str,
    backend: Backend | str | None = None,
    device: str = "cpu",
    onnx_providers: list[str] | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    auto_download: bool | None = None,
    memory_tracking: bool = True,
) -> ModelSession:
    """
    Build a ModelSession from a plugin class and a file source.

    This is called by autotagger.load() - you don't usually call this directly.
    """
    from autotagger.backends.runtime.onnx import ONNXBackend
    from autotagger.backends.runtime.pytorch import PyTorchBackend

    if backend is None:
        backend = _auto_select_backend(plugin_cls)
    elif isinstance(backend, str):
        try:
            backend = Backend(backend.lower())
        except ValueError:
            raise SessionError(f"Unknown backend '{backend}'. Choose from: {[b.value for b in Backend]}")

    if backend not in plugin_cls.supported_backends:
        raise SessionError(
            f"Model '{plugin_cls.model_id}' does not support backend '{backend.value}'. "
            f"Supported: {[b.value for b in plugin_cls.supported_backends]}"
        )

    try:
        normalized_device = normalize_device_string(
            device,
            backend="pytorch" if backend == Backend.PYTORCH else "onnx",
        )
    except ValueError as exc:
        raise SessionError(str(exc)) from exc

    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)

    try:
        file_map = resolve_from_source_string(
            source,
            plugin_cls.required_files,
            backend,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            allow_download=effective_auto_download,
        )
    except Exception as exc:
        raise SessionError(str(exc)) from exc

    weights_path = _find_weights(plugin_cls, file_map, backend)

    pool_key = _make_backend_pool_key(
        backend=backend,
        weights_path=weights_path,
        device=normalized_device,
        providers=onnx_providers,
    )
    rt, release_backend = _acquire_backend(
        key=pool_key,
        backend=backend,
        weights_path=weights_path,
        device=normalized_device,
        providers=onnx_providers,
        pytorch_cls=PyTorchBackend,
        onnx_cls=ONNXBackend,
    )

    try:
        plugin = plugin_cls()
        plugin.configure(
            auto_download=effective_auto_download,
        )
        plugin.load_ancillary(file_map.as_path_dict())
        return ModelSession(
            plugin=plugin,
            backend_instance=rt,
            backend=backend,
            file_map=file_map,
            source=source,
            auto_download=effective_auto_download,
            memory_tracking=memory_tracking,
            backend_release=release_backend,
        )
    except SessionError:
        release_backend()
        raise
    except Exception as exc:
        release_backend()
        raise SessionError(f"Plugin '{plugin_cls.model_id}' failed to load ancillary files: {exc}") from exc


def _make_backend_pool_key(
    *,
    backend: Backend,
    weights_path: Path,
    device: str,
    providers: list[str] | None,
) -> tuple[Any, ...]:
    resolved_path = str(weights_path.resolve())
    if backend == Backend.PYTORCH:
        return (backend.value, resolved_path, device)
    provider_key = tuple(providers) if providers is not None else ("AUTO",)
    return (backend.value, resolved_path, provider_key, device)


def _acquire_backend(
    *,
    key: tuple[Any, ...],
    backend: Backend,
    weights_path: Path,
    device: str,
    providers: list[str] | None,
    pytorch_cls: Any,
    onnx_cls: Any,
) -> tuple[Any, Callable[[], None]]:
    with _BACKEND_POOL_LOCK:
        cached = _BACKEND_POOL.get(key)
        if cached is not None:
            instance, refcount = cached
            _BACKEND_POOL[key] = (instance, refcount + 1)
            logger.debug("Reusing pooled backend key=%s refcount=%s", key, refcount + 1)
            return instance, lambda: _release_backend(key)

        if backend == Backend.PYTORCH:
            instance = pytorch_cls()
            instance.load(weights_path, device=device)
        else:
            instance = onnx_cls()
            instance.load(weights_path, providers=providers, device=device)

        _BACKEND_POOL[key] = (instance, 1)
        logger.debug("Created pooled backend key=%s refcount=1", key)
        return instance, lambda: _release_backend(key)


def _release_backend(key: tuple[Any, ...]) -> None:
    instance: Any | None = None
    with _BACKEND_POOL_LOCK:
        cached = _BACKEND_POOL.get(key)
        if cached is None:
            return

        instance, refcount = cached
        if refcount > 1:
            _BACKEND_POOL[key] = (instance, refcount - 1)
            logger.debug("Released pooled backend key=%s refcount=%s", key, refcount - 1)
            return

        popped = _BACKEND_POOL.pop(key, None)
        if popped is not None:
            instance, _ = popped
        else:
            instance = None

    if instance is None:
        return

    close_fn = getattr(instance, "close", None)
    if callable(close_fn):
        close_fn()
    logger.debug("Closed pooled backend key=%s", key)


def _auto_select_backend(plugin_cls: type[ModelPlugin]) -> Backend:
    """Prefer ONNX if available; fall back to PyTorch."""
    supported = plugin_cls.supported_backends
    if Backend.ONNX in supported:
        try:
            import onnxruntime  # noqa: F401

            return Backend.ONNX
        except ImportError:
            pass
    if Backend.PYTORCH in supported:
        return Backend.PYTORCH
    raise SessionError(f"No supported backend available for '{plugin_cls.model_id}'. Install onnxruntime or torch.")


def _find_weights(
    plugin_cls: type[ModelPlugin],
    file_map: FileMap,
    backend: Backend,
) -> Path:
    """Find the weights file from the resolved file map for the selected backend."""
    from autotagger.backends.base import FileRole

    for spec in plugin_cls.required_files:
        if spec.role == FileRole.WEIGHTS and spec.needed_for(backend):
            path = file_map.get(spec.name)
            if path is not None:
                return path

    raise SessionError(
        f"No weights file found for model '{plugin_cls.model_id}' "
        f"with backend '{backend.value}'. "
        f"Check the plugin's required_files declaration."
    )
