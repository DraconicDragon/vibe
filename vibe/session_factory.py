"""Session construction and backend pooling utilities."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

from vibe.backends.base import (
    ArtifactMap,
    Backend,
    ExecutionPreference,
    ExecutionRequest,
    HardwareIntent,
    ModelPlugin,
)
from vibe.exceptions import SessionError
from vibe.hf_downloader import get_auto_download_default
from vibe.loader import resolve_variant_artifacts
from vibe.precision import PrecisionPolicy, PrecisionRequest, parse_precision
from vibe.session import ModelSession

logger = logging.getLogger(__name__)

_RUNTIME_POOL_LOCK = threading.RLock()
_RUNTIME_POOL: dict[tuple[Any, ...], tuple[Any, int]] = {}
_LOADING_LOCKS: dict[tuple[Any, ...], threading.Lock] = {}


def build_session(
    plugin_cls: type[ModelPlugin],
    source: str,
    backend: Backend | str | None = None,
    device: str = "auto",
    precision: str | PrecisionRequest = "auto",
    onnx_providers: list[str] | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    auto_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    source_map: Mapping[str, str] | None = None,
    memory_tracking: bool = False,
) -> ModelSession:
    """
    Build a ModelSession from a plugin class and a file source.

    This is called by vibe.load() - you don't usually call this directly.
    """
    model_id = plugin_cls.identity.model_id
    supported_backends = [v.backend for v in plugin_cls.variants]

    logger.debug(
        "Building session model_id=%s requested_backend=%s requested_device=%s requested_precision=%s source=%s",
        model_id,
        backend.value if isinstance(backend, Backend) else backend or "auto",
        device,
        precision,
        source,
    )

    backend_was_explicit = backend is not None
    if backend is None:
        selected_backend = _auto_select_backend(plugin_cls, requested_device=device)
    elif isinstance(backend, str):
        try:
            selected_backend = Backend(backend.lower())
        except ValueError:
            valid_backends = [b.value for b in supported_backends]
            raise SessionError(
                f"Unknown backend '{backend}'. Choose from: {valid_backends}" if valid_backends else ""
            ) from None
    else:
        selected_backend = backend

    if selected_backend not in supported_backends:
        raise SessionError(
            f"Model '{model_id}' does not support backend '{selected_backend.value}'. "
            f"Supported: {[b.value for b in supported_backends]}"
        )

    backend_candidates = [selected_backend]
    if not backend_was_explicit:
        backend_candidates.extend([b for b in supported_backends if b != selected_backend])

    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)
    logger.debug("Session auto_download=%s", effective_auto_download)

    auto_resolution_failures: list[tuple[Backend, str]] = []

    for candidate_backend in backend_candidates:
        session = _attempt_session_build(
            plugin_cls=plugin_cls,
            source=source,
            candidate_backend=candidate_backend,
            selected_backend=selected_backend,
            backend_was_explicit=backend_was_explicit,
            device=device,
            precision=precision,
            onnx_providers=onnx_providers,
            hf_revision=hf_revision,
            hf_cache_dir=hf_cache_dir,
            auto_download_attempt=effective_auto_download,
            effective_auto_download=effective_auto_download,
            file_name_map=file_name_map,
            source_map=source_map,
            memory_tracking=memory_tracking,
            backend_candidates=backend_candidates,
            auto_resolution_failures=auto_resolution_failures,
        )
        if session is not None:
            return session

    if auto_resolution_failures:
        attempts = "; ".join([f"{candidate.value}: {reason}" for candidate, reason in auto_resolution_failures])
        raise SessionError(f"Could not resolve files for model '{model_id}' in auto backend mode. Attempts: {attempts}")

    raise SessionError(f"Failed to build session for model '{model_id}'.")


def _attempt_session_build(
    plugin_cls: type[ModelPlugin],
    source: str,
    candidate_backend: Backend,
    selected_backend: Backend,
    backend_was_explicit: bool,
    device: str,
    precision: str | PrecisionRequest,
    onnx_providers: list[str] | None,
    hf_revision: str | None,
    hf_cache_dir: str | None,
    auto_download_attempt: bool,
    effective_auto_download: bool,
    file_name_map: Mapping[str, str] | None,
    source_map: Mapping[str, str] | None,
    memory_tracking: bool,
    backend_candidates: list[Backend],
    auto_resolution_failures: list[tuple[Backend, str]],
) -> ModelSession | None:
    model_id = plugin_cls.identity.model_id

    try:
        preference = ExecutionPreference.parse(device)
    except ValueError as exc:
        raise SessionError(str(exc)) from exc

    try:
        precision_request = parse_precision(precision)
    except ValueError as exc:
        raise SessionError(str(exc)) from exc

    if not backend_was_explicit and candidate_backend != selected_backend:
        logger.info(
            "Auto backend selected %s for model_id=%s after %s was unavailable.",
            candidate_backend.value,
            model_id,
            selected_backend.value,
        )

    logger.debug(
        "Session backend selected model_id=%s backend=%s preference=%s precision=%s",
        model_id,
        candidate_backend.value,
        preference,
        precision_request,
    )

    variant = next((v for v in plugin_cls.variants if v.backend == candidate_backend), None)
    if variant is None:
        return None

    try:
        file_map = resolve_variant_artifacts(
            source=source,
            variant=variant,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            allow_download=auto_download_attempt,
            file_name_map=file_name_map,
            source_map=source_map,
        )
    except Exception as exc:
        if backend_was_explicit or len(backend_candidates) == 1:
            raise SessionError(str(exc)) from exc

        next_backend = _next_backend_candidate(backend_candidates, candidate_backend)
        if next_backend is not None:
            logger.info(
                "Auto backend '%s' unavailable for model_id=%s; trying %s next.",
                candidate_backend.value,
                model_id,
                next_backend.value,
            )
        auto_resolution_failures.append((candidate_backend, str(exc)))
        return None

    request = ExecutionRequest(
        backend=candidate_backend,
        preference=preference,
        precision=precision_request,
        onnx_providers=tuple(onnx_providers) if onnx_providers is not None else None,
    )

    try:
        plugin = plugin_cls()
        plugin.load_ancillary(file_map)
    except Exception as exc:
        raise SessionError(f"Plugin '{model_id}' failed to initialize artifacts: {exc}") from exc

    pool_key = _make_runtime_pool_key(
        plugin_cls=plugin_cls,
        artifacts=file_map,
        request=request,
    )
    try:
        runtime, release_runtime = _acquire_runtime(
            key=pool_key,
            model_id=model_id,
            build=lambda: plugin.build_runtime(file_map, request),
        )
    except Exception as exc:
        raise SessionError(f"Plugin '{model_id}' failed to build its runtime: {exc}") from exc

    try:
        if candidate_backend == Backend.ONNX and precision_request.compute in {
            PrecisionPolicy.FP16,
            PrecisionPolicy.BF16,
        }:
            logger.warning(
                "Precision '%s' requested while running ONNX backend; runtime casting is provider/model dependent.",
                precision_request.compute.value,
            )

        logger.debug("Session ready model_id=%s", model_id)
        return ModelSession(
            plugin=plugin,
            backend_instance=runtime,
            backend=candidate_backend,
            file_map=file_map,
            source=source,
            auto_download=effective_auto_download,
            memory_tracking=memory_tracking,
            backend_release=release_runtime,
        )
    except SessionError:
        release_runtime()
        raise
    except Exception as exc:
        release_runtime()
        raise SessionError(f"Plugin '{model_id}' failed to build a session: {exc}") from exc


def _next_backend_candidate(candidates: list[Backend], current: Backend) -> Backend | None:
    try:
        index = candidates.index(current)
    except ValueError:
        return None
    if index + 1 >= len(candidates):
        return None
    return candidates[index + 1]


def _make_runtime_pool_key(
    *,
    plugin_cls: type[ModelPlugin],
    artifacts: ArtifactMap,
    request: ExecutionRequest,
) -> tuple[Any, ...]:
    """Key a completed runtime by all inputs which can affect its construction."""
    artifact_key = tuple(
        sorted((artifact_id, str(path.resolve())) for artifact_id, path in artifacts.as_path_dict().items())
    )
    return (
        plugin_cls.__module__,
        plugin_cls.__qualname__,
        artifact_key,
        request.backend.value,
        request.preference.intent.value,
        request.preference.ordinal,
        request.precision.weight.value,
        request.precision.compute.value,
        request.onnx_providers,
    )


def _acquire_runtime(
    *,
    key: tuple[Any, ...],
    model_id: str,
    build: Callable[[], Any],
) -> tuple[Any, Callable[[], None]]:
    with _RUNTIME_POOL_LOCK:
        if key in _RUNTIME_POOL:
            instance, refcount = _RUNTIME_POOL[key]
            _RUNTIME_POOL[key] = (instance, refcount + 1)
            logger.debug("Reusing pooled runtime model_id=%s refcount=%s", model_id, refcount + 1)
            return instance, lambda: _release_runtime(key)

        if key not in _LOADING_LOCKS:
            _LOADING_LOCKS[key] = threading.Lock()
        model_lock = _LOADING_LOCKS[key]

    with model_lock:
        with _RUNTIME_POOL_LOCK:
            if key in _RUNTIME_POOL:
                instance, refcount = _RUNTIME_POOL[key]
                _RUNTIME_POOL[key] = (instance, refcount + 1)
                return instance, lambda: _release_runtime(key)

        try:
            instance = build()

            with _RUNTIME_POOL_LOCK:
                _RUNTIME_POOL[key] = (instance, 1)

            logger.info("Runtime ready model_id=%s", model_id)
            return instance, lambda: _release_runtime(key)

        finally:
            with _RUNTIME_POOL_LOCK:
                _LOADING_LOCKS.pop(key, None)


def _release_runtime(key: tuple[Any, ...]) -> None:
    instance: Any | None = None
    with _RUNTIME_POOL_LOCK:
        cached = _RUNTIME_POOL.get(key)
        if cached is None:
            return

        instance, refcount = cached
        if refcount > 1:
            _RUNTIME_POOL[key] = (instance, refcount - 1)
            logger.debug("Released pooled runtime key=%s refcount=%s", key, refcount - 1)
            return

        popped = _RUNTIME_POOL.pop(key, None)
        if popped is not None:
            instance, _ = popped
        else:
            instance = None

    if instance is None:
        return

    close_fn = getattr(instance, "close", None)
    if callable(close_fn):
        close_fn()
    logger.debug("Closed pooled runtime key=%s", key)


def _auto_select_backend(plugin_cls: type[ModelPlugin], *, requested_device: str = "auto") -> Backend:
    """Select backend using runtime availability + accelerator preference."""
    supported = [v.backend for v in plugin_cls.variants]

    onnx_runtime_ok, onnx_has_accel = _onnx_runtime_capabilities()
    torch_runtime_ok, torch_has_accel = _pytorch_runtime_capabilities()

    onnx_candidate = Backend.ONNX in supported and onnx_runtime_ok
    torch_candidate = Backend.PYTORCH in supported and torch_runtime_ok

    if onnx_candidate and torch_candidate:
        preference = ExecutionPreference.parse(requested_device)

        if preference.hint in {"mps", "xpu"}:
            logger.info("Backend auto-selection chose PyTorch due to requested device hint '%s'.", preference.hint)
            return Backend.PYTORCH
        if preference.hint in {"rocm", "dml", "openvino"}:
            logger.info("Backend auto-selection chose ONNX due to requested device hint '%s'.", preference.hint)
            return Backend.ONNX

        if preference.intent == HardwareIntent.ACCELERATOR:
            if onnx_has_accel and not torch_has_accel:
                logger.info("Backend auto-selection chose ONNX (accelerator available).")
                return Backend.ONNX
            if torch_has_accel and not onnx_has_accel:
                logger.info("Backend auto-selection chose PyTorch (accelerator available).")
                return Backend.PYTORCH

        logger.info("Backend auto-selection chose ONNX (default preference when capabilities are equivalent).")
        return Backend.ONNX

    if onnx_candidate:
        logger.info("Backend auto-selection chose ONNX (runtime available).")
        return Backend.ONNX
    if torch_candidate:
        logger.info("Backend auto-selection chose PyTorch (runtime available).")
        return Backend.PYTORCH

    raise SessionError(
        f"No supported backend available for model '{plugin_cls.identity.model_id}'. Install onnxruntime or torch."
    )


def _onnx_runtime_capabilities() -> tuple[bool, bool]:
    try:
        import onnxruntime as ort  # ty:ignore[unresolved-import]
    except ImportError:
        return False, False

    try:
        get_providers = getattr(ort, "get_available_providers", None)
        if callable(get_providers):
            available = {str(provider) for provider in get_providers()}
        else:
            available = set()
    except Exception:
        available = set()

    has_accelerator = any(provider != "CPUExecutionProvider" for provider in available)
    return True, has_accelerator


def _pytorch_runtime_capabilities() -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False

    has_cuda = bool(torch.cuda.is_available())

    xpu_mod = getattr(torch, "xpu", None)
    has_xpu = bool(xpu_mod and callable(getattr(xpu_mod, "is_available", None)) and xpu_mod.is_available())

    mps_backend = getattr(torch.backends, "mps", None)
    has_mps = bool(mps_backend and callable(getattr(mps_backend, "is_available", None)) and mps_backend.is_available())

    return True, (has_cuda or has_xpu or has_mps)
