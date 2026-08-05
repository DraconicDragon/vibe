"""Session construction and backend pooling utilities."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

from vibe.backends.base import (
    ArtifactMap,
    Backend,
    ExecutionPlan,
    ExecutionPreference,
    HardwareIntent,
    ModelPlugin,
)
from vibe.exceptions import SessionError
from vibe.hf_downloader import get_auto_download_default
from vibe.loader import LoaderError, resolve_variant_artifacts
from vibe.precision import PrecisionPolicy, PrecisionRequest, parse_precision
from vibe.session import ModelSession

logger = logging.getLogger(__name__)

_RUNTIME_POOL_LOCK = threading.RLock()
_RUNTIME_POOL: dict[tuple[Any, ...], tuple[Any, int]] = {}
_LOADING_LOCKS: dict[tuple[Any, ...], threading.Lock] = {}


def _resolve_backend_candidates(
    plugin_cls: type[ModelPlugin],
    requested_backend: Backend | str | None,
    preference: ExecutionPreference,
) -> list[Backend]:
    """Returns a prioritized list of backends to try."""
    supported = [v.backend for v in plugin_cls.variants]

    if requested_backend is not None:
        if isinstance(requested_backend, str):
            try:
                selected = Backend(requested_backend.lower())
            except ValueError:
                valid = [b.value for b in supported]
                raise SessionError(f"Unknown backend '{requested_backend}'. Choose from: {valid}") from None
        else:
            selected = requested_backend

        if selected not in supported:
            raise SessionError(
                f"Model '{plugin_cls.identity.model_id}' does not support backend '{selected.value}'. "
                f"Supported: {[b.value for b in supported]}"
            )
        return [selected]

    onnx_ok, onnx_accel = _onnx_runtime_capabilities()
    torch_ok, torch_accel = _pytorch_runtime_capabilities()

    candidates = []
    if Backend.ONNX in supported and onnx_ok:
        candidates.append((Backend.ONNX, onnx_accel))
    if Backend.PYTORCH in supported and torch_ok:
        candidates.append((Backend.PYTORCH, torch_accel))

    if not candidates:
        raise SessionError(
            f"No supported backend available for model '{plugin_cls.identity.model_id}'. Install onnxruntime or torch."
        )

    if len(candidates) == 1:
        return [candidates[0][0]]

    if preference.hint in {"mps", "xpu"}:
        return [Backend.PYTORCH, Backend.ONNX]
    if preference.hint in {"rocm", "dml", "openvino"}:
        return [Backend.ONNX, Backend.PYTORCH]

    if preference.intent == HardwareIntent.ACCELERATOR:
        if candidates[0][1] and not candidates[1][1]:
            return [Backend.ONNX, Backend.PYTORCH]
        if candidates[1][1] and not candidates[0][1]:
            return [Backend.PYTORCH, Backend.ONNX]

    return [Backend.ONNX, Backend.PYTORCH]


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
    """Build a ModelSession from a plugin class and a file source."""
    model_id = plugin_cls.identity.model_id

    try:
        preference = ExecutionPreference.parse(device)
        precision_req = parse_precision(precision)
    except ValueError as exc:
        raise SessionError(str(exc)) from exc

    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)
    candidates = _resolve_backend_candidates(plugin_cls, backend, preference)

    failures = []
    for candidate_backend in candidates:
        variant = next((v for v in plugin_cls.variants if v.backend == candidate_backend), None)
        if not variant:
            continue

        try:
            file_map = resolve_variant_artifacts(
                source=source,
                variant=variant,
                revision=hf_revision,
                cache_dir=hf_cache_dir,
                allow_download=effective_auto_download,
                file_name_map=file_name_map,
                source_map=source_map,
            )
        except LoaderError as exc:
            # Artifact resolution specifically failed (e.g., file missing)
            failures.append((candidate_backend, f"Artifact missing: {exc}"))
            logger.info("Artifacts unavailable for %s (backend %s): %s", model_id, candidate_backend.value, exc)
            continue
        except Exception as exc:
            failures.append((candidate_backend, f"Resolution error: {exc}"))
            logger.warning("Unexpected error resolving artifacts for %s: %s", model_id, exc)
            continue

        plan = ExecutionPlan(
            backend=candidate_backend,
            preference=preference,
            precision=precision_req,
            onnx_providers=tuple(onnx_providers) if onnx_providers is not None else None,
        )

        try:
            plugin = plugin_cls()
            plugin.load_ancillary(file_map)

            pool_key = _make_runtime_pool_key(plugin_cls, file_map, plan)
            runtime, release_fn = _acquire_runtime(
                key=pool_key,
                model_id=model_id,
                build=lambda p=plugin, fm=file_map, ep=plan: p.build_runtime(fm, ep),
            )
        except Exception as exc:
            if len(candidates) == 1 or backend is not None:
                raise SessionError(f"Failed to build runtime for '{model_id}': {exc}") from exc
            failures.append((candidate_backend, str(exc)))
            continue

        if candidate_backend == Backend.ONNX and precision_req.compute in {PrecisionPolicy.FP16, PrecisionPolicy.BF16}:
            logger.warning(
                "Precision '%s' requested while running ONNX backend; runtime casting is provider/model dependent.",
                precision_req.compute.value,
            )

        logger.debug("Session ready model_id=%s backend=%s", model_id, candidate_backend.value)
        return ModelSession(
            plugin=plugin,
            backend_instance=runtime,
            plan=plan,
            file_map=file_map,
            source=source,
            auto_download=effective_auto_download,
            memory_tracking=memory_tracking,
            backend_release=release_fn,
        )

    attempts = "; ".join(f"{b.value}: {reason}" for b, reason in failures)
    raise SessionError(f"Failed to resolve files or build runtime for '{model_id}'. Attempts: {attempts}")


def _make_runtime_pool_key(
    plugin_cls: type[ModelPlugin],
    artifacts: ArtifactMap,
    plan: ExecutionPlan,
) -> tuple[Any, ...]:
    """Key a completed runtime by all inputs which can affect its construction."""
    artifact_key = tuple(
        sorted((artifact_id, str(path.resolve())) for artifact_id, path in artifacts.as_path_dict().items())
    )
    return (
        plugin_cls.__module__,
        plugin_cls.__qualname__,
        artifact_key,
        plan.backend.value,
        plan.preference.intent.value,
        plan.preference.ordinal,
        plan.precision.weight.value,
        plan.precision.compute.value,
        plan.onnx_providers,
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
        instance = popped[0] if popped else None

    if instance is None:
        return

    close_fn = getattr(instance, "close", None)
    if callable(close_fn):
        close_fn()
    logger.debug("Closed pooled runtime key=%s", key)


def _onnx_runtime_capabilities() -> tuple[bool, bool]:
    try:
        import onnxruntime as ort  # ty:ignore[unresolved-import]
    except ImportError:
        return False, False
    try:
        get_providers = getattr(ort, "get_available_providers", None)
        available = {str(provider) for provider in get_providers()} if callable(get_providers) else set()
    except Exception:
        available = set()
    return True, any(provider != "CPUExecutionProvider" for provider in available)


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
