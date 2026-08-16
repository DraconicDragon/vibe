"""Session construction and backend pooling utilities."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from vibe.backends.base import (
    ArtifactMap,
    Backend,
    ExecutionPlan,
    ExecutionPreference,
    HardwareIntent,
    ModelPlugin,
    ModelVariant,
    RuntimeExecutor,
)
from vibe.exceptions import LoaderError, SessionError
from vibe.hf_downloader import get_auto_download_default
from vibe.loader import resolve_variant_artifacts
from vibe.precision import PrecisionPolicy, PrecisionRequest, parse_precision
from vibe.session import ModelSession

logger = logging.getLogger(__name__)

_RUNTIME_POOL_LOCK = threading.RLock()
_RUNTIME_POOL: dict[tuple[Any, ...], tuple[RuntimeExecutor, int]] = {}
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

    # Map available backends to whether they have GPU/accelerator support
    available: dict[Backend, bool] = {}
    if Backend.PYTORCH in supported and torch_ok:
        available[Backend.PYTORCH] = torch_accel
    if Backend.ONNX in supported and onnx_ok:
        available[Backend.ONNX] = onnx_accel

    if not available:
        raise SessionError(
            f"No supported backend available for model '{plugin_cls.identity.model_id}'. "
            f"Checked PyTorch (installed={torch_ok}), ONNX Runtime (installed={onnx_ok})."
        )

    # Fail hard if an accelerator was explicitly requested but no GPU acceleration is active in any runtime
    if preference.intent == HardwareIntent.ACCELERATOR and not any(available.values()):
        device_str = preference.hint or "accelerator"
        diag = [
            f"PyTorch (installed={torch_ok}, accelerator={torch_accel})",
            f"ONNX Runtime (installed={onnx_ok}, accelerator={onnx_accel})",
        ]
        logger.error(
            "Accelerator device '%s' requested for model '%s', but no backend has accelerator support: %s",
            device_str,
            plugin_cls.identity.model_id,
            ", ".join(diag),
        )
        raise SessionError(
            f"Accelerator device '{device_str}' explicitly requested, but no active GPU/accelerator support was found. "
            f"Backend capabilities: {', '.join(diag)}. "
            "Install PyTorch with CUDA/ROCm/MPS support or 'onnxruntime-gpu'."
        )

    if len(available) == 1:
        return list(available.keys())

    # Framework-specific hints force a preferred ordering
    if preference.hint in {"mps", "xpu"}:
        return [Backend.PYTORCH, Backend.ONNX]
    if preference.hint in {"rocm", "dml", "openvino"}:
        return [Backend.ONNX, Backend.PYTORCH]

    # If accelerator requested, prefer the backend that actually has GPU acceleration available
    if preference.intent == HardwareIntent.ACCELERATOR:
        pytorch_accel = available[Backend.PYTORCH]
        onnx_accel = available[Backend.ONNX]

        if pytorch_accel and not onnx_accel:
            return [Backend.PYTORCH, Backend.ONNX]
        if onnx_accel and not pytorch_accel:
            return [Backend.ONNX, Backend.PYTORCH]

    # Default preference when capabilities are equivalent (PyTorch preferred)
    return [Backend.PYTORCH, Backend.ONNX]


def build_session(
    plugin_cls: type[ModelPlugin],
    source: str,
    backend: Backend | str | None = None,
    variant: str | None = None,
    device: str = "auto",
    precision: str | PrecisionRequest = "auto",
    onnx_providers: list[str] | None = None,
    hf_token: str | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    auto_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    source_map: Mapping[str, str] | None = None,
    options: Mapping[str, Any] | None = None,
    memory_tracking: bool = False,
) -> ModelSession:
    """Build a ModelSession from a plugin class and a file source."""
    started_at = time.perf_counter()
    model_id = plugin_cls.identity.model_id

    try:
        preference = ExecutionPreference.parse(device)
        precision_req = parse_precision(precision)
    except ValueError as exc:
        raise SessionError(str(exc)) from exc

    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)

    # 1. Resolve which variants to try
    variants_to_try: list[ModelVariant] = []
    if variant is not None:
        target_variant = next((v for v in plugin_cls.variants if v.variant_id == variant), None)
        if not target_variant:
            available = [v.variant_id for v in plugin_cls.variants if v.variant_id]
            raise SessionError(f"Model '{model_id}' has no variant '{variant}'. Available variants: {available}")

        # Check for explicit backend conflict
        if backend is not None:
            req_b = Backend(backend.lower()) if isinstance(backend, str) else backend
            if req_b != target_variant.backend:
                raise SessionError(
                    f"Variant '{variant}' requires backend '{target_variant.backend.value}', "
                    f"but backend '{req_b.value}' was requested."
                )
        variants_to_try = [target_variant]
    else:
        candidates = _resolve_backend_candidates(plugin_cls, backend, preference)
        # Add all variants matching the candidate backends, preserving order of declaration
        for cand in candidates:
            variants_to_try.extend(v for v in plugin_cls.variants if v.backend == cand)

    failures = []

    # 2. Try loading variants until one succeeds
    for selected_variant in variants_to_try:
        resolved_variant = selected_variant.resolve(plugin_cls.default_repo_id)
        candidate_backend = resolved_variant.backend
        try:
            file_map = resolve_variant_artifacts(
                source=source,
                variant=resolved_variant,
                revision=hf_revision,
                cache_dir=hf_cache_dir,
                allow_download=effective_auto_download,
                file_name_map=file_name_map,
                source_map=source_map,
                token=hf_token,
            )
        except LoaderError as exc:
            # Artifact resolution failed. If we have more variants to try, continue seamlessly.
            vid_str = f" (variant: {selected_variant.variant_id})" if selected_variant.variant_id else ""
            failures.append((candidate_backend, f"Artifact missing{vid_str}: {exc}"))
            logger.info(
                "Artifacts unavailable for %s backend %s%s: %s", model_id, candidate_backend.value, vid_str, exc
            )
            continue
        except Exception as exc:
            failures.append((candidate_backend, f"Resolution error: {exc}"))
            logger.warning("Unexpected error resolving artifacts for %s: %s", model_id, exc)
            continue

        plan = ExecutionPlan(
            backend=candidate_backend,
            preference=preference,
            precision=precision_req,
            variant_id=selected_variant.variant_id,
            onnx_providers=tuple(onnx_providers) if onnx_providers is not None else None,
            hf_token=hf_token,
        )

        try:
            plugin = plugin_cls()
            plugin.set_options(options)
            plugin.load_ancillary(file_map)

            pool_key = _make_runtime_pool_key(plugin_cls, file_map, plan, options=plugin._options)
            runtime, release_fn = _acquire_runtime(
                key=pool_key,
                model_id=model_id,
                build=lambda p=plugin, fm=file_map, ep=plan: p.build_runtime(fm, ep),
            )
        except Exception as exc:
            if len(variants_to_try) == 1 or backend is not None:
                raise SessionError(f"Failed to build runtime for '{model_id}': {exc}") from exc
            failures.append((candidate_backend, str(exc)))
            continue

        if candidate_backend == Backend.ONNX and precision_req.compute in {PrecisionPolicy.FP16, PrecisionPolicy.BF16}:
            logger.warning(
                "Precision '%s' requested while running ONNX backend; runtime casting is provider/model dependent.",
                precision_req.compute.value,
            )

        runtime_info = runtime.execution_info()
        variant_str = selected_variant.variant_id or "default"
        load_seconds = time.perf_counter() - started_at

        if candidate_backend == Backend.PYTORCH:
            prec = runtime_info.get("precision") or {}
            logger.info(
                "Session ready model_id=%s | variant=%s | backend=pytorch | device=%s | weights=%s | compute=%s | autocast=%s | time=%.2fs",
                model_id,
                variant_str,
                runtime_info.get("device"),
                prec.get("weight_dtype"),
                prec.get("compute_dtype"),
                prec.get("autocast_enabled"),
                load_seconds,
            )
        else:  # ONNX
            providers = runtime_info.get("providers") or []
            primary_ep = providers[0] if providers else "unknown"
            logger.info(
                "Session ready model_id=%s | variant=%s | backend=onnx | provider=%s | graph_precision=%s | time=%.2fs",
                model_id,
                variant_str,
                primary_ep,
                runtime_info.get("graph_precision", "unknown"),
                load_seconds,
            )

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


def _make_hashable(val: Any) -> Any:
    """Recursively convert arbitrary data structures (dicts, lists, sets, arrays) into hashable tuples."""
    if isinstance(val, dict):
        return tuple(sorted(((str(k), _make_hashable(v)) for k, v in val.items()), key=lambda item: item[0]))
    if isinstance(val, (list, tuple)):
        return tuple(_make_hashable(v) for v in val)
    if isinstance(val, set):
        return tuple(sorted((_make_hashable(v) for v in val), key=repr))
    if hasattr(val, "tolist") and callable(val.tolist):  # Handles numpy arrays and torch tensors
        try:
            return _make_hashable(val.tolist())
        except Exception as exc:
            logger.debug("Failed to convert array-like object with tolist() for pool key hashing: %s", exc)
    try:
        hash(val)
        return val
    except TypeError:
        return repr(val)


def _make_runtime_pool_key(
    plugin_cls: type[ModelPlugin],
    artifacts: ArtifactMap,
    plan: ExecutionPlan,
    options: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Key a completed runtime by all inputs which can affect its construction."""
    artifact_key = tuple(
        sorted((artifact_id, str(path.resolve())) for artifact_id, path in artifacts.as_path_dict().items())
    )
    options_key = tuple(sorted((k, _make_hashable(v)) for k, v in options.items())) if options else ()
    return (
        plugin_cls.__module__,
        plugin_cls.__qualname__,
        artifact_key,
        plan.backend.value,
        plan.variant_id,
        plan.preference.intent.value,
        plan.preference.ordinal,
        plan.precision.weight.value,
        plan.precision.compute.value,
        plan.onnx_providers,
        options_key,
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

            logger.debug("Runtime ready model_id=%s", model_id)
            return instance, lambda: _release_runtime(key)

        finally:
            with _RUNTIME_POOL_LOCK:
                _LOADING_LOCKS.pop(key, None)


def _release_runtime(key: tuple[Any, ...]) -> None:
    with _RUNTIME_POOL_LOCK:
        cached = _RUNTIME_POOL.get(key)
        if cached is None:
            return

        if cached[1] > 1:
            _RUNTIME_POOL[key] = (cached[0], cached[1] - 1)
            logger.debug("Released pooled runtime key=%s refcount=%s", key, cached[1] - 1)
            return

        popped = _RUNTIME_POOL.pop(key, None)
        instance = popped[0] if popped else None

    if instance is None:
        return

    try:
        instance.close()
    except Exception:
        logger.exception("Failed to close pooled runtime during release.")

    logger.debug("Closed pooled runtime key=%s", key)


def _onnx_runtime_capabilities() -> tuple[bool, bool]:
    try:
        import onnxruntime as ort  # ty:ignore[unresolved-import, unused-ignore-comment]
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
