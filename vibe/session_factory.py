"""Session construction and backend pooling utilities."""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import logging
import platform
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from vibe.backends.base import (
    Backend,
    ExecutionPlan,
    ExecutionPreference,
    HardwareIntent,
    ModelPlugin,
    ModelVariant,
    RuntimeExecutor,
)
from vibe.contracts import verify_plugin_contract
from vibe.exceptions import LoaderError, PluginContractError, SessionError
from vibe.hf_downloader import get_auto_download_default
from vibe.loader import resolve_variant_artifacts
from vibe.precision import PrecisionPolicy, PrecisionRequest, parse_precision
from vibe.session import ModelSession

logger = logging.getLogger(__name__)

_RUNTIME_POOL_LOCK = threading.RLock()
_RUNTIME_POOL: dict[tuple[Any, ...], tuple[RuntimeExecutor, int]] = {}
_LOADING_LOCKS: dict[tuple[Any, ...], threading.Lock] = {}

# Known accelerated ONNX Runtime package distributions on PyPI
_ONNX_ACCELERATED_PACKAGES = (
    "onnxruntime-gpu",
    "onnxruntime-rocm",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "onnxruntime-cann",
    "onnxruntime-qnn",
)


def _onnx_runtime_capabilities() -> tuple[bool, bool]:
    """
    Check if ONNX Runtime is installed and whether GPU/hardware acceleration is available.
    Uses fast metadata inspection if not already loaded into memory.
    """
    # 1. Fast Path: Check without importing
    if "onnxruntime" not in sys.modules:
        if importlib.util.find_spec("onnxruntime") is None:
            return False, False

        # macOS Apple Silicon includes CoreML Execution Provider in standard package
        if sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64"):
            return True, True

        # Check if an accelerated build of ONNX Runtime is installed
        for pkg in _ONNX_ACCELERATED_PACKAGES:
            try:
                importlib.metadata.version(pkg)
                return True, True
            except Exception:
                pass

        return True, False

    # 2. Loaded Path: Inspect live providers if module is already imported
    try:
        import onnxruntime as ort  # ty:ignore[unresolved-import, unused-ignore-comment]

        get_providers = getattr(ort, "get_available_providers", None)
        available = {str(provider) for provider in get_providers()} if callable(get_providers) else set()
        has_accel = any(provider != "CPUExecutionProvider" for provider in available)
        return True, has_accel
    except Exception:
        return False, False


def _pytorch_runtime_capabilities() -> tuple[bool, bool]:
    """Check PyTorch and hardware accelerators statically without importing."""
    if "torch" in sys.modules:
        try:
            import torch

            has_cuda = bool(torch.cuda.is_available())
            xpu_mod = getattr(torch, "xpu", None)
            has_xpu = bool(xpu_mod and callable(getattr(xpu_mod, "is_available", None)) and xpu_mod.is_available())
            mps_backend = getattr(torch.backends, "mps", None)
            has_mps = bool(
                mps_backend and callable(getattr(mps_backend, "is_available", None)) and mps_backend.is_available()
            )
            return True, (has_cuda or has_xpu or has_mps)
        except Exception:
            return False, False

    if importlib.util.find_spec("torch") is None:
        return False, False

    # Static inspection matching system_info.py (0.5ms, zero imports)
    try:
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            v_py = Path(spec.origin).parent / "version.py"
            if v_py.exists():
                tree = ast.parse(v_py.read_text(encoding="utf-8"))
                var_map: dict[str, Any] = {}
                for stmt in tree.body:
                    if isinstance(stmt, ast.Assign):
                        for t in stmt.targets:
                            if isinstance(t, ast.Name):
                                var_map[t.id] = ast.literal_eval(stmt.value)
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        var_map[stmt.target.id] = ast.literal_eval(stmt.value)

                # Check if CUDA/ROCm/XPU is defined and non-None in version.py
                has_cuda = bool(var_map.get("cuda"))
                has_hip = bool(var_map.get("hip"))
                has_xpu = bool(var_map.get("xpu"))
                ver_str = str(var_map.get("__version__", ""))
                has_tag = any(k in ver_str.lower() for k in ("+cu", "+rocm", "+xpu"))

                return True, (has_cuda or has_hip or has_xpu or has_tag)
    except Exception:
        pass

    return True, False


def _resolve_backend_candidates(
    plugin_cls: type[ModelPlugin],
    requested_backend: Backend | str | None,
    preference: ExecutionPreference,
) -> set[Backend]:
    """Returns the set of backends to try based on hardware preference and availability."""
    supported = {v.backend for v in plugin_cls.variants}

    # Explicit backend requested by user
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
        return {selected}

    onnx_ok, onnx_accel = _onnx_runtime_capabilities()
    torch_ok, torch_accel = _pytorch_runtime_capabilities()

    # Map available backends to whether they have active accelerator support
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

    # Fail hard if an accelerator was explicitly requested but no GPU acceleration is active
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
            f"Backend capabilities: {', '.join(diag)}."
        )

    # If accelerator was requested or auto-resolved, prioritize GPU-accelerated backends over CPU
    if preference.intent in (HardwareIntent.AUTO, HardwareIntent.ACCELERATOR):
        accel_backends = {b for b, has_accel in available.items() if has_accel}
        if accel_backends:
            return accel_backends

    return set(available.keys())


def resolve_candidate_variants(
    plugin_cls: type[ModelPlugin],
    *,
    backend: Backend | str | None = None,
    variant: str | None = None,
    device: str = "auto",
) -> list[ModelVariant]:
    """Shared single source of truth for resolving variants to try."""
    model_id = plugin_cls.identity.model_id
    preference = ExecutionPreference.parse(device)

    if variant is not None:
        target_variant = next((v for v in plugin_cls.variants if v.variant_id == variant), None)
        if not target_variant:
            available_vars = [v.variant_id for v in plugin_cls.variants if v.variant_id]
            raise SessionError(f"Model '{model_id}' has no variant '{variant}'. Available variants: {available_vars}")

        if backend is not None:
            req_b = Backend(backend.lower()) if isinstance(backend, str) else backend
            if req_b != target_variant.backend:
                raise SessionError(
                    f"Variant '{variant}' requires backend '{target_variant.backend.value}', "
                    f"but backend '{req_b.value}' was requested."
                )
        return [target_variant]

    candidate_backends = _resolve_backend_candidates(plugin_cls, backend, preference)
    candidates = [v for v in plugin_cls.variants if v.backend in candidate_backends]

    # When both PyTorch and ONNX are candidates on "auto", prioritize PyTorch
    if backend is None and len(candidate_backends) > 1 and Backend.PYTORCH in candidate_backends:
        candidates.sort(key=lambda v: 0 if v.backend == Backend.PYTORCH else 1)

    return candidates


def resolve_variant(
    plugin_cls: type[ModelPlugin],
    *,
    backend: Backend | str | None = None,
    variant: str | None = None,
    device: str = "auto",
) -> ModelVariant:
    """Dry-run variant resolver that returns the exact ModelVariant vibe will select."""
    candidates = resolve_candidate_variants(plugin_cls, backend=backend, variant=variant, device=device)
    return candidates[0]


def build_session(
    plugin_cls: type[ModelPlugin],
    source: str | None = None,
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

    # 1. Resolve which variants to try using shared logic
    variants_to_try = resolve_candidate_variants(
        plugin_cls,
        backend=backend,
        variant=variant,
        device=device,
    )

    failures = []

    # 2. Try loading variants in declared priority order until one succeeds
    for selected_variant in variants_to_try:
        candidate_backend = selected_variant.backend

        # Resolve effective source for this variant:
        # User explicit source takes priority; otherwise use variant.repo_id or default_repo_id.
        if source is not None and source.strip():
            effective_source = source.strip()
        else:
            default_repo = selected_variant.repo_id or getattr(plugin_cls, "default_repo_id", "")
            if not default_repo:
                raise SessionError(
                    f"Model '{model_id}' variant '{selected_variant.variant_id}' has no default repository. "
                    "Provide source explicitly via source=..."
                )
            effective_source = f"hf:{default_repo}"

        try:
            file_map = resolve_variant_artifacts(
                source=effective_source,
                variant=selected_variant,
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
            plugin.load_ancillary(file_map)
            plugin.bind_execution_state(plan)

            # FAIL-FAST: Verify plugin contract immediately after loading attributes
            verify_plugin_contract(plugin)

            pool_key = (plugin_cls, file_map.cache_key, plan)
            runtime, release_fn = _acquire_runtime(
                key=pool_key,
                model_id=model_id,
                build=lambda p=plugin, fm=file_map, ep=plan: p.build_runtime(fm, ep),
            )
        except PluginContractError:
            # Do not swallow contract errors during fallback iterations
            raise
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
            source=effective_source,
            auto_download=effective_auto_download,
            memory_tracking=memory_tracking,
            backend_release=release_fn,
        )

    attempts = "; ".join(f"{b.value}: {reason}" for b, reason in failures)
    raise SessionError(f"Failed to resolve files or build runtime for '{model_id}'. Attempts: {attempts}")


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
            logger.debug("Released pooled runtime refcount=%s", cached[1] - 1)
            return

        popped = _RUNTIME_POOL.pop(key, None)
        instance = popped[0] if popped else None

    if instance is None:
        return

    try:
        instance.close()
    except Exception:
        logger.exception("Failed to close pooled runtime during release.")

    logger.debug("Closed pooled runtime")
