"""Session construction and backend pooling utilities."""

from __future__ import annotations
from vibe.precision import parse_precision

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from vibe.backends.base import ArtifactMap, Backend, FileRole, ModelPlugin, ModelVariant
from vibe.devices import normalize_device_string
from vibe.exceptions import SessionError
from vibe.hf_downloader import get_auto_download_default
from vibe.loader import resolve_variant_artifacts
from vibe.session import ModelSession

logger = logging.getLogger(__name__)

_BACKEND_POOL_LOCK = threading.RLock()
_BACKEND_POOL: dict[tuple[Any, ...], tuple[Any, int]] = {}


def build_session(
    plugin_cls: type[ModelPlugin],
    source: str,
    backend: Backend | str | None = None,
    device: str = "auto",
    precision: str = "auto",
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
            raise SessionError(f"Unknown backend '{backend}'. Choose from: {[b.value for b in Backend]}")
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
    source_is_unprefixed_local_dir = _is_unprefixed_local_dir_source(source)

    auto_resolution_failures: list[tuple[Backend, str]] = []

    resolve_plan: list[tuple[bool, bool]] = []
    if not backend_was_explicit and source_is_unprefixed_local_dir:
        # Phase 1: try local files only across backends, no HF fallback.
        resolve_plan.append((False, False))
        # Phase 2: if neither backend resolved locally, allow normal fallback/download behavior.
        resolve_plan.append((effective_auto_download, True))
        logger.info(
            "Auto backend source is local directory; trying local files across supported backends before HuggingFace fallback."
        )
    else:
        resolve_plan.append((effective_auto_download, True))

    for allow_download_for_attempt, allow_hf_fallback in resolve_plan:
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
                auto_download_attempt=allow_download_for_attempt,
                allow_hf_fallback=allow_hf_fallback,
                effective_auto_download=effective_auto_download,
                file_name_map=file_name_map,
                source_map=source_map,
                memory_tracking=memory_tracking,
                backend_candidates=backend_candidates,
                auto_resolution_failures=auto_resolution_failures,
            )
            if session is not None:
                return session

        if source_is_unprefixed_local_dir and not allow_hf_fallback:
            logger.info(
                "No local backend files resolved for model_id=%s; trying HuggingFace fallback for auto backend mode.",
                model_id,
            )

    if auto_resolution_failures:
        attempts = "; ".join([f"{candidate.value}: {reason}" for candidate, reason in auto_resolution_failures])
        raise SessionError(f"Could not resolve files for model '{model_id}' in auto backend mode. Attempts: {attempts}")

    raise SessionError(f"Failed to build session for model '{model_id}'.")


def _is_unprefixed_local_dir_source(source: str) -> bool:
    normalized = str(source).strip()
    if normalized.startswith("local:") or normalized.startswith("hf:"):
        return False
    return Path(normalized).expanduser().is_dir()


def _attempt_session_build(
    plugin_cls: type[ModelPlugin],
    source: str,
    candidate_backend: Backend,
    selected_backend: Backend,
    backend_was_explicit: bool,
    device: str,
    precision: str,
    onnx_providers: list[str] | None,
    hf_revision: str | None,
    hf_cache_dir: str | None,
    auto_download_attempt: bool,
    allow_hf_fallback: bool,
    effective_auto_download: bool,
    file_name_map: Mapping[str, str] | None,
    source_map: Mapping[str, str] | None,
    memory_tracking: bool,
    backend_candidates: list[Backend],
    auto_resolution_failures: list[tuple[Backend, str]],
) -> ModelSession | None:
    from vibe.backends.runtime.onnx import ONNXBackend
    from vibe.backends.runtime.pytorch import PyTorchBackend

    model_id = plugin_cls.identity.model_id

    try:
        normalized_device = normalize_device_string(
            device, backend="pytorch" if candidate_backend == Backend.PYTORCH else "onnx"
        )
    except ValueError as exc:
        raise SessionError(str(exc)) from exc

    if candidate_backend == Backend.PYTORCH and normalized_device == "auto":
        normalized_device = _auto_select_pytorch_device()
        logger.info("PyTorch device auto-selected: %s", normalized_device)

    try:
        normalized_precision = parse_precision(precision).value
    except ValueError as exc:
        raise SessionError(str(exc)) from exc

    if candidate_backend == Backend.PYTORCH and normalized_precision == "int8_ov":
        logger.warning(
            "Precision 'int8_ov'/'ov' is ONNX/OpenVINO-oriented and is not supported by PyTorch backend; falling back to auto."
        )
        normalized_precision = "auto"

    if not backend_was_explicit and candidate_backend != selected_backend:
        logger.info(
            "Auto backend selected %s for model_id=%s after %s was unavailable.",
            candidate_backend.value,
            model_id,
            selected_backend.value,
        )

    logger.debug(
        "Session backend selected model_id=%s backend=%s device=%s precision=%s",
        model_id,
        candidate_backend.value,
        normalized_device,
        normalized_precision,
    )

    variant = next((v for v in plugin_cls.variants if v.backend == candidate_backend), None)
    if variant is None:
        return None

    fallback_repo_id = plugin_cls.default_repo_id if allow_hf_fallback else None

    try:
        file_map = resolve_variant_artifacts(
            source=source,
            variant=variant,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            allow_download=auto_download_attempt,
            file_name_map=file_name_map,
            fallback_hf_repo_id=fallback_repo_id,
            source_map=source_map,
        )
    except Exception as exc:
        if backend_was_explicit or len(backend_candidates) == 1:
            raise SessionError(str(exc)) from exc

        next_backend = _next_backend_candidate(backend_candidates, candidate_backend)
        if next_backend is not None:
            logger.info(
                "Auto backend '%s' unavailable %s for model_id=%s; trying %s next.",
                candidate_backend.value,
                "locally" if not allow_hf_fallback else "",
                model_id,
                next_backend.value,
            )
        auto_resolution_failures.append((candidate_backend, str(exc)))
        return None

    weights_path = _find_weights(plugin_cls, variant, file_map)
    logger.debug("Resolved model weights for model_id=%s path=%s", model_id, weights_path)

    pool_key = _make_backend_pool_key(
        backend=candidate_backend,
        weights_path=weights_path,
        device=normalized_device,
        providers=onnx_providers,
        precision=normalized_precision,
    )
    rt, release_backend = _acquire_backend(
        key=pool_key,
        backend=candidate_backend,
        weights_path=weights_path,
        device=normalized_device,
        providers=onnx_providers,
        precision=normalized_precision,
        pytorch_cls=PyTorchBackend,
        onnx_cls=ONNXBackend,
    )

    try:
        plugin = plugin_cls()
        plugin.configure(
            auto_download=effective_auto_download,
            backend=candidate_backend,
            backend_instance=rt,
            device=normalized_device,
            precision=normalized_precision,
            source=source,
            optional_missing_files=file_map.optional_missing,
        )
        plugin.load_ancillary(file_map)
        if candidate_backend == Backend.ONNX and normalized_precision in {"fp16", "bf16"}:
            logger.warning(
                "Precision '%s' requested while running ONNX backend; runtime casting is provider/model dependent.",
                normalized_precision,
            )

        logger.debug("Session ready model_id=%s", model_id)
        return ModelSession(
            plugin=plugin,
            backend_instance=rt,
            backend=candidate_backend,
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
        raise SessionError(f"Plugin '{model_id}' failed to load ancillary files: {exc}") from exc


def _next_backend_candidate(candidates: list[Backend], current: Backend) -> Backend | None:
    try:
        index = candidates.index(current)
    except ValueError:
        return None
    if index + 1 >= len(candidates):
        return None
    return candidates[index + 1]


def _make_backend_pool_key(
    *,
    backend: Backend,
    weights_path: Path,
    device: str,
    providers: list[str] | None,
    precision: str,
) -> tuple[Any, ...]:
    resolved_path = str(weights_path.resolve())
    if backend == Backend.PYTORCH:
        return (backend.value, resolved_path, device, precision)
    provider_key = tuple(providers) if providers is not None else ("AUTO",)
    return (backend.value, resolved_path, provider_key, device, precision)


def _acquire_backend(
    *,
    key: tuple[Any, ...],
    backend: Backend,
    weights_path: Path,
    device: str,
    providers: list[str] | None,
    precision: str,
    pytorch_cls: Any,
    onnx_cls: Any,
) -> tuple[Any, Callable[[], None]]:
    with _BACKEND_POOL_LOCK:
        cached = _BACKEND_POOL.get(key)
        if cached is not None:
            instance, refcount = cached
            _BACKEND_POOL[key] = (instance, refcount + 1)
            logger.debug("Reusing pooled backend instance backend=%s refcount=%s", backend.value, refcount + 1)
            logger.debug("Reusing pooled backend key=%s", key)
            return instance, lambda: _release_backend(key)

        logger.debug("Loading backend backend=%s", backend.value)
        logger.debug(
            "Backend load options backend=%s device=%s precision=%s weights_path=%s",
            backend.value,
            device,
            precision,
            weights_path,
        )
        if backend == Backend.PYTORCH:
            instance = pytorch_cls()
            instance.load(weights_path, device=device, precision=precision)
        else:
            instance = onnx_cls()
            instance.load(weights_path, providers=providers, device=device, precision=precision)

        _BACKEND_POOL[key] = (instance, 1)
        logger.info("Backend ready backend=%s device=%s", backend.value, device)
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


def _auto_select_backend(plugin_cls: type[ModelPlugin], *, requested_device: str = "auto") -> Backend:
    """Select backend using runtime availability + accelerator preference."""
    supported = [v.backend for v in plugin_cls.variants]
    model_id = plugin_cls.identity.model_id

    onnx_runtime_ok, onnx_has_accel = _onnx_runtime_capabilities()
    torch_runtime_ok, torch_has_accel = _pytorch_runtime_capabilities()

    onnx_candidate = Backend.ONNX in supported and onnx_runtime_ok
    torch_candidate = Backend.PYTORCH in supported and torch_runtime_ok

    if onnx_candidate and torch_candidate:
        requested = str(requested_device or "auto").strip().lower()
        if requested in {"mps"}:
            logger.info("Backend auto-selection chose PyTorch due to requested device '%s'.", requested)
            return Backend.PYTORCH
        if requested in {"rocm", "dml"} or requested.startswith(("rocm:", "dml:")):
            logger.info("Backend auto-selection chose ONNX due to requested device '%s'.", requested)
            return Backend.ONNX

        if onnx_has_accel and not torch_has_accel:
            logger.info("Backend auto-selection chose ONNX (accelerator available, PyTorch accelerator unavailable).")
            return Backend.ONNX
        if torch_has_accel and not onnx_has_accel:
            logger.info("Backend auto-selection chose PyTorch (accelerator available, ONNX accelerator unavailable).")
            return Backend.PYTORCH

        logger.info("Backend auto-selection chose ONNX (default preference when capabilities are equivalent).")
        return Backend.ONNX

    if onnx_candidate:
        logger.info("Backend auto-selection chose ONNX (runtime available).")
        return Backend.ONNX
    if torch_candidate:
        logger.info("Backend auto-selection chose PyTorch (runtime available).")
        return Backend.PYTORCH

    raise SessionError(f"No supported backend available for '{model_id}'. Install onnxruntime or torch.")


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
    mps_backend = getattr(torch.backends, "mps", None)
    has_mps = False
    if mps_backend is not None and callable(getattr(mps_backend, "is_available", None)):
        has_mps = bool(mps_backend.is_available())

    return True, (has_cuda or has_mps)


def _auto_select_pytorch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if bool(torch.cuda.is_available()):
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and callable(getattr(mps_backend, "is_available", None)):
        if bool(mps_backend.is_available()):
            return "mps"

    return "cpu"


def _find_weights(
    plugin_cls: type[ModelPlugin],
    variant: ModelVariant,
    file_map: ArtifactMap,
) -> Path:
    """Find the weights file from the resolved ArtifactMap for the selected variant."""
    for spec in variant.artifacts:
        if spec.role == FileRole.WEIGHTS:
            path = file_map.get_optional(spec.id)
            if path is not None:
                return path

    raise SessionError(
        f"No weights file found for model '{plugin_cls.identity.model_id}' "
        f"with backend '{variant.backend.value}'. "
        f"Check the plugin's artifacts declaration."
    )
