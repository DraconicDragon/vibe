"""
ModelSession — a loaded model, ready to run inference.

This is the object users interact with after calling autotagger.load().
It holds the resolved plugin instance, the active runtime backend,
and optional result processors. Calling .infer() is the one thing you do with it.

session = autotagger.load("wd-eva02-large")
result  = session.infer(image)
result  = session.infer(image, processors=[...])
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from autotagger.backends.base import Backend, ModelPlugin
from autotagger.devices import normalize_device_string
from autotagger.hf_downloader import get_auto_download_default
from autotagger.loader import FileMap, resolve_from_source_string
from autotagger.memory_stats import MemoryTracker
from autotagger.result_processors import ResultProcessor, ResultProcessorContext
from autotagger.results import InferenceResult

logger = logging.getLogger(__name__)


_BACKEND_POOL_LOCK = threading.RLock()
_BACKEND_POOL: dict[tuple[Any, ...], tuple[Any, int]] = {}


class SessionError(Exception):
    """Raised when session setup or inference fails."""


# region ModelSession


class ModelSession:
    """
    A loaded model instance, ready for inference.

    Attributes:
        plugin:     The plugin instance (holds ancillary data like tag lists).
        backend:    Which runtime backend is active ("pytorch" or "onnx").
        model_id:   Canonical model ID from the plugin.
        source:     Where the files came from (for debugging/display).
    """

    def __init__(
        self,
        plugin: ModelPlugin,
        backend_instance: Any,
        backend: Backend,
        file_map: FileMap,
        source: str,
        auto_download: bool = True,
        memory_tracking: bool = True,
        backend_release: Callable[[], None] | None = None,
    ) -> None:
        self._plugin = plugin
        self._backend_instance = backend_instance
        self._backend = backend
        self._file_map = file_map
        self._source = source
        self._closed = False
        self._backend_release = backend_release
        self._memory_tracker = MemoryTracker(enabled=memory_tracking)
        self._processor_context = ResultProcessorContext(
            file_map=file_map,
            source=source,
            auto_download=auto_download,
        )

    # region Primary Interface

    def infer(
        self,
        image: Any,
        processors: list[ResultProcessor] | None = None,
    ) -> InferenceResult:
        """
        Run inference on an image.

        Args:
            image:  A PIL.Image.Image, or a numpy array (H×W×C uint8).
                    The plugin's preprocess() handles conversion.
            processors: Optional ordered list of result processor instances.
                        Processors run after model postprocess in the order given.

        Returns:
            A TagResult, ScoreResult, or MultiScoreResult depending on the model.
        """
        return self._infer_validated(image, processors=processors)

    def _infer_validated(
        self,
        image: Any,
        processors: list[ResultProcessor] | None = None,
    ) -> InferenceResult:
        """Run inference for one image."""
        if self._closed:
            raise SessionError("Session is closed. Load a new session before inferring.")

        before = self._memory_tracker.snapshot() if self._memory_tracker.enabled else None

        # Preprocess: image → tensor/array
        try:
            tensor = self._plugin.preprocess(image)
        except Exception as exc:
            raise SessionError(f"Preprocessing failed for model '{self.model_id}': {exc}") from exc

        # Forward pass via runtime backend
        try:
            raw_output = self._backend_instance.run(tensor)
        except Exception as exc:
            raise SessionError(f"Inference failed for model '{self.model_id}': {exc}") from exc

        # Postprocess: raw output → typed result
        try:
            result = self._plugin.postprocess(raw_output)
        except Exception as exc:
            raise SessionError(f"Postprocessing failed for model '{self.model_id}': {exc}") from exc

        output = self._apply_processors(result, processors=processors)

        if self._memory_tracker.enabled and before is not None:
            after = self._memory_tracker.snapshot()
            record = self._memory_tracker.observe("infer", before, after)
            logger.debug(
                "Memory telemetry op=%s call=%s rss_delta=%s gpu_delta=%s",
                record.operation,
                record.index,
                record.delta_process_rss_bytes,
                record.delta_gpu_process_used_bytes,
            )

        return output

    def infer_many(
        self,
        images: list[Any],
        processors: list[ResultProcessor] | None = None,
        *,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
    ) -> list[InferenceResult]:
        """
        Run inference over multiple images.

        batch_method:
          - "auto": choose by backend/device
          - "true": run stacked tensor batches when supported
          - "sequential": process one image at a time
        """
        if batch_size < 1:
            raise SessionError("batch_size must be >= 1")
        if not images:
            return []

        method = self._resolve_batch_method(batch_method, batch_size)
        if method == "sequential":
            return [self._infer_validated(image, processors=processors) for image in images]

        return self._infer_many_true_batch(images, batch_size, processors=processors)

    def _resolve_batch_method(
        self,
        requested: Literal["auto", "true", "sequential"],
        batch_size: int,
    ) -> Literal["true", "sequential"]:
        if batch_size <= 1:
            return "sequential"
        if requested == "true":
            return "true"
        if requested == "sequential":
            return "sequential"

        # Auto strategy: true batching on GPU-like contexts, sequential on CPU.
        if self._backend == Backend.PYTORCH:
            device = str(getattr(self._backend_instance, "device", "cpu")).lower()
            return "true" if device != "cpu" else "sequential"

        providers = [str(p) for p in getattr(self._backend_instance, "providers", [])]
        has_accelerator_provider = any(p.strip() and p.strip() != "CPUExecutionProvider" for p in providers)
        return "true" if has_accelerator_provider else "sequential"

    def _infer_many_true_batch(
        self,
        images: list[Any],
        batch_size: int,
        processors: list[ResultProcessor] | None = None,
    ) -> list[InferenceResult]:
        if self._closed:
            raise SessionError("Session is closed. Load a new session before inferring.")

        before = self._memory_tracker.snapshot() if self._memory_tracker.enabled else None
        results: list[InferenceResult] = []
        for start in range(0, len(images), batch_size):
            raw_chunk = images[start : start + batch_size]
            try:
                chunk = [self._plugin.preprocess(image) for image in raw_chunk]
            except Exception as exc:
                raise SessionError(f"Preprocessing failed for model '{self.model_id}': {exc}") from exc

            batch_tensor = self._stack_batch(chunk)
            try:
                raw_output = self._backend_instance.run(batch_tensor)
            except Exception as exc:
                raise SessionError(f"Inference failed for model '{self.model_id}': {exc}") from exc

            for sample_output in self._split_batch_output(raw_output, len(chunk)):
                try:
                    result = self._plugin.postprocess(sample_output)
                except Exception as exc:
                    raise SessionError(f"Postprocessing failed for model '{self.model_id}': {exc}") from exc
                results.append(self._apply_processors(result, processors=processors))

        if self._memory_tracker.enabled and before is not None:
            after = self._memory_tracker.snapshot()
            record = self._memory_tracker.observe("infer_many", before, after)
            logger.debug(
                "Memory telemetry op=%s call=%s rss_delta=%s gpu_delta=%s",
                record.operation,
                record.index,
                record.delta_process_rss_bytes,
                record.delta_gpu_process_used_bytes,
            )

        return results

    def _stack_batch(self, chunk: list[Any]) -> Any:
        first = chunk[0]
        if isinstance(first, np.ndarray):
            try:
                return np.concatenate(chunk, axis=0)
            except Exception as exc:
                raise SessionError(
                    "Could not build true batch tensor for numpy backend output. "
                    f"Falling back to sequential is recommended: {exc}"
                ) from exc

        # Torch-like tensor handling without hard dependency.
        try:
            import torch

            if isinstance(first, torch.Tensor):
                return torch.cat(chunk, dim=0)
        except Exception:
            pass

        raise SessionError("Unsupported preprocessed tensor type for true batching. Use batch_method='sequential'.")

    def _split_batch_output(self, raw_output: Any, expected: int) -> list[Any]:
        arr = np.asarray(raw_output)
        if arr.ndim == 0:
            return [arr for _ in range(expected)]
        if arr.shape[0] == expected:
            return [arr[i : i + 1] for i in range(expected)]
        if expected == 1:
            return [arr]
        raise SessionError(f"Backend output batch dimension mismatch: expected {expected}, got {arr.shape}.")

    def _apply_processors(
        self,
        result: InferenceResult,
        processors: list[ResultProcessor] | None = None,
    ) -> InferenceResult:
        if not processors:
            return result

        current = result
        for processor in processors:
            if not any(isinstance(processor, supported) for supported in self._plugin.supported_processors):
                logger.warning(
                    "Processor '%s' is not declared as supported by model '%s'; attempting to apply anyway.",
                    processor.__class__.__name__,
                    self.model_id,
                )

            try:
                current = processor.process(
                    current,
                    context=self._processor_context,
                )
            except Exception as exc:
                raise SessionError(
                    f"Result processor '{processor.__class__.__name__}' failed for model '{self.model_id}': {exc}"
                ) from exc
        return current

    # endregion Primary Interface

    # region Introspection

    @property
    def model_id(self) -> str:
        return self._plugin.model_id

    @property
    def plugin(self) -> ModelPlugin:
        return self._plugin

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def source(self) -> str:
        return self._source

    def describe(self) -> dict[str, Any]:
        # todo: rename this func?
        """Full description of this session (model_id, display_name, backend, source, output_type)."""
        return {
            "model_id": self.model_id,
            "display_name": self._plugin.display_name,
            "backend": self._backend.value,
            "source": self._source,
            "output_type": self._plugin.output_type.value,
        }

    def close(self) -> None:
        """Release runtime resources for this session."""
        if self._closed:
            return

        logger.debug("Closing session model_id=%s backend=%s", self.model_id, self._backend.value)

        if self._backend_release is not None:
            try:
                self._backend_release()
            except Exception:
                logger.exception("Failed to release pooled backend for model '%s'.", self.model_id)
        else:
            close_fn = getattr(self._backend_instance, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.exception("Failed to close backend for model '%s'.", self.model_id)

        self._closed = True
        logger.debug("Session closed model_id=%s backend=%s", self.model_id, self._backend.value)

    def is_closed(self) -> bool:
        """Return whether this session has been closed."""
        return self._closed

    def set_memory_tracking(self, enabled: bool) -> None:
        """Enable or disable per-call memory tracking."""
        self._memory_tracker.enabled = bool(enabled)

    def memory_stats(self) -> dict[str, Any]:
        """Return current memory telemetry stats for this session."""
        return self._memory_tracker.stats().to_dict()

    def memory_snapshot(self) -> dict[str, Any]:
        """Return an immediate memory snapshot."""
        return self._memory_tracker.snapshot().to_dict()

    def reset_memory_stats(self) -> None:
        """Clear aggregated session memory telemetry counters."""
        self._memory_tracker.reset()

    def __enter__(self) -> ModelSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception:
            # Avoid noisy teardown failures at interpreter shutdown.
            pass

    def __repr__(self) -> str:
        return f"ModelSession(model_id={self.model_id!r}, backend={self._backend.value!r})"


# endregion ModelSession


# region Session Factory


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

    This is called by autotagger.load() — you don't usually call this directly.

    Args:
        plugin_cls:      The plugin class (from the registry).
        source:          Where to load files from (source string).
        backend:         "pytorch", "onnx", or Backend enum. None = auto-select:
                         prefers ONNX if available, falls back to PyTorch.
        device:          Logical device string ("cpu", "gpu", "gpu:1", "cuda:0", etc.).
                 For ONNX, this guides auto provider selection. 'cuda' and 'gpu' are interchangeable.
        onnx_providers:  Override ONNX execution providers.
    """
    from autotagger.backends.runtime.onnx import ONNXBackend
    from autotagger.backends.runtime.pytorch import PyTorchBackend

    # Resolve backend
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

    # Resolve files
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

    # Find the weights file for this backend
    weights_path = _find_weights(plugin_cls, file_map, backend)

    # Load or reuse a pooled runtime backend.
    input_name = plugin_cls().get_input_name() if backend == Backend.ONNX else None
    pool_key = _make_backend_pool_key(
        backend=backend,
        weights_path=weights_path,
        device=normalized_device,
        providers=onnx_providers,
        input_name=input_name,
    )
    rt, release_backend = _acquire_backend(
        key=pool_key,
        backend=backend,
        weights_path=weights_path,
        device=normalized_device,
        providers=onnx_providers,
        input_name=input_name,
        pytorch_cls=PyTorchBackend,
        onnx_cls=ONNXBackend,
    )

    # Instantiate plugin and let it load ancillary files (tag lists, etc.)
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
    input_name: str | None,
) -> tuple[Any, ...]:
    resolved_path = str(weights_path.resolve())
    if backend == Backend.PYTORCH:
        return (backend.value, resolved_path, device)
    provider_key = tuple(providers) if providers is not None else ("AUTO",)
    return (backend.value, resolved_path, provider_key, device, input_name or "")


def _acquire_backend(
    *,
    key: tuple[Any, ...],
    backend: Backend,
    weights_path: Path,
    device: str,
    providers: list[str] | None,
    input_name: str | None,
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
            instance.load(weights_path, providers=providers, input_name=input_name, device=device)

        _BACKEND_POOL[key] = (instance, 1)
        logger.debug("Created pooled backend key=%s refcount=1", key)
        return instance, lambda: _release_backend(key)


def _release_backend(key: tuple[Any, ...]) -> None:
    with _BACKEND_POOL_LOCK:
        cached = _BACKEND_POOL.get(key)
        if cached is None:
            return

        instance, refcount = cached
        if refcount > 1:
            _BACKEND_POOL[key] = (instance, refcount - 1)
            logger.debug("Released pooled backend key=%s refcount=%s", key, refcount - 1)
            return

        _BACKEND_POOL.pop(key, None)

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
    """
    Find the weights file from the resolved file map.

    Looks for a FileSpec with role=WEIGHTS that is needed for this backend.
    Raises SessionError if not found.
    """
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


# endregion Session Factory
