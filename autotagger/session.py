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
from autotagger.results import InferenceResult, InferenceResultItem, ModelResult

logger = logging.getLogger(__name__)

# Log pillow_jxl availability at module load time for diagnostics
_has_pillow_jxl: bool
try:
    import pillow_jxl as _pjxl  # noqa: F401

    _has_pillow_jxl = True
except ImportError:
    _has_pillow_jxl = False
    logger.info(
        "pillow-jxl-plugin not installed; JPEG XL image format support is unavailable. "
        "Install 'pillow-jxl-plugin' package to enable JPEG XL support, e.g.: pip install pillow-jxl-plugin"
    )
    logger.debug("pillow-jxl-plugin import failed; JPEG XL support unavailable.", exc_info=True)


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
        images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
        processors: list[ResultProcessor] | None = None,
        *,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
    ) -> InferenceResult:
        """
        Run inference on one image or many images.

        Args:
                images: A single image/path, list of images/paths, or list of
                    (image_or_path, ref) tuples.
                    The plugin's preprocess() handles conversion.
            processors: Optional ordered list of result processor instances.
                        Processors run after model postprocess in the order given.
            batch_size: Batch chunk size used when true batching is selected.
            batch_method:
              - "auto": choose by backend/device (GPU-like -> true batching | CPU -> sequential).
              - "true": run stacked tensor batches when supported
              - "sequential": process one image at a time, like batch_size = 1

            CPU inference usually does not benefit from batch tensors and may be slower than simply batch size 1/sequential processing.

        Returns:
            InferenceResult with one item per input image.
            For single image input this still returns batch shape with one item.
        """
        if self._closed:
            raise SessionError("Session is closed. Load a new session before inferring.")
        if batch_size < 1:
            raise SessionError("batch_size must be >= 1")

        values, refs = self._normalize_input_format(images)
        normalized_images = self._load_images(values)
        if not normalized_images:
            return InferenceResult(total_inputs=0, items=[])

        method = self._resolve_batch_method(batch_method, batch_size)
        before = self._memory_tracker.snapshot() if self._memory_tracker.enabled else None

        if method == "sequential":
            raw_results = [self._infer_single(image, processors=processors) for image in normalized_images]
        else:
            raw_results = self._infer_many_true_batch(
                normalized_images,
                batch_size,
                processors=processors,
                fallback_to_sequential_on_stack_error=(batch_method == "auto"),
            )

        memory_record = None
        if before is not None:
            after = self._memory_tracker.snapshot()
            memory_record = self._memory_tracker.observe("infer", before, after)
            logger.debug(
                "Memory telemetry op=%s call=%s rss_delta=%s gpu_delta=%s",
                memory_record.operation,
                memory_record.index,
                memory_record.delta_process_rss_bytes,
                memory_record.delta_gpu_process_used_bytes,
            )

        # Input ordering is preserved end-to-end; each item's index/ref maps to
        # the exact input position used for preprocess, forward, and postprocess.
        items = [
            InferenceResultItem(
                index=index,
                input_ref=refs[index],
                result=result,
            )
            for index, result in enumerate(raw_results)
        ]
        return InferenceResult(
            total_inputs=len(normalized_images),
            items=items,
            memory=memory_record.to_dict() if memory_record is not None else None,
        )

    def _normalize_input_format(
        self,
        images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
    ) -> tuple[list[Any | str], list[Any]]:
        entries = images if isinstance(images, list) else [images]
        if not entries:
            return [], []

        has_tuple_items = any(isinstance(item, tuple) for item in entries)
        all_tuple_items = all(isinstance(item, tuple) for item in entries)
        if has_tuple_items and not all_tuple_items:
            raise SessionError(
                "Mixed input formats are not supported. Use either all bare images/paths or all (image_or_path, ref) tuples."
            )

        values: list[Any | str] = []
        refs: list[Any] = []
        if all_tuple_items:
            for i, item in enumerate(entries):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise SessionError(f"Tuple input at index {i} must be exactly (image_or_path, ref).")
                value, ref = item
                values.append(value)
                refs.append(ref)
            duplicates = self._find_duplicates(refs)
            if duplicates:
                duplicate_str = ", ".join(repr(value) for value in duplicates)
                raise SessionError(f"Explicit refs must be unique. Duplicate refs: {duplicate_str}")
        else:
            values = list(entries)
            refs = list(range(len(values)))

        return values, refs

    def _load_images(self, values: list[Any | str]) -> list[Any]:
        normalized_images = [self._load_image_if_path(value, index=i) for i, value in enumerate(values)]
        return normalized_images

    def _find_duplicates(self, values: list[Any]) -> list[Any]:
        seen_hashable: set[Any] = set()
        seen_unhashable: list[Any] = []
        duplicates: list[Any] = []
        duplicates_set: set[Any] = set()  # fast membership check for hashable dupes

        for value in values:
            try:
                is_seen = value in seen_hashable
            except TypeError:
                # Unhashable, fall back to linear scan
                if any(value == existing for existing in seen_unhashable):
                    if not any(value == existing for existing in duplicates):
                        duplicates.append(value)
                else:
                    seen_unhashable.append(value)
                continue

            if is_seen:
                if value not in duplicates_set:
                    duplicates.append(value)
                    duplicates_set.add(value)
            else:
                seen_hashable.add(value)

        return duplicates

    def _load_image_if_path(self, value: Any | str, *, index: int) -> Any:
        if not isinstance(value, (str, Path)):
            return value

        from PIL import Image

        path = Path(value)
        try:
            with Image.open(path) as img:
                return img.copy()
        except Exception as exc:
            suffix = Path(path).suffix.lower()
            hint = ""
            if suffix == ".jxl" and not _has_pillow_jxl:
                hint = " Install 'pillow-jxl-plugin' to enable JPEG XL support: pip install pillow-jxl-plugin"
            raise SessionError(f"Failed to load image at index {index} from path '{path}': {exc}.{hint}") from exc

    def _infer_single(
        self,
        image: Any,
        processors: list[ResultProcessor] | None = None,
    ) -> ModelResult:
        """Run inference for one image."""
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

        return self._apply_processors(result, processors=processors)

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

        if self._supports_true_batching():
            return "true"
        return "sequential"

    def _supports_true_batching(self) -> bool:
        supports_fn = getattr(self._backend_instance, "supports_true_batching", None)
        if callable(supports_fn):
            try:
                return bool(supports_fn())
            except Exception:
                logger.exception("Backend supports_true_batching() failed; using conservative fallback.")

        # Legacy fallback for third-party/custom backend instances.
        if self._backend == Backend.PYTORCH:
            device = str(getattr(self._backend_instance, "device", "cpu")).lower()
            return device != "cpu"

        providers = [str(p) for p in getattr(self._backend_instance, "providers", [])]
        return any(p.strip() and p.strip() != "CPUExecutionProvider" for p in providers)

    def _infer_many_true_batch(
        self,
        images: list[Any],
        batch_size: int,
        processors: list[ResultProcessor] | None = None,
        fallback_to_sequential_on_stack_error: bool = False,
    ) -> list[ModelResult]:
        results: list[ModelResult] = []
        for start in range(0, len(images), batch_size):
            raw_chunk = images[start : start + batch_size]
            try:
                chunk = [self._plugin.preprocess(image) for image in raw_chunk]
            except Exception as exc:
                raise SessionError(f"Preprocessing failed for model '{self.model_id}': {exc}") from exc

            try:
                batch_tensor = self._stack_batch(chunk)
            except SessionError:
                if fallback_to_sequential_on_stack_error:
                    for tensor in chunk:
                        try:
                            raw_output = self._backend_instance.run(tensor)
                        except Exception as exc:
                            raise SessionError(f"Inference failed for model '{self.model_id}': {exc}") from exc

                        try:
                            result = self._plugin.postprocess(raw_output)
                        except Exception as exc:
                            raise SessionError(f"Postprocessing failed for model '{self.model_id}': {exc}") from exc
                        results.append(self._apply_processors(result, processors=processors))
                    continue
                raise

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

        return results

    def _stack_batch(self, chunk: list[Any]) -> Any:
        first = chunk[0]
        if isinstance(first, np.ndarray):
            try:
                return np.concatenate(chunk, axis=0)
            except Exception as exc:
                raise SessionError(
                    "Could not build a true batch tensor. This usually means preprocessed "
                    "samples have incompatible shapes for concatenation. "
                    f"Details: {exc}"
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
        shape = getattr(raw_output, "shape", None)
        ndim = getattr(raw_output, "ndim", None)

        if ndim == 0:
            return [raw_output for _ in range(expected)]

        if shape is not None and len(shape) > 0 and shape[0] == expected:
            return [raw_output[i : i + 1] for i in range(expected)]

        if expected == 1:
            return [raw_output]

        try:
            arr = np.asarray(raw_output)
        except Exception as exc:
            raise SessionError(
                f"Backend output batch dimension mismatch: expected {expected}, got unknown output type."
            ) from exc

        if arr.ndim == 0:
            return [arr for _ in range(expected)]
        if arr.shape[0] == expected:
            return [arr[i : i + 1] for i in range(expected)]
        raise SessionError(f"Backend output batch dimension mismatch: expected {expected}, got {arr.shape}.")

    def _apply_processors(
        self,
        result: ModelResult,
        processors: list[ResultProcessor] | None = None,
    ) -> ModelResult:
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
