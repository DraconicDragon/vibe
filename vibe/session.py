"""
ModelSession — a loaded model, ready to run inference.

This is the object users interact with after calling vibe.load().
It holds the resolved plugin instance, the active runtime backend,
and optional result processors. Calling .infer() is the one thing you do with it.

session = vibe.load("wd-eva02-large-v3")
result  = session.infer(image)
result  = session.infer(image, result_processors=[...])
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Literal

import numpy as np

from vibe.backends.base import Backend, ModelPlugin
from vibe.image_loading import (
    iter_loaded_image_chunks,
    load_image_if_path,
    normalize_input_format,
    should_prefetch_image_loading,
)
from vibe.loader import FileMap
from vibe.memory_stats import MemoryTracker
from vibe.result_processors import CleanTags, ResultProcessor, ResultProcessorContext
from vibe.results import InferenceResult, InferenceResultItem, ModelResult

logger = logging.getLogger(__name__)


def _fmt_shape(value: Any) -> Any:
    return getattr(value, "shape", None)


def _fmt_dtype(value: Any) -> Any:
    return getattr(value, "dtype", None)


# Log optional image format support at module load time for diagnostics
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

_has_pillow_heif: bool
try:
    import pillow_heif as _pheif  # noqa: F401

    _has_pillow_heif = True
except ImportError:
    _has_pillow_heif = False
    logger.info(
        "pillow-heif not installed; HEIF/HEIC image format support is unavailable. "
        "Install 'pillow-heif' package to enable HEIF/HEIC support, e.g.: pip install pillow-heif"
    )
    logger.debug("pillow-heif import failed; HEIF/HEIC support unavailable.", exc_info=True)


_ASYNC_INFER_DONE = object()


class SessionError(Exception):
    """Raised when session setup or inference fails."""


class InferenceCancelled(SessionError):
    """Raised when an in-flight inference run is cancelled by user request."""


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
        memory_tracking: bool = False,
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
        self._inference_lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._run_state_lock = threading.Lock()
        self._run_active = False
        logger.debug("Session created model_id=%s backend=%s", self.model_id, self._backend.value)
        logger.debug("Session memory_tracking=%s", self._memory_tracker.enabled)
        if not self._memory_tracker.enabled:
            logger.info("Memory tracking disabled by user for model_id=%s", self.model_id)
        else:
            snap = self._memory_tracker.snapshot()
            if snap.process_rss_bytes is None:
                logger.debug("Memory tracking available with partial metrics for model_id=%s", self.model_id)
            else:
                logger.info("Memory tracking enabled")
            if snap.gpu_process_used_bytes is None:
                logger.debug("GPU process memory metric unavailable (likely missing NVML/pynvml).")

    # region Primary Interface
    # todo: theres 3 entry points when there only need to be one, consider changing infer_batches() to internal function, mind infer_single() too
    def infer(
        self,
        images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
        result_processors: list[ResultProcessor] | None = None,
        *,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
        on_cancel: Literal["raise", "return_partial"] = "raise",
    ) -> InferenceResult:
        """
        Run inference on one image or many images.

        Args:
                images: A single image/path, list of images/paths, or list of
                    (image_or_path, ref) tuples.
                    The plugin's preprocess() handles conversion.
            result_processors: Optional ordered list of result processor instances.
                        Result processors run after model postprocess in the order given.
            batch_size: Batch chunk size used when true batching is selected.
            batch_method:
              - "auto": choose by backend/device (GPU-like -> true batching | CPU -> sequential).
              - "true": run stacked tensor batches when supported
              - "sequential": process one image at a time, like batch_size = 1
                        on_cancel:
                            - "raise": raise InferenceCancelled when cancellation is requested (default).
                            - "return_partial": return already processed items instead of raising.

            CPU inference usually does not benefit from batch tensors and may be slower than simply batch size 1/sequential processing.

        Returns:
            InferenceResult with one item per input image.
            For single image input this still returns batch shape with one item.
        """
        if on_cancel not in ("raise", "return_partial"):
            raise SessionError("on_cancel must be one of: 'raise', 'return_partial'")

        logger.info("Starting inference model_id=%s batch_size=%s", self.model_id, batch_size)
        logger.debug("Infer options batch_size=%s batch_method=%s on_cancel=%s", batch_size, batch_method, on_cancel)

        tracker_before_calls = self._memory_tracker.stats().inference_calls
        total_inputs: int | None = None
        items: list[InferenceResultItem] = []

        try:
            for chunk in self.infer_batches(
                images,
                result_processors=result_processors,
                batch_size=batch_size,
                batch_method=batch_method,
            ):
                total_inputs = chunk.total_inputs
                items.extend(chunk.items)
        except InferenceCancelled:
            if on_cancel == "raise":
                raise

        memory = self._last_memory_record_dict(
            operation="infer_batches",
            min_call_index=tracker_before_calls,
        )
        logger.info(
            "Inference completed model_id=%s outputs=%s batch_size=%s cancelled=%s",
            self.model_id,
            len(items),
            batch_size,
            bool(total_inputs is not None and len(items) < total_inputs),
        )
        return InferenceResult(
            total_inputs=total_inputs if total_inputs is not None else len(items),
            items=items,
            memory=memory,
        )

    def infer_batches(
        self,
        images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
        result_processors: list[ResultProcessor] | None = None,
        *,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
    ) -> Iterator[InferenceResult]:
        """
        Stream inference results as each completed chunk becomes available.

        Yields one InferenceResult per finished chunk. In sequential mode, each
        yielded chunk contains exactly one item.
        """
        with self._inference_lock:
            if self._closed:
                raise SessionError("Session is closed. Load a new session before inferring.")
            if batch_size < 1:
                raise SessionError("batch_size must be >= 1")
            self._start_run()

            before = self._memory_tracker.snapshot() if self._memory_tracker.enabled else None
            try:
                values, refs = normalize_input_format(images, error_cls=SessionError)
                if not values:
                    logger.warning("No input images provided for model_id=%s", self.model_id)
                    return

                self._notify_result_processors_infer_start(result_processors)

                total_inputs = len(values)
                path_inputs = sum(1 for value in values if isinstance(value, (str, Path)))
                loaded_path_inputs = 0
                method = self._resolve_batch_method(batch_method, batch_size)
                prefetch_images = should_prefetch_image_loading(path_inputs=path_inputs)
                if batch_size > 1 and method == "sequential" and batch_method != "sequential":
                    logger.warning(
                        "Batching disabled for model_id=%s backend=%s; using sequential processing",
                        self.model_id,
                        self._backend.value,
                    )
                logger.debug(
                    "Inference run prepared model_id=%s inputs=%s resolved_batch_method=%s",
                    self.model_id,
                    total_inputs,
                    method,
                )
                if path_inputs:
                    logger.info(
                        "Loading input images model_id=%s paths=%s total_inputs=%s",
                        self.model_id,
                        path_inputs,
                        total_inputs,
                    )
                    if prefetch_images:
                        logger.debug("Image prefetch enabled model_id=%s", self.model_id)

                def _loader(value: Any | str, index: int) -> Any:
                    return load_image_if_path(
                        value,
                        index=index,
                        cancel_check=self._check_cancelled,
                        error_cls=SessionError,
                        has_pillow_jxl=_has_pillow_jxl,
                        has_pillow_heif=_has_pillow_heif,
                    )

                if method == "sequential":
                    for start, chunk_images in iter_loaded_image_chunks(
                        values,
                        chunk_size=1,
                        use_prefetch=prefetch_images,
                        load_image_fn=_loader,
                        cancel_check=self._check_cancelled,
                    ):
                        index = start
                        value = values[index]
                        image = chunk_images[0]
                        if isinstance(value, (str, Path)):
                            loaded_path_inputs += 1
                        self._check_cancelled()
                        result = self._infer_single(image, result_processors=result_processors)
                        logger.debug("Completed sequential item index=%s/%s", index + 1, total_inputs)
                        yield InferenceResult(
                            total_inputs=total_inputs,
                            items=[InferenceResultItem(index=index, input_ref=refs[index], result=result)],
                        )
                else:
                    index = 0
                    for start, chunk_images in iter_loaded_image_chunks(
                        values,
                        chunk_size=batch_size,
                        use_prefetch=prefetch_images,
                        load_image_fn=_loader,
                        cancel_check=self._check_cancelled,
                    ):
                        chunk_values = values[start : start + len(chunk_images)]
                        loaded_path_inputs += sum(1 for value in chunk_values if isinstance(value, (str, Path)))
                        chunk_results = next(
                            self._infer_many_true_batch_chunks(
                                chunk_images,
                                len(chunk_images),
                                result_processors=result_processors,
                                fallback_to_sequential_on_stack_error=(batch_method == "auto"),
                            )
                        )
                        chunk_items: list[InferenceResultItem] = []
                        for result in chunk_results:
                            chunk_items.append(InferenceResultItem(index=index, input_ref=refs[index], result=result))
                            index += 1
                        self._check_cancelled()
                        logger.debug(
                            "Completed inference batch model_id=%s done=%s/%s",
                            self.model_id,
                            index,
                            total_inputs,
                        )
                        yield InferenceResult(total_inputs=total_inputs, items=chunk_items)

                if path_inputs:
                    logger.info(
                        "Loaded input images model_id=%s loaded=%s/%s",
                        self.model_id,
                        loaded_path_inputs,
                        path_inputs,
                    )
            finally:
                if before is not None:
                    after = self._memory_tracker.snapshot()
                    memory_record = self._memory_tracker.observe("infer_batches", before, after)
                    logger.debug(
                        "Memory telemetry op=%s call=%s rss_delta=%s gpu_delta=%s",
                        memory_record.operation,
                        memory_record.index,
                        memory_record.delta_process_rss_bytes,
                        memory_record.delta_gpu_process_used_bytes,
                    )
                self._finish_run()
                logger.debug("Inference run finished model_id=%s", self.model_id)

    async def infer_async(
        self,
        images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
        result_processors: list[ResultProcessor] | None = None,
        *,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
    ) -> AsyncIterator[InferenceResult]:
        """
        Async wrapper over infer_batches() for progressive consumption.

        Backend execution is still blocking; work is moved to a worker thread so
        callers can await chunk results as they complete.

        Accepts the same input forms as infer(), including a single image/path.
        Device selection is configured when the session is created via load().
        """
        import asyncio

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()

        def _worker() -> None:
            try:
                for chunk in self.infer_batches(
                    images,
                    result_processors=result_processors,
                    batch_size=batch_size,
                    batch_method=batch_method,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _ASYNC_INFER_DONE)

        thread = threading.Thread(target=_worker, name="vibe-infer-async", daemon=True)
        thread.start()

        while True:
            payload = await queue.get()
            if payload is _ASYNC_INFER_DONE:
                break
            if isinstance(payload, Exception):
                raise payload
            yield payload

    def cancel_current_inference(self) -> bool:
        """
        Request cooperative cancellation of the currently running inference.

        Returns True if a run was active and cancellation was requested, False
        if no inference run is currently active.
        """
        with self._run_state_lock:
            if not self._run_active:
                return False
        self._cancel_event.set()
        logger.warning("Cancellation requested for model_id=%s", self.model_id)
        return True

    def is_inference_running(self) -> bool:
        """Return whether an inference run is currently active."""
        with self._run_state_lock:
            return self._run_active

    def is_cancellation_requested(self) -> bool:
        """Return whether cancellation has been requested for the active run."""
        return self._cancel_event.is_set()

    def _start_run(self) -> None:
        with self._run_state_lock:
            self._run_active = True
            self._cancel_event.clear()
        logger.debug("Run state -> active for model_id=%s", self.model_id)

    def _finish_run(self) -> None:
        with self._run_state_lock:
            self._run_active = False
            self._cancel_event.clear()
        logger.debug("Run state -> idle for model_id=%s", self.model_id)

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            logger.warning("Inference cancelled before completing current step model_id=%s", self.model_id)
            raise InferenceCancelled("Inference cancelled by user request.")

    def _last_memory_record_dict(self, *, operation: str, min_call_index: int) -> dict[str, Any] | None:
        if not self._memory_tracker.enabled:
            return None
        stats = self._memory_tracker.stats()
        record = stats.last_record
        if record is None:
            return None
        if record.operation != operation:
            return None
        if record.index <= min_call_index:
            return None
        return record.to_dict()

    def _infer_single(
        self,
        image: Any,
        result_processors: list[ResultProcessor] | None = None,
    ) -> ModelResult:
        """Run inference for one image."""
        # Preprocess: image → tensor/array
        try:
            tensor = self._plugin.preprocess(image)
            logger.debug("Preprocess output shape=%s dtype=%s", _fmt_shape(tensor), _fmt_dtype(tensor))
        except Exception as exc:
            raise SessionError(f"Preprocessing failed for model '{self.model_id}': {exc}") from exc

        # Forward pass via runtime backend
        try:
            raw_output = self._backend_instance.run(tensor)
            logger.debug("Raw backend output shape=%s dtype=%s", _fmt_shape(raw_output), _fmt_dtype(raw_output))
        except Exception as exc:
            raise SessionError(f"Inference failed for model '{self.model_id}': {exc}") from exc

        # Postprocess: raw output → typed result
        try:
            result = self._plugin.postprocess(raw_output)
        except Exception as exc:
            raise SessionError(f"Postprocessing failed for model '{self.model_id}': {exc}") from exc

        return self._apply_processors(result, result_processors=result_processors)

    def _resolve_batch_method(
        self,
        requested: Literal["auto", "true", "sequential"],
        batch_size: int,
    ) -> Literal["true", "sequential"]:
        if batch_size <= 1:
            return "sequential"
        supports_true_batching = self._supports_true_batching()
        if requested == "true":
            if not supports_true_batching:
                logger.warning(
                    "Model_id=%s backend=%s does not support true batching; the run may fail if the export is batch-incompatible",
                    self.model_id,
                    self._backend.value,
                )
            return "true"
        if requested == "sequential":
            return "sequential"

        if supports_true_batching:
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

    def _infer_many_true_batch_chunks(
        self,
        images: list[Any],
        batch_size: int,
        result_processors: list[ResultProcessor] | None = None,
        fallback_to_sequential_on_stack_error: bool = False,
    ) -> Iterator[list[ModelResult]]:
        for start in range(0, len(images), batch_size):
            self._check_cancelled()
            raw_chunk = images[start : start + batch_size]
            try:
                chunk = [self._plugin.preprocess(image) for image in raw_chunk]
            except Exception as exc:
                raise SessionError(f"Preprocessing failed for model '{self.model_id}': {exc}") from exc

            chunk_results: list[ModelResult] = []
            try:
                batch_tensor = self._stack_batch(chunk)
            except SessionError:
                if fallback_to_sequential_on_stack_error:
                    for tensor in chunk:
                        self._check_cancelled()
                        try:
                            raw_output = self._backend_instance.run(tensor)
                        except Exception as exc:
                            raise SessionError(f"Inference failed for model '{self.model_id}': {exc}") from exc

                        try:
                            result = self._plugin.postprocess(raw_output)
                        except Exception as exc:
                            raise SessionError(f"Postprocessing failed for model '{self.model_id}': {exc}") from exc
                        chunk_results.append(self._apply_processors(result, result_processors=result_processors))
                    yield chunk_results
                    continue
                raise

            try:
                raw_output = self._backend_instance.run(batch_tensor)
            except Exception as exc:
                raise SessionError(f"Inference failed for model '{self.model_id}': {exc}") from exc

            for sample_output in self._split_batch_output(raw_output, len(chunk)):
                self._check_cancelled()
                try:
                    result = self._plugin.postprocess(sample_output)
                except Exception as exc:
                    raise SessionError(f"Postprocessing failed for model '{self.model_id}': {exc}") from exc
                chunk_results.append(self._apply_processors(result, result_processors=result_processors))
            yield chunk_results

    def _notify_result_processors_infer_start(self, result_processors: list[ResultProcessor] | None) -> None:
        if not result_processors:
            return
        for result_processor in result_processors:
            try:
                result_processor.on_infer_start(context=self._processor_context)
            except Exception as exc:
                raise SessionError(
                    f"Result processor '{result_processor.__class__.__name__}' failed during infer startup "
                    f"for model '{self.model_id}': {exc}"
                ) from exc

    def _stack_batch(self, chunk: list[Any]) -> Any:
        first = chunk[0]
        if self._is_structured_jtp3_batch(first):
            try:
                import torch

                from vibe.plugins.jtp_3.jtp3_modelplugin import JTP3Batch

                patches = torch.stack([item.patches for item in chunk], dim=0)
                patch_coords = torch.stack([item.patch_coords for item in chunk], dim=0)
                patch_valid = torch.stack([item.patch_valid for item in chunk], dim=0)
                logger.debug(
                    "Stacked JTP-3 batch batch_size=%d patches_shape=%s patch_coords_shape=%s patch_valid_shape=%s",
                    len(chunk),
                    patches.shape,
                    patch_coords.shape,
                    patch_valid.shape,
                )
                return JTP3Batch(patches, patch_coords, patch_valid)
            except Exception as exc:
                logger.error(
                    "Failed to stack JTP-3 batch for model_id=%s sample_descriptions=%s",
                    self.model_id,
                    [self._describe_preprocessed_sample(item) for item in chunk],
                )
                raise SessionError(
                    "Could not build a true JTP-3 batch. This usually means preprocessed "
                    "patch tensors have incompatible shapes for stacking. "
                    f"Details: {exc}"
                ) from exc

        if isinstance(first, np.ndarray):
            try:
                stacked = np.concatenate(chunk, axis=0)
                logger.debug("Stacked numpy batch shape=%s dtype=%s", stacked.shape, stacked.dtype)
                return stacked
            except Exception as exc:
                logger.error(
                    "Failed to stack numpy batch for model_id=%s sample_shapes=%s",
                    self.model_id,
                    [getattr(item, "shape", None) for item in chunk],
                )
                raise SessionError(
                    "Could not build a true batch tensor. This usually means preprocessed "
                    "samples have incompatible shapes for concatenation. "
                    f"Details: {exc}"
                ) from exc

        # Torch-like tensor handling without hard dependency.
        try:
            import torch

            if isinstance(first, torch.Tensor):
                stacked = torch.cat(chunk, dim=0)
                logger.debug("Stacked torch batch shape=%s dtype=%s", stacked.shape, stacked.dtype)
                return stacked
        except Exception:
            pass

        logger.error(
            "Unsupported preprocessed batch type for model_id=%s sample_descriptions=%s",
            self.model_id,
            [self._describe_preprocessed_sample(item) for item in chunk],
        )
        raise SessionError("Unsupported preprocessed tensor type for true batching. Use batch_method='sequential'.")

    def _describe_preprocessed_sample(self, item: Any) -> dict[str, Any]:
        shape = getattr(item, "shape", None)
        if shape is not None:
            return {
                "type": type(item).__name__,
                "shape": tuple(shape) if isinstance(shape, tuple) else shape,
                "dtype": str(getattr(item, "dtype", None)),
            }

        parts: dict[str, Any] = {"type": type(item).__name__}
        for field in ("patches", "patch_coords", "patch_valid"):
            value = getattr(item, field, None)
            if value is not None:
                parts[field] = {
                    "shape": tuple(getattr(value, "shape", ())) if getattr(value, "shape", None) is not None else None,
                    "dtype": str(getattr(value, "dtype", None)),
                }
        return parts

    def _is_structured_jtp3_batch(self, item: Any) -> bool:
        try:
            from vibe.plugins.jtp_3.jtp3_modelplugin import JTP3Batch
        except Exception:
            JTP3Batch = None  # ty:ignore[invalid-assignment]

        if JTP3Batch is not None and isinstance(item, JTP3Batch):
            return True

        return all(hasattr(item, field) for field in ("patches", "patch_coords", "patch_valid"))

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
        logger.error(
            "Backend output batch mismatch model_id=%s expected=%s actual_shape=%s",
            self.model_id,
            expected,
            arr.shape,
        )
        raise SessionError(f"Backend output batch dimension mismatch: expected {expected}, got {arr.shape}.")

    def _apply_processors(
        self,
        result: ModelResult,
        result_processors: list[ResultProcessor] | None = None,
    ) -> ModelResult:
        if not result_processors:
            return result

        # Pull CleanTags to the end if present, regardless of input order.
        effective_processors: list[ResultProcessor] = []
        cleanup_processors: list[CleanTags] = []

        for rp in result_processors:
            if isinstance(rp, CleanTags):
                cleanup_processors.append(rp)
            else:
                effective_processors.append(rp)

        effective_processors.extend(cleanup_processors)

        current = result
        for result_processor in effective_processors:
            if not any(isinstance(result_processor, supported) for supported in self._plugin.supported_processors):
                proc_name = result_processor.__class__.__name__
                self._processor_context.warn_once(
                    f"unsupported-processor:{proc_name}",
                    f"Processor '{proc_name}' is not declared as supported by model '{self.model_id}'; "
                    "attempting to apply anyway.",
                )

            try:
                current = result_processor.process(
                    current,
                    context=self._processor_context,
                )
            except Exception as exc:
                raise SessionError(
                    f"Result processor '{result_processor.__class__.__name__}' failed for model '{self.model_id}': {exc}"
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
        with self._inference_lock:
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
