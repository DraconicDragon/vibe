"""
ModelSession — a loaded model, ready to run inference.

This is the object users interact with after calling vibe.load().
It holds the resolved plugin instance, the active runtime backend,
and optional result processors. Calling .infer() is the one thing you do with it.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Literal

from vibe.backends.base import Backend, ModelPlugin
from vibe.exceptions import InferenceCancelled, SessionError
from vibe.image_loading import (
    iter_loaded_image_chunks,
    load_image_if_path,
    normalize_input_format,
    should_prefetch_image_loading,
)
from vibe.loader import FileMap
from vibe.memory_stats import MemoryTracker
from vibe.processor_pipeline import ProcessorPipeline
from vibe.result_processors import ResultProcessor, ResultProcessorContext
from vibe.results import InferenceResult, InferenceResultItem
from vibe.runners import BatchRunner, InferenceEngine, SessionRunnerState

logger = logging.getLogger(__name__)


def _fmt_shape(value: Any) -> Any:
    return getattr(value, "shape", None)


def _fmt_dtype(value: Any) -> Any:
    return getattr(value, "dtype", None)


_has_pillow_jxl: bool
try:
    import pillow_jxl as _pjxl  # noqa: F401

    _has_pillow_jxl = True
except ImportError:
    _has_pillow_jxl = False
    logger.info("pillow-jxl-plugin not installed; JPEG XL image format support is unavailable.")

_has_pillow_heif: bool
try:
    import pillow_heif as _pheif  # noqa: F401

    _has_pillow_heif = True
except ImportError:
    _has_pillow_heif = False
    logger.info("pillow-heif not installed; HEIF/HEIC image format support is unavailable.")


_ASYNC_INFER_DONE = object()

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
        self._backend = backend
        self._backend_instance = backend_instance
        self._file_map = file_map
        self._source = source
        self._closed = False
        self._backend_release = backend_release
        self._memory_tracker = MemoryTracker(enabled=memory_tracking)

        self._state = SessionRunnerState(plugin.model_id)

        # Lock and Event aliases for backwards-compatibility with unmodified methods
        self._inference_lock = self._state.lock
        self._run_state_lock = self._state.run_state_lock
        self._cancel_event = self._state.cancel_event

        self._processor_context = ResultProcessorContext(
            file_map=file_map,
            source=source,
            auto_download=auto_download,
        )
        self._pipeline = ProcessorPipeline(plugin.model_id, self._processor_context, plugin.supported_processors)
        self._engine = InferenceEngine(plugin, backend_instance, self._pipeline)
        self._runner = BatchRunner(self._engine, self._state, backend)

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
        """
        with self._state.lock:
            if self._closed:
                raise SessionError("Session is closed. Load a new session before inferring.")
            if batch_size < 1:
                raise SessionError("batch_size must be >= 1")

            self._state.start_run()
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
                method = self._runner.resolve_batch_method(batch_method, batch_size)
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
                        cancel_check=self._state.check_cancelled,
                        error_cls=SessionError,
                        has_pillow_jxl=_has_pillow_jxl,
                        has_pillow_heif=_has_pillow_heif,
                    )

                index = 0
                for start, chunk_images in iter_loaded_image_chunks(
                    values,
                    chunk_size=batch_size if method != "sequential" else 1,
                    use_prefetch=prefetch_images,
                    load_image_fn=_loader,
                    cancel_check=self._state.check_cancelled,
                ):
                    chunk_values = values[start : start + len(chunk_images)]
                    loaded_path_inputs += sum(1 for value in chunk_values if isinstance(value, (str, Path)))

                    if method == "sequential":
                        chunk_items = []
                        for img in chunk_images:
                            self._state.check_cancelled()
                            result = self._engine.execute_single(img, processors=result_processors)
                            chunk_items.append(InferenceResultItem(index=index, input_ref=refs[index], result=result))
                            index += 1
                        logger.debug("Completed sequential chunk, current index=%s/%s", index, total_inputs)
                    else:
                        chunk_results = self._runner.execute_chunk(
                            chunk_images, processors=result_processors, fallback_to_sequential=(batch_method == "auto")
                        )
                        chunk_items = []
                        for result in chunk_results:
                            chunk_items.append(InferenceResultItem(index=index, input_ref=refs[index], result=result))
                            index += 1
                        logger.debug(
                            "Completed inference batch model_id=%s done=%s/%s", self.model_id, index, total_inputs
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
                self._state.finish_run()
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
        """
        with self._state.run_state_lock:
            if not self._state.run_active:
                return False
        self._state.cancel_event.set()
        logger.warning("Cancellation requested for model_id=%s", self.model_id)
        return True

    def is_inference_running(self) -> bool:
        """Return whether an inference run is currently active."""
        with self._state.run_state_lock:
            return self._state.run_active

    def is_cancellation_requested(self) -> bool:
        """Return whether cancellation has been requested for the active run."""
        return self._state.cancel_event.is_set()

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
        with self._state.lock:
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
