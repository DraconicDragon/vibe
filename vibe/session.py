"""
ModelSession — a loaded model, ready to run inference.

This is the object users interact with after calling vibe.load().
It holds the resolved plugin instance, the active runtime backend,
and optional result transforms. Calling .infer() is the one thing you do with it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from vibe import ModelResult
from vibe.backends.base import ArtifactMap, Backend, ExecutionPlan, ModelPlugin, RuntimeExecutor
from vibe.exceptions import InferenceCancelled, SessionError
from vibe.image_loading import (
    iter_load_images,
    normalize_input_format,
    should_prefetch_image_loading,
)
from vibe.memory_stats import MemoryTracker
from vibe.result_transforms import ResultTransform, TransformContext
from vibe.results import InferenceResult, InferenceResultItem
from vibe.runners import BatchRunner, InferenceEngine, SessionRunnerState
from vibe.transform_pipeline import TransformPipeline

logger = logging.getLogger(__name__)


_ASYNC_INFER_DONE = object()

# region ModelSession


class ModelSession:
    """
    A loaded model instance, ready for inference.
    """

    def __init__(
        self,
        plugin: ModelPlugin,
        backend_instance: RuntimeExecutor,
        plan: ExecutionPlan,
        file_map: ArtifactMap,
        source: str,
        auto_download: bool = True,
        memory_tracking: bool = False,
        backend_release: Callable[[], None] | None = None,
    ) -> None:
        self._plugin = plugin
        self._backend_instance = backend_instance
        self._plan = plan
        self._backend = plan.backend
        self._file_map = file_map
        self._source = source
        self._closed = False
        self._backend_release = backend_release
        self._memory_tracker = MemoryTracker(enabled=memory_tracking)

        self._state = SessionRunnerState(plugin.identity.model_id)

        self._transform_context = TransformContext(
            model_id=plugin.identity.model_id,
            artifacts=file_map,
            source=source,
            auto_download=auto_download,
            token=plan.hf_token,
            _plugin_data={type(d): d for d in plugin.provide_transform_data()},
        )
        self._pipeline = TransformPipeline(
            plugin.identity.model_id, self._transform_context, plugin.capabilities.transforms
        )
        self._engine = InferenceEngine(plugin, backend_instance, self._pipeline, self._state)
        self._runner = BatchRunner(self._engine, self._state, self._backend)

        logger.debug("Session created model_id=%s backend=%s", self.model_id, self._backend.value)
        logger.debug("Session memory_tracking=%s", self._memory_tracker.enabled)
        if not self._memory_tracker.enabled:
            logger.debug("Memory tracking disabled for model_id=%s", self.model_id)
        else:
            snap = self._memory_tracker.snapshot()
            if snap.process_rss_bytes is None:
                logger.debug("Memory tracking available with partial metrics for model_id=%s", self.model_id)
            else:
                logger.debug("Memory tracking enabled")
            if snap.gpu_process_used_bytes is None:
                logger.debug("GPU process memory metric unavailable (likely missing NVML/pynvml).")

    # region Primary Interface

    def execution_info(self) -> dict[str, Any]:
        """Return a diagnostic summary of the requested plan vs actual runtime state."""
        return {
            "model_id": self.model_id,
            "source": self.source,
            "plan": {
                "backend": self._plan.backend.value,
                "variant_id": self._plan.variant_id,
                "preference": self._plan.preference.intent.value,
                "precision": {
                    "weight": self._plan.precision.weight.value,
                    "compute": self._plan.precision.compute.value,
                },
            },
            "runtime": self._backend_instance.execution_info(),
        }

    def infer(
        self,
        images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
        transforms: list[ResultTransform] | None = None,
        *,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
        prefetch_batch_limit: int = 8,
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
                transforms=transforms,
                batch_size=batch_size,
                batch_method=batch_method,
                prefetch_batch_limit=prefetch_batch_limit,
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
        num_items = len(items)
        if total_inputs is not None and total_inputs > 1:
            logger.info(
                "Inference completed model_id=%s outputs=%s/%s batch_size=%s cancelled=%s",
                self.model_id,
                num_items,
                total_inputs,
                batch_size,
                bool(num_items < total_inputs),
            )
        else:
            logger.debug(
                "Inference completed model_id=%s outputs=%s batch_size=%s",
                self.model_id,
                num_items,
                batch_size,
            )

        return InferenceResult(
            total_inputs=total_inputs if total_inputs is not None else len(items),
            items=items,
            memory=memory,
        )

    def infer_batches(
        self,
        images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
        transforms: list[ResultTransform] | None = None,
        *,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
        prefetch_batch_limit: int = 8,
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
                values, _refs = normalize_input_format(images, error_cls=SessionError)
                if not values:
                    logger.warning("No input images provided for model_id=%s", self.model_id)
                    return

                # Delegated to the transforms pipeline
                self._pipeline.notify_infer_start(transforms)

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

                for chunk in iter_load_images(
                    images=images,
                    batch_size=batch_size,
                    prefetch_batch_limit=prefetch_batch_limit,
                    prefetch=prefetch_images,
                    cancel_check=self._state.check_cancelled,
                    error_cls=SessionError,
                ):
                    start = chunk.start_index
                    chunk_images = chunk.images
                    chunk_refs = chunk.refs

                    loaded_path_inputs += sum(
                        1 for i in range(start, start + len(chunk_images)) if isinstance(values[i], (str, Path))
                    )

                    if method == "sequential":
                        chunk_items = []
                        for i, img in enumerate(chunk_images):
                            self._state.check_cancelled()
                            result = self._engine.execute_single(img, transforms=transforms)
                            global_idx = start + i
                            chunk_items.append(
                                InferenceResultItem(index=global_idx, input_ref=chunk_refs[i], result=result)
                            )
                        logger.debug(
                            "Completed sequential chunk, current index=%s/%s", start + len(chunk_images), total_inputs
                        )
                    else:
                        chunk_results = self._runner.execute_chunk(
                            chunk_images, transforms=transforms, fallback_to_sequential=(batch_method == "auto")
                        )
                        chunk_items = []
                        for i, result in enumerate(chunk_results):
                            global_idx = start + i
                            chunk_items.append(
                                InferenceResultItem(index=global_idx, input_ref=chunk_refs[i], result=result)
                            )
                        logger.debug(
                            "Completed inference batch model_id=%s done=%s/%s",
                            self.model_id,
                            start + len(chunk_images),
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
                self._state.finish_run()
                logger.debug("Inference run finished model_id=%s", self.model_id)

    async def infer_async(
        self,
        images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
        transforms: list[ResultTransform] | None = None,
        *,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
        prefetch_batch_limit: int = 8,
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
                    transforms=transforms,
                    batch_size=batch_size,
                    batch_method=batch_method,
                    prefetch_batch_limit=prefetch_batch_limit,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                logger.debug("Async worker caught exception: %s", exc, exc_info=True)
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                except RuntimeError as loop_exc:
                    logger.debug("Event loop closed before async exception could be queued: %s", loop_exc)
            finally:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, _ASYNC_INFER_DONE)
                except RuntimeError as loop_exc:
                    logger.debug("Event loop closed before async completion signal could be queued: %s", loop_exc)

        # Daemon thread so pending async inference doesn't prevent interpreter shutdown
        thread = threading.Thread(target=_worker, name="vibe-infer-async", daemon=True)
        thread.start()

        try:
            while True:
                payload = await queue.get()
                if payload is _ASYNC_INFER_DONE:
                    break
                if isinstance(payload, Exception):
                    raise payload

                assert isinstance(payload, InferenceResult)
                yield payload
        except asyncio.CancelledError:
            self.cancel_current_inference()
            raise  # Exit immediately. Let the daemon thread die naturally in the background.

    def cancel_current_inference(self) -> bool:
        """
        Request cooperative cancellation of the currently running inference.
        """
        return self._state.cancel()

    def is_inference_running(self) -> bool:
        """Return whether an inference run is currently active."""
        return self._state.is_running

    def is_cancellation_requested(self) -> bool:
        """Return whether cancellation has been requested for the active run."""
        return self._state.is_cancellation_requested

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
        return self._plugin.identity.model_id

    @property
    def plugin(self) -> ModelPlugin:
        return self._plugin

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def source(self) -> str:
        return self._source

    # todo: describe/session info func for the session?

    def apply_transforms(self, result: ModelResult, transforms: list[ResultTransform]) -> ModelResult:
        """
        Manually apply a list of transforms to an existing result.
        """
        if self._closed:
            raise SessionError("Cannot apply transforms: Session is closed.")

        self._pipeline.notify_infer_start(transforms)
        return self._pipeline.apply(result, transforms)

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
                try:
                    self._backend_instance.close()
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

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception as exc:
            # Log teardown failures at debug level to avoid try-except-pass anti-pattern
            logger.debug("Ignored exception during session teardown in __del__: %s", exc)

    def __repr__(self) -> str:
        return f"ModelSession(model_id={self.model_id!r}, backend={self._backend.value!r})"


# endregion ModelSession
