"""
ModelSession — a loaded model, ready to run inference.

This is the primary object users interact with after calling vibe.load().
It holds the resolved plugin instance and the active runtime backend.
Calling .infer() runs the model forward pass.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Callable, Generator, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any, Literal, Self, TypeVar

from vibe.backends.base import ArtifactMap, Backend, ExecutionPlan, ModelPlugin, RuntimeExecutor
from vibe.contracts import (
    CalibrationProvider,
    LabelCatalogProvider,
    ScorerView,
    TaggerView,
    ThresholdProvider,
    _get_protocol_required_attrs,
)
from vibe.exceptions import InferenceCancelled, SessionCapabilityError, SessionError
from vibe.image_loading import (
    iter_load_normalized,
    normalize_input_format,
)
from vibe.memory_stats import MemoryTracker
from vibe.metadata import ModelDescriptor, OutputKind, StandardConsumerSettingId, TagFilterRecommendation
from vibe.results import InferenceResult, InferenceResultItem
from vibe.runners import BatchRunner, InferenceEngine, SessionRunnerState
from vibe.settings import InferenceRequest, OptionScope, compile_settings

logger = logging.getLogger(__name__)

_ASYNC_INFER_DONE = object()
T = TypeVar("T")


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
        self._auto_download = auto_download
        self._closed = False
        self._backend_release = backend_release
        self._memory_tracker = MemoryTracker(enabled=memory_tracking)
        self._state = SessionRunnerState(plugin.identity.model_id)

        # Pre-cache static descriptor
        self._metadata: ModelDescriptor = plugin.describe()

        self._engine = InferenceEngine(plugin, backend_instance, self._state)
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
            "plan": self._plan.to_dict(),
            "runtime": self._backend_instance.execution_info(),
        }

    def infer(
        self,
        images: Any | str | Sequence[Any] | Mapping[Any, Any],
        *,
        refs: Sequence[Any] | Iterable[Any] | None = None,
        settings: Mapping[str, Mapping[str, Any]] | Mapping[str, Any] | Sequence[Any] | InferenceRequest | None = None,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
        prefetch_batch_limit: int = 8,
        on_cancel: Literal["raise", "return_partial"] = "raise",
    ) -> InferenceResult:
        """Run inference on one image, a collection of images, or a streaming generator."""
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
                refs=refs,
                settings=settings,
                batch_size=batch_size,
                batch_method=batch_method,
                prefetch_batch_limit=prefetch_batch_limit,
            ):
                total_inputs = chunk.total_inputs
                items.extend(chunk.items)
        except InferenceCancelled as exc:
            if on_cancel == "raise":
                memory = self._last_memory_record_dict(
                    operation="infer_batches",
                    min_call_index=tracker_before_calls,
                )
                exc.partial_result = InferenceResult(
                    total_inputs=total_inputs if total_inputs is not None else len(items),
                    items=items,
                    memory=memory,
                )
                raise

        memory = self._last_memory_record_dict(
            operation="infer_batches",
            min_call_index=tracker_before_calls,
        )
        num_items = len(items)
        is_cancelled = bool(total_inputs is not None and num_items < total_inputs)

        if total_inputs is not None and total_inputs > 1:
            logger.info(
                "Inference completed model_id=%s outputs=%s/%s batch_size=%s cancelled=%s",
                self.model_id,
                num_items,
                total_inputs,
                batch_size,
                is_cancelled,
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
        images: Any | str | Sequence[Any] | Mapping[Any, Any],
        *,
        refs: Sequence[Any] | Iterable[Any] | None = None,
        settings: Mapping[str, Mapping[str, Any]] | Mapping[str, Any] | Sequence[Any] | InferenceRequest | None = None,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
        prefetch_batch_limit: int = 8,
    ) -> Generator[InferenceResult, None, None]:
        """Stream inference results as each completed chunk becomes available."""
        with self._state.lock:
            if self._closed:
                raise SessionError("Session is closed. Load a new session before inferring.")
            if batch_size < 1:
                raise SessionError("batch_size must be >= 1")

            self._state.start_run()
            before = self._memory_tracker.snapshot() if self._memory_tracker.enabled else None

            try:
                norm = normalize_input_format(images, refs=refs, error_cls=SessionError)
                inference_request = compile_settings(
                    settings, self._plugin.settings, allowed_scopes={OptionScope.INFER}
                )
                method = self._runner.resolve_batch_method(batch_method, batch_size)
                total_inputs = norm.total

                if batch_size > 1 and method == "sequential" and batch_method != "sequential":
                    logger.info(
                        "Auto-batching using sequential mode for model_id=%s backend=%s (accelerator not active or graph has fixed batch dimension)",
                        self.model_id,
                        self._backend.value,
                    )

                if total_inputs is not None:
                    logger.debug(
                        "Inference run prepared model_id=%s inputs=%s resolved_batch_method=%s",
                        self.model_id,
                        total_inputs,
                        method,
                    )
                else:
                    logger.debug(
                        "Inference run prepared (streaming) model_id=%s resolved_batch_method=%s",
                        self.model_id,
                        method,
                    )

                chunk_gen = iter_load_normalized(
                    norm=norm,
                    batch_size=batch_size,
                    prefetch_batch_limit=prefetch_batch_limit,
                    cancel_check=self._state.check_cancelled,
                    error_cls=SessionError,
                )

                try:
                    for chunk in chunk_gen:
                        start = chunk.start_index
                        chunk_images = chunk.images
                        chunk_refs = chunk.refs

                        if method == "sequential":
                            chunk_items = []
                            try:
                                for i, img in enumerate(chunk_images):
                                    self._state.check_cancelled()
                                    result = self._engine.execute_single(img, request=inference_request)
                                    global_idx = start + i
                                    chunk_items.append(
                                        InferenceResultItem(index=global_idx, input_ref=chunk_refs[i], result=result)
                                    )
                            except InferenceCancelled:
                                # Yield any completed items before re-raising cancellation!
                                if chunk_items:
                                    yield InferenceResult(total_inputs=total_inputs, items=chunk_items)
                                raise
                        else:
                            chunk_results = self._runner.execute_chunk(
                                chunk_images,
                                request=inference_request,
                                fallback_to_sequential=(batch_method == "auto"),
                            )
                            chunk_items = []
                            for i, result in enumerate(chunk_results):
                                global_idx = start + i
                                chunk_items.append(
                                    InferenceResultItem(index=global_idx, input_ref=chunk_refs[i], result=result)
                                )

                        if total_inputs is not None:
                            logger.debug(
                                "Completed inference batch model_id=%s done=%s/%s",
                                self.model_id,
                                start + len(chunk_images),
                                total_inputs,
                            )
                        else:
                            logger.debug(
                                "Completed inference batch model_id=%s done=%s (streaming)",
                                self.model_id,
                                start + len(chunk_images),
                            )

                        yield InferenceResult(total_inputs=total_inputs, items=chunk_items)
                finally:
                    chunk_gen.close()
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
        images: Any | str | Sequence[Any] | Mapping[Any, Any],
        *,
        refs: Sequence[Any] | None = None,
        settings: Mapping[str, Mapping[str, Any]] | Mapping[str, Any] | Sequence[Any] | InferenceRequest | None = None,
        batch_size: int = 1,
        batch_method: Literal["auto", "true", "sequential"] = "auto",
        prefetch_batch_limit: int = 8,
    ) -> AsyncIterator[InferenceResult]:
        """
        Async wrapper over infer_batches() for progressive consumption.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()

        def _queue_from_worker(payload: object) -> bool:
            """Queue a worker payload while tolerating event-loop shutdown."""
            try:
                loop.call_soon_threadsafe(queue.put_nowait, payload)
            except RuntimeError as exc:
                logger.debug("Event loop closed before async payload could be queued: %s", exc)
                return False
            return True

        def _worker() -> None:
            batches = self.infer_batches(
                images,
                refs=refs,
                settings=settings,
                batch_size=batch_size,
                batch_method=batch_method,
                prefetch_batch_limit=prefetch_batch_limit,
            )
            try:
                for chunk in batches:
                    if self._state.is_cancellation_requested:
                        logger.debug("Async inference cancellation observed model_id=%s", self.model_id)
                        raise InferenceCancelled("Inference cancelled by user request.")
                    if not _queue_from_worker(chunk):
                        break
            except Exception as exc:
                logger.debug("Async worker caught exception: %s", exc, exc_info=True)
                _queue_from_worker(exc)
            finally:
                batches.close()
                _queue_from_worker(_ASYNC_INFER_DONE)

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
        finally:
            # Covers task cancellation, caller exceptions, and early async-generator
            # closure after an async-for break/return.
            if self.cancel_current_inference():
                logger.debug("Async inference cancellation requested model_id=%s", self.model_id)

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
        if record is None or record.operation != operation or record.index <= min_call_index:
            return None
        return record.to_dict()

    # endregion Primary Interface

    # region Introspection & Trusted Views

    @property
    def model_id(self) -> str:
        """The canonical string ID of this model."""
        return self._plugin.identity.model_id

    @property
    def plugin(self) -> ModelPlugin:
        """The loaded plugin instance."""
        return self._plugin

    @property
    def metadata(self) -> ModelDescriptor:
        """Structured metadata descriptor for this model."""
        return self._metadata

    @property
    def artifacts(self) -> ArtifactMap:
        """Resolved file artifacts for this session."""
        return self._file_map

    @property
    def is_tagger(self) -> bool:
        """Return True if the model produces tag outputs."""
        return self.metadata.output.kind == OutputKind.TAGS

    @property
    def is_scorer(self) -> bool:
        """Return True if the model produces score outputs."""
        return self.metadata.output.kind in (OutputKind.SCORE, OutputKind.MULTI_SCORE)

    @property
    def tagger(self) -> TaggerView:
        """Access tagger-specific domain data. Raises SessionCapabilityError if not a tagger."""
        plugin = self._plugin
        if not self.is_tagger or not isinstance(plugin, LabelCatalogProvider):
            raise SessionCapabilityError(f"Model '{self.model_id}' is not a tagger.")

        recommendation = None
        tag_filter_spec = self.metadata.get_consumer_setting(StandardConsumerSettingId.TAG_FILTER)
        if tag_filter_spec and isinstance(tag_filter_spec.recommended, TagFilterRecommendation):
            recommendation = tag_filter_spec.recommended

        thresholds = plugin.thresholds if isinstance(plugin, ThresholdProvider) else None

        return TaggerView(
            catalog=plugin.catalog,
            thresholds=thresholds,
            recommendation=recommendation,
        )

    @property
    def scorer(self) -> ScorerView:
        """Access scorer-specific domain data. Raises SessionCapabilityError if not a scorer."""
        plugin = self._plugin
        if not self.is_scorer:
            raise SessionCapabilityError(f"Model '{self.model_id}' is not a scorer.")

        catalog = plugin.catalog if isinstance(plugin, LabelCatalogProvider) else None
        calibration = plugin.calibration if isinstance(plugin, CalibrationProvider) else None

        return ScorerView(
            catalog=catalog,
            calibration=calibration,
        )

    def as_tagger(self) -> TaggerView | None:
        """Return the TaggerView if available, else None."""
        return self.tagger if self.is_tagger else None

    def as_scorer(self) -> ScorerView | None:
        """Return the ScorerView if available, else None."""
        return self.scorer if self.is_scorer else None

    def inspect_data(self) -> dict[str, Any]:
        """Generic dictionary dump of all active domain data for tooling/serialization."""
        manifest: dict[str, Any] = {}
        for proto in self._plugin.implements:
            attrs = _get_protocol_required_attrs(proto)
            for attr in attrs:
                value = getattr(self._plugin, attr, None)
                if value is not None:
                    manifest[attr] = value.to_dict() if hasattr(value, "to_dict") else value
        return manifest

    @property
    def backend(self) -> Backend:
        """The active framework backend for this session."""
        return self._backend

    @property
    def variant(self) -> str | None:
        """The variant ID loaded for this session, if any."""
        return self._plan.variant_id

    @property
    def source(self) -> str:
        """The source location resolved for this session."""
        return self._source

    @property
    def auto_download(self) -> bool:
        """Whether downloads were permitted during session setup."""
        return self._auto_download

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
        if not hasattr(self, "_closed") or self._closed:
            return
        try:
            self.close()
        except Exception as exc:
            # Log teardown failures at debug level to avoid try-except-pass anti-pattern
            logger.debug("Ignored exception during session teardown in __del__: %s", exc)

    def __repr__(self) -> str:
        return f"ModelSession(model_id={self.model_id!r}, backend={self._backend.value!r})"

    # endregion Introspection & Trusted Views
