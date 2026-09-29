"""
Execution runners, pipelined batch processing, and postprocessing auditing for inference sessions.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from collections.abc import Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from vibe.backends.base import Backend, ModelPlugin, RuntimeExecutor
from vibe.exceptions import InferenceCancelled, SessionError
from vibe.image_loading import NormalizedInput, iter_raw_chunks, load_image_if_path
from vibe.results import InferenceResult, InferenceResultItem, ModelResult, is_multi_score_result, is_tag_result
from vibe.settings import InferenceRequest

logger = logging.getLogger(__name__)

_PIPELINE_SENTINEL = object()


def _fmt_shape(value: Any) -> Any:
    return getattr(value, "shape", None)


def _fmt_dtype(value: Any) -> Any:
    return getattr(value, "dtype", None)


class SessionRunnerState:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.lock = threading.RLock()
        self.run_state_lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.run_active = False
        self.metadata_audited = False
        self._warned_keys: set[str] = set()

    def warn_once(self, key: str, message: str, level: int = logging.WARNING) -> None:
        """Log a message exactly once for the lifetime of this session state."""
        with self.lock:
            if key in self._warned_keys:
                return
            self._warned_keys.add(key)
        logger.log(level, message)

    def start_run(self) -> None:
        with self.run_state_lock:
            self.run_active = True
            self.cancel_event.clear()
        logger.debug("Run state -> active for model_id=%s", self.model_id)

    def finish_run(self) -> None:
        with self.run_state_lock:
            self.run_active = False
            self.cancel_event.clear()
        logger.debug("Run state -> idle for model_id=%s", self.model_id)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            logger.warning("Inference cancelled before completing current step model_id=%s", self.model_id)
            raise InferenceCancelled("Inference cancelled by user request.")

    def cancel(self) -> bool:
        with self.run_state_lock:
            if not self.run_active:
                return False
        self.cancel_event.set()
        logger.warning("Cancellation requested for model_id=%s", self.model_id)
        return True

    @property
    def is_running(self) -> bool:
        with self.run_state_lock:
            return self.run_active

    @property
    def is_cancellation_requested(self) -> bool:
        return self.cancel_event.is_set()


@contextmanager
def execution_boundary(operation: str, state: SessionRunnerState) -> Iterator[None]:
    """
    Guards execution steps:
    1. Always lets InferenceCancelled and existing SessionError pass through untouched.
    2. If an internal exception occurs while cancellation was requested,
       re-checks state and raises InferenceCancelled instead of SessionError.
    3. Wraps true runtime failures into SessionError.
    """
    try:
        yield
    except (InferenceCancelled, SessionError):
        raise
    except Exception as exc:
        if state.is_cancellation_requested:
            state.check_cancelled()
        raise SessionError(f"{operation} failed for model '{state.model_id}': {exc}") from exc


class InferenceEngine:
    def __init__(
        self,
        plugin: ModelPlugin,
        backend_instance: RuntimeExecutor,
        state: SessionRunnerState,
    ) -> None:
        self.plugin = plugin
        self.backend_instance = backend_instance
        self.model_id = plugin.identity.model_id
        self.state = state

    def execute_single(
        self,
        image: Any,
        request: InferenceRequest | None = None,
    ) -> ModelResult:
        with execution_boundary("Preprocessing", self.state):
            tensor = self.plugin.preprocess(image, request=request)
            logger.debug("Preprocess output shape=%s dtype=%s", _fmt_shape(tensor), _fmt_dtype(tensor))

        return self.execute_tensor(tensor)

    def execute_tensor(self, tensor: Any) -> ModelResult:
        with execution_boundary("Inference", self.state):
            raw_output = self.backend_instance.run(tensor)
            logger.debug("Raw backend output shape=%s dtype=%s", _fmt_shape(raw_output), _fmt_dtype(raw_output))

        return self.postprocess_and_audit(raw_output)

    def postprocess_and_audit(self, raw_output: Any) -> ModelResult:
        """Handles postprocessing and runtime metadata auditing."""
        with execution_boundary("Postprocessing", self.state):
            result = self.plugin.postprocess(raw_output)

        if not self.state.metadata_audited:
            out_spec = self.plugin.profile.output_spec

            # 1. Audit Output Type
            if result.output_type != out_spec.kind:
                self.state.warn_once(
                    key=f"metadata-type-{self.model_id}",
                    message=(
                        f"Metadata mismatch for model '{self.model_id}': declared output kind "
                        f"is '{out_spec.kind.value}', but postprocess returned '{result.output_type.value}'."
                    ),
                    level=logging.ERROR,
                )

            # 2. Audit Output Extras
            plugin_extras = set(result.extras.keys())
            undocumented_plugin_extras = plugin_extras - set(out_spec.output_extras.keys())
            if undocumented_plugin_extras:
                self.state.warn_once(
                    key=f"metadata-plugin-extras-{self.model_id}",
                    message=(
                        f"Metadata mismatch for model '{self.model_id}': plugin postprocess returned undocumented "
                        f"top-level extras {undocumented_plugin_extras}."
                    ),
                    level=logging.ERROR,
                )

            # 3. Audit Entry Extras (executed only once per session)
            plugin_entry_extras: set[str] = set()
            if is_tag_result(result):
                for entries in result.categories.values():
                    for entry in entries:
                        plugin_entry_extras.update(entry.extras.keys())
            elif is_multi_score_result(result):
                for entry in result.entries:
                    plugin_entry_extras.update(entry.extras.keys())

            undocumented_plugin_entry_extras = plugin_entry_extras - set(out_spec.entry_extras.keys())
            if undocumented_plugin_entry_extras:
                self.state.warn_once(
                    key=f"metadata-plugin-entry-extras-{self.model_id}",
                    message=(
                        f"Metadata mismatch for model '{self.model_id}': plugin postprocess returned undocumented "
                        f"entry extras {undocumented_plugin_entry_extras}."
                    ),
                    level=logging.ERROR,
                )

            self.state.metadata_audited = True

        return result


# region Batch Pipeline


@dataclass(slots=True)
class PreparedBatch:
    """Preprocessed and collated batch ready for execution."""

    start_index: int
    batch_tensor: Any | None
    chunk_tensors: list[Any] | None
    refs: list[Any]
    item_count: int


class BatchPipeline:
    """
    Double-buffered producer-consumer pipeline.
    Preprocesses and collates batch N+1 in the background while the GPU computes batch N.
    """

    def __init__(
        self,
        runner: BatchRunner,
        method: Literal["true", "sequential"],
        fallback_to_sequential: bool,
        request: InferenceRequest | None,
        batch_size: int,
        prefetch_batches: int = 2,
    ) -> None:
        self.runner = runner
        self.engine = runner.engine
        self.state = runner.state
        self.method = method
        self.fallback_to_sequential = fallback_to_sequential
        self.request = request
        self.batch_size = batch_size
        self.prefetch_batches = max(1, prefetch_batches)

        # Intra-batch multi-core scaling for CPU preprocessing
        cpu_count = os.cpu_count() or 4
        self.num_workers = 1 if batch_size == 1 else min(cpu_count, batch_size, 8)

    def _producer_thread(
        self,
        q: queue.Queue[PreparedBatch | Exception | object],
        stop_event: threading.Event,
        norm: NormalizedInput,
    ) -> None:
        """Background worker thread that runs load, preprocess, and collate."""

        def _init_worker() -> None:
            # Run once per worker thread upon creation (safe for non-torch runtimes)
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None:
                torch_mod.set_num_threads(1)

        def _load_and_preprocess(task: tuple[int, Any]) -> Any:
            global_idx, val = task
            self.state.check_cancelled()
            img = load_image_if_path(val, index=global_idx, error_cls=SessionError)
            self.state.check_cancelled()
            with execution_boundary("Preprocessing", self.state):
                return self.engine.plugin.preprocess(img, request=self.request)

        try:
            with ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix=f"vibe-prep-{self.engine.model_id}",
                initializer=_init_worker,
            ) as pool:
                for start_idx, raw_values, chunk_refs in iter_raw_chunks(norm, self.batch_size, error_cls=SessionError):
                    if stop_event.is_set() or self.state.is_cancellation_requested:
                        break

                    # Construct tasks with pre-computed indices
                    tasks = [(start_idx + i, val) for i, val in enumerate(raw_values)]

                    # Preprocess images (parallel across pool if batch_size > 1)
                    if len(tasks) == 1:
                        chunk_tensors = [_load_and_preprocess(tasks[0])]
                    else:
                        chunk_tensors = list(pool.map(_load_and_preprocess, tasks))

                    if stop_event.is_set() or self.state.is_cancellation_requested:
                        break

                    batch_tensor = None
                    if self.method == "true" and not self.runner._batching_disabled:
                        try:
                            batch_tensor = self.engine.plugin.collate_batch(chunk_tensors)
                        except Exception as exc:
                            if not self.fallback_to_sequential:
                                raise SessionError(
                                    f"Could not collate batch for model '{self.engine.model_id}': {exc}"
                                ) from exc
                            batch_tensor = None  # Fallback to sequential execution

                    # Retain chunk_tensors only when sequential execution or fallback might be needed
                    saved_chunk_tensors = (
                        chunk_tensors if (batch_tensor is None or self.fallback_to_sequential) else None
                    )

                    prepared = PreparedBatch(
                        start_index=start_idx,
                        batch_tensor=batch_tensor,
                        chunk_tensors=saved_chunk_tensors,
                        refs=chunk_refs,
                        item_count=len(raw_values),
                    )

                    while not stop_event.is_set() and not self.state.is_cancellation_requested:
                        try:
                            q.put(prepared, timeout=0.1)
                            break
                        except queue.Full:
                            continue

        except Exception as exc:
            if not stop_event.is_set():
                while not stop_event.is_set():
                    try:
                        q.put(exc, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            while not stop_event.is_set():
                try:
                    q.put(_PIPELINE_SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    if stop_event.is_set():
                        break

    def stream_batches(self, norm: NormalizedInput) -> Generator[InferenceResult, None, None]:
        """Main consumer generator running GPU forward pass and postprocessing."""
        q: queue.Queue[PreparedBatch | Exception | object] = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        total_inputs = norm.total

        producer = threading.Thread(
            target=self._producer_thread,
            args=(q, stop_event, norm),
            name=f"vibe-pipeline-{self.engine.model_id}",
            daemon=True,
        )
        producer.start()

        try:
            while True:
                self.state.check_cancelled()

                item = None
                while not stop_event.is_set():
                    self.state.check_cancelled()
                    try:
                        item = q.get(timeout=0.05)
                        break
                    except queue.Empty:
                        continue

                if item is None or item is _PIPELINE_SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item

                # Narrow type for type checkers (fixes invalid-assignment)
                assert isinstance(item, PreparedBatch)
                batch = item

                start = batch.start_index
                refs = batch.refs
                count = batch.item_count

                # GPU / Accelerator Execution
                if batch.batch_tensor is not None and not self.runner._batching_disabled:
                    try:
                        with execution_boundary("Inference", self.state):
                            raw_output = self.engine.backend_instance.run(batch.batch_tensor)

                        with execution_boundary("OutputSplitting", self.state):
                            split_outputs = self.engine.plugin.split_batch(raw_output, count)

                        with execution_boundary("Postprocessing", self.state):
                            results = []
                            for sample_output in split_outputs:
                                self.state.check_cancelled()
                                results.append(self.engine.postprocess_and_audit(sample_output))

                    except Exception as exc:
                        self.state.check_cancelled()
                        if self.fallback_to_sequential and batch.chunk_tensors is not None:
                            self.runner._batching_disabled = True
                            self.state.warn_once(
                                key="batch_run_fallback",
                                message=f"Batch execution failed for model '{self.engine.model_id}': {exc}. Sequential fallback active.",
                            )
                            results = self.runner._run_sequential(batch.chunk_tensors)
                        else:
                            raise
                else:
                    assert batch.chunk_tensors is not None
                    results = self.runner._run_sequential(batch.chunk_tensors)

                chunk_items = [
                    InferenceResultItem(index=start + i, input_ref=refs[i], result=results[i]) for i in range(count)
                ]

                yield InferenceResult(total_inputs=total_inputs, items=chunk_items)

        finally:
            stop_event.set()
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            producer.join(timeout=1.0)


# endregion Batch Pipeline


class BatchRunner:
    def __init__(self, engine: InferenceEngine, state: SessionRunnerState, backend: Backend) -> None:
        self.engine = engine
        self.state = state
        self.backend = backend
        self._batching_disabled = False

    def resolve_batch_method(
        self, requested: Literal["auto", "true", "sequential"], batch_size: int
    ) -> Literal["true", "sequential"]:
        # Reset per inference run
        self._batching_disabled = False
        if batch_size <= 1:
            return "sequential"
        supports_true = self._supports_true_batching()
        if requested == "true":
            if not supports_true:
                logger.warning(
                    "Model_id=%s backend=%s does not support true batching; the run may fail if the export is batch-incompatible",
                    self.engine.model_id,
                    self.backend.value,
                )
            return "true"
        if requested == "sequential":
            return "sequential"
        return "true" if supports_true else "sequential"

    def _supports_true_batching(self) -> bool:
        """Query the backend strictly through the RuntimeExecutor protocol."""
        try:
            return bool(self.engine.backend_instance.supports_true_batching())
        except Exception:
            logger.exception("Backend supports_true_batching() failed; using conservative sequential fallback.")
            return False

    def _run_sequential(self, chunk_tensors: list[Any]) -> list[ModelResult]:
        """Process preprocessed tensors one-by-one with cancellation checks and backend cache cleanup."""
        clear_cache = getattr(self.engine.backend_instance, "clear_cache", None)
        if callable(clear_cache):
            try:
                clear_cache()
            except Exception as exc:
                logger.debug("Backend clear_cache() failed during sequential fallback: %s", exc)

        results = []
        for tensor in chunk_tensors:
            self.state.check_cancelled()
            results.append(self.engine.execute_tensor(tensor))
        return results

    def create_pipeline(
        self,
        method: Literal["true", "sequential"],
        fallback_to_sequential: bool,
        request: InferenceRequest | None,
        batch_size: int,
        prefetch_batches: int = 2,
    ) -> BatchPipeline:
        return BatchPipeline(
            runner=self,
            method=method,
            fallback_to_sequential=fallback_to_sequential,
            request=request,
            batch_size=batch_size,
            prefetch_batches=prefetch_batches,
        )
