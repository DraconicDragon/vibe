"""
Execution runners and postprocessing auditing for inference sessions.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from vibe.backends.base import Backend, ModelPlugin, RuntimeExecutor
from vibe.exceptions import InferenceCancelled, SessionError
from vibe.results import ModelResult, is_multi_score_result, is_tag_result
from vibe.settings import InferenceRequest

logger = logging.getLogger(__name__)


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

    def execute_chunk(
        self,
        chunk_images: list[Any],
        request: InferenceRequest | None,
        fallback_to_sequential: bool,
    ) -> list[ModelResult]:
        self.state.check_cancelled()

        # Preprocessing
        chunk_tensors = []
        with execution_boundary("Preprocessing", self.state):
            for img in chunk_images:
                self.state.check_cancelled()
                chunk_tensors.append(self.engine.plugin.preprocess(img, request=request))

        # Skip batch attempt if a previous chunk in this run already failed
        if self._batching_disabled and fallback_to_sequential:
            logger.debug(
                "Batching remains disabled for model '%s'; processing chunk sequentially",
                self.engine.model_id,
            )
            return self._run_sequential(chunk_tensors)

        # Collation
        try:
            batch_tensor = self.engine.plugin.collate_batch(chunk_tensors)
        except Exception as exc:
            self.state.check_cancelled()  # Abort immediately if cancelled; do NOT fall back!
            if fallback_to_sequential:
                self._batching_disabled = True
                logger.debug(
                    "Batch collation failure for model '%s'; sequential fallback details",
                    self.engine.model_id,
                    exc_info=True,
                )
                self.state.warn_once(
                    key="batch_collate_fallback",
                    message=f"Batch stacking failed for model '{self.engine.model_id}': {exc}. Sequential fallback active.",
                )
                return self._run_sequential(chunk_tensors)
            logger.exception("Batch collation failed for model '%s'", self.engine.model_id)
            raise SessionError(f"Could not collate batch for model '{self.engine.model_id}': {exc}") from exc

        # Execution
        try:
            raw_output = self.engine.backend_instance.run(batch_tensor)
        except Exception as exc:
            self.state.check_cancelled()  # Abort immediately if cancelled; do NOT fall back!
            if fallback_to_sequential:
                self._batching_disabled = True
                del batch_tensor
                logger.debug(
                    "Batch execution failure for model '%s'; sequential fallback details",
                    self.engine.model_id,
                    exc_info=True,
                )
                self.state.warn_once(
                    key="batch_run_fallback",
                    message=f"Batch execution failed for model '{self.engine.model_id}': {exc}. Sequential fallback active.",
                )
                return self._run_sequential(chunk_tensors)
            logger.exception("Batch execution failed for model '%s'", self.engine.model_id)
            raise SessionError(f"Inference failed for model '{self.engine.model_id}': {exc}") from exc

        # Output Splitting
        try:
            split_outputs = self.engine.plugin.split_batch(raw_output, len(chunk_images))
        except Exception as exc:
            self.state.check_cancelled()  # Abort immediately if cancelled; do NOT fall back!
            if fallback_to_sequential:
                self._batching_disabled = True
                del batch_tensor, raw_output
                logger.debug(
                    "Batch output splitting failure for model '%s'; sequential fallback details",
                    self.engine.model_id,
                    exc_info=True,
                )
                self.state.warn_once(
                    key="batch_split_fallback",
                    message=f"Batch splitting failed for model '{self.engine.model_id}': {exc}. Sequential fallback active.",
                )
                return self._run_sequential(chunk_tensors)
            logger.exception("Batch output splitting failed for model '%s'", self.engine.model_id)
            raise SessionError(f"Could not split batch output for model '{self.engine.model_id}': {exc}") from exc

        # Postprocessing
        results = []
        for sample_output in split_outputs:
            self.state.check_cancelled()
            results.append(self.engine.postprocess_and_audit(sample_output))
        return results
