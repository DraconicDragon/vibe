import logging
import threading
from typing import Any, Literal

from vibe.backends.base import Backend, ModelPlugin, RuntimeExecutor
from vibe.exceptions import InferenceCancelled, SessionError
from vibe.features import InferenceRequest
from vibe.result_transforms import ResultTransform
from vibe.results import ModelResult, is_multi_score_result, is_tag_result
from vibe.transform_pipeline import TransformPipeline

logger = logging.getLogger(__name__)


def _fmt_shape(value: Any) -> Any:
    return getattr(value, "shape", None)


def _fmt_dtype(value: Any) -> Any:
    return getattr(value, "dtype", None)


class SessionRunnerState:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.lock = threading.RLock()
        self.run_state_lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.run_active = False
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


class InferenceEngine:
    def __init__(
        self,
        plugin: ModelPlugin,
        backend_instance: RuntimeExecutor,
        pipeline: TransformPipeline,
        state: SessionRunnerState,
    ):
        self.plugin = plugin
        self.backend_instance = backend_instance
        self.pipeline = pipeline
        self.model_id = plugin.identity.model_id
        self.state = state

    def execute_single(
        self,
        image: Any,
        transforms: list[ResultTransform] | None,
        request: InferenceRequest | None = None,
    ) -> ModelResult:
        try:
            tensor = self.plugin.preprocess(image, request=request)
            logger.debug("Preprocess output shape=%s dtype=%s", _fmt_shape(tensor), _fmt_dtype(tensor))
        except Exception as exc:
            raise SessionError(f"Preprocessing failed for model '{self.model_id}': {exc}") from exc

        return self.execute_tensor(tensor, transforms)

    def execute_tensor(self, tensor: Any, transforms: list[ResultTransform] | None) -> ModelResult:
        try:
            raw_output = self.backend_instance.run(tensor)
            logger.debug("Raw backend output shape=%s dtype=%s", _fmt_shape(raw_output), _fmt_dtype(raw_output))
        except Exception as exc:
            raise SessionError(f"Inference failed for model '{self.model_id}': {exc}") from exc

        return self.postprocess_and_audit(raw_output, transforms)

    def postprocess_and_audit(self, raw_output: Any, transforms: list[ResultTransform] | None) -> ModelResult:
        """Handles postprocessing, result transforms, and runtime metadata auditing safely."""
        try:
            result = self.plugin.postprocess(raw_output)
        except Exception as exc:
            raise SessionError(f"Postprocessing failed for model '{self.model_id}': {exc}") from exc

        capabilities = self.plugin.capabilities

        # Audit plugin-produced extras before transforms run. This keeps a transform's
        # declarations from masking an undeclared extra emitted by the plugin itself.
        plugin_extras = set(result.extras.keys())
        undocumented_plugin_extras = plugin_extras - set(capabilities.output_extras.keys())
        if undocumented_plugin_extras:
            self.state.warn_once(
                key=f"metadata-plugin-extras-{self.model_id}",
                message=(
                    f"Metadata mismatch for model '{self.model_id}': plugin postprocess returned undocumented "
                    f"top-level extras {undocumented_plugin_extras}."
                ),
                level=logging.ERROR,
            )

        plugin_entry_extras = set()
        if is_tag_result(result):
            for entries in result.tags.values():
                for entry in entries:
                    plugin_entry_extras.update(entry.extras.keys())
        elif is_multi_score_result(result):
            for entry in result.entries:
                plugin_entry_extras.update(entry.extras.keys())

        undocumented_plugin_entry_extras = plugin_entry_extras - set(capabilities.entry_extras.keys())
        if undocumented_plugin_entry_extras:
            self.state.warn_once(
                key=f"metadata-plugin-entry-extras-{self.model_id}",
                message=(
                    f"Metadata mismatch for model '{self.model_id}': plugin postprocess returned undocumented "
                    f"entry extras {undocumented_plugin_entry_extras}."
                ),
                level=logging.ERROR,
            )

        # Run transforms
        final_result = self.pipeline.apply(result, transforms)

        # region Metadata Audit
        # Audit Output Type
        if final_result.output_type != capabilities.output_type:
            self.state.warn_once(
                key=f"metadata-type-{self.model_id}",
                message=(
                    f"Metadata mismatch for model '{self.model_id}': declared output_type "
                    f"is '{capabilities.output_type.value}', but postprocess returned '{final_result.output_type.value}'."
                ),
                level=logging.ERROR,
            )

        # Audit Output Categories (TagResult only)
        if is_tag_result(final_result):
            undocumented_cats = set(final_result.tags.keys()) - set(capabilities.output_categories)
            if undocumented_cats:
                self.state.warn_once(
                    key=f"metadata-cats-{self.model_id}",
                    message=f"Metadata mismatch for model '{self.model_id}': returned undocumented categories {undocumented_cats}.",
                    level=logging.WARNING,
                )

        # Transforms may intentionally add declared metadata. Audit the final
        # result against both model and active-transform declarations so those
        # additions remain visible without masking genuine mismatches.
        declared_output_extras = set(capabilities.output_extras)
        declared_entry_extras = set(capabilities.entry_extras)
        for transform in transforms or ():
            declared_output_extras.update(transform.output_extras)
            declared_entry_extras.update(transform.entry_extras)

        undocumented_extras = set(final_result.extras) - declared_output_extras
        if undocumented_extras:
            self.state.warn_once(
                key=f"metadata-extras-{self.model_id}",
                message=(
                    f"Metadata mismatch for model '{self.model_id}': returned undocumented top-level extras "
                    f"{undocumented_extras}."
                ),
                level=logging.ERROR,
            )

        undocumented_entry_extras: set[str] = set()
        if is_tag_result(final_result):
            for entries in final_result.tags.values():
                for entry in entries:
                    undocumented_entry_extras.update(set(entry.extras) - declared_entry_extras)
        elif is_multi_score_result(final_result):
            for entry in final_result.entries:
                undocumented_entry_extras.update(set(entry.extras) - declared_entry_extras)

        if undocumented_entry_extras:
            self.state.warn_once(
                key=f"metadata-entry-extras-{self.model_id}",
                message=(
                    f"Metadata mismatch for model '{self.model_id}': returned undocumented entry extras "
                    f"{undocumented_entry_extras}."
                ),
                level=logging.ERROR,
            )

        return final_result


class BatchRunner:
    def __init__(self, engine: InferenceEngine, state: SessionRunnerState, backend: Backend):
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

    def _run_sequential(self, chunk_tensors: list[Any], transforms: list[ResultTransform] | None) -> list[ModelResult]:
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
            try:
                results.append(self.engine.execute_tensor(tensor, transforms))
            except InferenceCancelled:
                raise
            except Exception:
                logger.exception(
                    "Sequential inference failed for model '%s' during batch fallback",
                    self.engine.model_id,
                )
                raise
        return results

    def execute_chunk(
        self,
        chunk_images: list[Any],
        transforms: list[ResultTransform] | None,
        request: InferenceRequest | None,
        fallback_to_sequential: bool,
    ) -> list[ModelResult]:
        self.state.check_cancelled()
        try:
            chunk_tensors = []
            for img in chunk_images:
                self.state.check_cancelled()
                chunk_tensors.append(self.engine.plugin.preprocess(img, request=request))
        except Exception as exc:
            raise SessionError(f"Preprocessing failed for model '{self.engine.model_id}': {exc}") from exc

        # Skip batch attempt if a previous chunk in this run already failed
        if self._batching_disabled and fallback_to_sequential:
            logger.debug(
                "Batching remains disabled for model '%s'; processing chunk sequentially",
                self.engine.model_id,
            )
            return self._run_sequential(chunk_tensors, transforms)

        # Collation
        try:
            batch_tensor = self.engine.plugin.collate_batch(chunk_tensors)
        except Exception as exc:
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
                return self._run_sequential(chunk_tensors, transforms)
            logger.exception("Batch collation failed for model '%s'", self.engine.model_id)
            raise SessionError(f"Could not collate batch for model '{self.engine.model_id}': {exc}") from exc

        # Execution
        try:
            raw_output = self.engine.backend_instance.run(batch_tensor)
        except Exception as exc:
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
                return self._run_sequential(chunk_tensors, transforms)
            logger.exception("Batch execution failed for model '%s'", self.engine.model_id)
            raise SessionError(f"Inference failed for model '{self.engine.model_id}': {exc}") from exc

        # Output Splitting
        try:
            split_outputs = self.engine.plugin.split_batch(raw_output, len(chunk_images))
        except Exception as exc:
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
                return self._run_sequential(chunk_tensors, transforms)
            logger.exception("Batch output splitting failed for model '%s'", self.engine.model_id)
            raise SessionError(f"Could not split batch output for model '{self.engine.model_id}': {exc}") from exc

        # Success path
        results = []
        for sample_output in split_outputs:
            self.state.check_cancelled()
            results.append(self.engine.postprocess_and_audit(sample_output, transforms))
        return results
