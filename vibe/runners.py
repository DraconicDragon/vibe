import logging
import threading
from typing import Any, Literal

from vibe.backends.base import Backend, ModelPlugin
from vibe.batch_utils import split_batch_output, stack_batch
from vibe.exceptions import InferenceCancelled, SessionError
from vibe.result_transforms import ResultTransform
from vibe.results import ModelResult
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
    def __init__(self, plugin: ModelPlugin, backend_instance: Any, pipeline: TransformPipeline):
        self.plugin = plugin
        self.backend_instance = backend_instance
        self.pipeline = pipeline
        self.model_id = plugin.identity.model_id

    def execute_single(self, image: Any, transforms: list[ResultTransform] | None) -> ModelResult:
        try:
            tensor = self.plugin.preprocess(image)
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

        try:
            result = self.plugin.postprocess(raw_output)
        except Exception as exc:
            raise SessionError(f"Postprocessing failed for model '{self.model_id}': {exc}") from exc

        return self.pipeline.apply(result, transforms)


class BatchRunner:
    def __init__(self, engine: InferenceEngine, state: SessionRunnerState, backend: Backend):
        self.engine = engine
        self.state = state
        self.backend = backend

    def resolve_batch_method(
        self, requested: Literal["auto", "true", "sequential"], batch_size: int
    ) -> Literal["true", "sequential"]:
        if batch_size <= 1:
            return "sequential"
        supports_true_batching = self._supports_true_batching()
        if requested == "true":
            if not supports_true_batching:
                logger.warning(
                    "Model_id=%s backend=%s does not support true batching; the run may fail if the export is batch-incompatible",
                    self.engine.model_id,
                    self.backend.value,
                )
            return "true"
        if requested == "sequential":
            return "sequential"
        return "true" if supports_true_batching else "sequential"

    def _supports_true_batching(self) -> bool:
        supports_fn = getattr(self.engine.backend_instance, "supports_true_batching", None)
        if callable(supports_fn):
            try:
                return bool(supports_fn())
            except Exception:
                logger.exception("Backend supports_true_batching() failed; using conservative fallback.")

        if self.backend == Backend.PYTORCH:
            device = str(getattr(self.engine.backend_instance, "device", "cpu")).lower()
            return device != "cpu"

        providers = [str(p) for p in getattr(self.engine.backend_instance, "providers", [])]
        return any(p.strip() and p.strip() != "CPUExecutionProvider" for p in providers)

    def execute_chunk(
        self, chunk_images: list[Any], transforms: list[ResultTransform] | None, fallback_to_sequential: bool
    ) -> list[ModelResult]:
        try:
            chunk_tensors = []
            for img in chunk_images:
                self.state.check_cancelled()
                chunk_tensors.append(self.engine.plugin.preprocess(img))
        except Exception as exc:
            raise SessionError(f"Preprocessing failed for model '{self.engine.model_id}': {exc}") from exc

        try:
            batch_tensor = stack_batch(chunk_tensors, self.engine.model_id)
        except SessionError:
            if fallback_to_sequential:
                self.state.warn_once(
                    key="batch_fallback",
                    message=f"Batch stacking failed for model '{self.engine.model_id}'; falling back to sequential execution.",
                )

                results = []
                for tensor in chunk_tensors:
                    self.state.check_cancelled()
                    results.append(self.engine.execute_tensor(tensor, transforms))
                return results
            raise

        try:
            raw_output = self.engine.backend_instance.run(batch_tensor)
        except Exception as exc:
            raise SessionError(f"Inference failed for model '{self.engine.model_id}': {exc}") from exc

        results = []
        for sample_output in split_batch_output(raw_output, len(chunk_images), self.engine.model_id):
            self.state.check_cancelled()
            try:
                result = self.engine.plugin.postprocess(sample_output)
            except Exception as exc:
                raise SessionError(f"Postprocessing failed for model '{self.engine.model_id}': {exc}") from exc
            results.append(self.engine.pipeline.apply(result, transforms))
        return results
