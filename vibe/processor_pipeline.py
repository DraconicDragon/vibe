from typing import Sequence

from vibe.result_processors import CleanTags, ResultProcessor, ResultProcessorContext
from vibe.results import ModelResult
from vibe.session import ProcessorError


class ProcessorPipeline:
    def __init__(
        self,
        model_id: str,
        processor_context: ResultProcessorContext,
        supported_processors: Sequence[type[ResultProcessor]],
    ):
        self.model_id = model_id
        self.processor_context = processor_context
        self.supported_processors = supported_processors

    def apply(self, result: ModelResult, processors: list[ResultProcessor] | None) -> ModelResult:
        if not processors:
            return result

        effective_processors: list[ResultProcessor] = []
        cleanup_processors: list[CleanTags] = []

        for rp in processors:
            if isinstance(rp, CleanTags):
                cleanup_processors.append(rp)
            else:
                effective_processors.append(rp)

        effective_processors.extend(cleanup_processors)

        current = result
        for processor in effective_processors:
            if not any(isinstance(processor, supported) for supported in self.supported_processors):
                proc_name = processor.__class__.__name__
                self.processor_context.warn_once(
                    f"unsupported-processor:{proc_name}",
                    f"Processor '{proc_name}' is not declared as supported by model '{self.model_id}'; attempting to apply anyway.",
                )

            try:
                current = processor.process(current, context=self.processor_context)
            except Exception as exc:
                raise ProcessorError(
                    f"Result processor '{processor.__class__.__name__}' failed for model '{self.model_id}': {exc}"
                ) from exc
        return current
