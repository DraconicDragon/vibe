from collections.abc import Sequence

from vibe.exceptions import TransformError
from vibe.result_transforms import CleanTags, ResultTransform, TransformContext
from vibe.results import ModelResult


class TransformPipeline:
    def __init__(
        self,
        model_id: str,
        context: TransformContext,
        plugin_transforms: Sequence[type[ResultTransform] | ResultTransform],
    ):
        self.model_id = model_id
        self.context = context

        # Deduce the supported classes directly from the unified tuple
        self.supported_transforms = []
        for t in plugin_transforms:
            if isinstance(t, type) and issubclass(t, ResultTransform):
                self.supported_transforms.append(t)
            elif isinstance(t, ResultTransform):
                self.supported_transforms.append(type(t))

    def apply(self, result: ModelResult, transforms: list[ResultTransform] | None) -> ModelResult:
        if not transforms:
            return result

        ordered_transforms = sorted(transforms, key=lambda t: t.priority)

        clean_tags_idx = next((i for i, t in enumerate(ordered_transforms) if isinstance(t, CleanTags)), -1)
        if clean_tags_idx != -1 and clean_tags_idx != len(ordered_transforms) - 1:
            self.context.warn_once(
                "clean-tags-order",
                "CleanTags is not the last result transform in the execution order. "
                "This may cause tag matching errors with subsequent transforms.",
            )

        current = result
        for transform in ordered_transforms:
            if not any(isinstance(transform, supported) for supported in self.supported_transforms):
                self.context.warn_once(
                    f"unsupported-transform:{transform.transform_id}",
                    f"Transform '{transform.transform_id}' is not declared as supported by model '{self.model_id}'.",
                )

            try:
                current = transform.apply(current, context=self.context)
            except Exception as exc:
                raise TransformError(
                    f"Transform '{transform.transform_id}' failed for model '{self.model_id}': {exc}"
                ) from exc

        return current

    def notify_infer_start(self, transforms: list[ResultTransform] | None) -> None:
        if not transforms:
            return
        for transform in transforms:
            try:
                transform.on_infer_start(context=self.context)
            except Exception as exc:
                raise TransformError(
                    f"Transform '{transform.transform_id}' failed during infer startup for model '{self.model_id}': {exc}"
                ) from exc
