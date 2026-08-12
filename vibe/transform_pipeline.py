from collections.abc import Sequence

from vibe.exceptions import TransformError, TransformRequirementError
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
        self._failed_startup_transforms = set()

        # Deduce the supported classes directly from the unified tuple
        self.supported_transforms = []
        for t in plugin_transforms:
            if isinstance(t, type) and issubclass(t, ResultTransform):
                self.supported_transforms.append(t)
            elif isinstance(t, ResultTransform):
                self.supported_transforms.append(type(t))

    def is_supported(self, transform: ResultTransform) -> bool:
        """Check if a transform is declared in the model's capabilities."""
        return any(isinstance(transform, supported) for supported in self.supported_transforms)

    def apply(self, result: ModelResult, transforms: list[ResultTransform] | None) -> ModelResult:
        if not transforms:
            return result

        # Higher priority runs first (e.g., 100 -> 0 -> -100)
        ordered_transforms = sorted(transforms, key=lambda t: t.priority, reverse=True)

        clean_tags_idx = next((i for i, t in enumerate(ordered_transforms) if isinstance(t, CleanTags)), -1)
        if clean_tags_idx != -1 and clean_tags_idx != len(ordered_transforms) - 1:
            self.context.warn_once(
                "clean-tags-order",
                "CleanTags is not the last result transform in the execution order. "
                "This may cause tag matching errors with subsequent transforms.",
            )

        current = result
        for transform in ordered_transforms:
            # Skip transforms that already failed during the startup hook
            if id(transform) in self._failed_startup_transforms:
                continue

            is_declared = self.is_supported(transform)
            if not is_declared:
                self.context.warn_once(
                    f"unsupported-transform:{transform.transform_id}",
                    f"Transform '{transform.transform_id}' is not declared in ModelCapabilities.transforms "
                    f"for model '{self.model_id}'. Applying on a best-effort basis.",
                )

            # Enforce result type compatibility (treated as a requirement failure)
            if not transform.accepts_result(current):
                expected_name = transform.requires_result_type.__name__
                actual_name = type(current).__name__
                if is_declared:
                    raise TransformError(
                        f"Transform '{transform.transform_id}' declared by model '{self.model_id}' "
                        f"expects output type {expected_name}, but received {actual_name}."
                    )

                self.context.warn_once(
                    f"type-mismatch-transform:{transform.transform_id}",
                    f"Transform '{transform.transform_id}' expects {expected_name}, "
                    f"but received {actual_name} from model '{self.model_id}'. Skipping transform.",
                )
                continue

            # Execute transform with strict vs best-effort policy
            try:
                current = transform.apply(current, context=self.context)
            except TransformRequirementError as exc:
                if is_declared:
                    raise TransformError(
                        f"Transform '{transform.transform_id}' failed for model '{self.model_id}': {exc}"
                    ) from exc

                self.context.warn_once(
                    f"apply-req-failed-transform:{transform.transform_id}",
                    f"Undeclared transform '{transform.transform_id}' is missing requirements and will be skipped for model '{self.model_id}': {exc}",
                )
                continue
            except Exception as exc:
                # Always crash on real bugs, regardless of whether it's declared
                raise TransformError(
                    f"Transform '{transform.transform_id}' crashed during apply for model '{self.model_id}': {exc}"
                ) from exc

        return current

    def notify_infer_start(self, transforms: list[ResultTransform] | None) -> None:
        if not transforms:
            return

        for transform in transforms:
            is_declared = self.is_supported(transform)
            if not is_declared:
                self.context.warn_once(
                    f"unsupported-transform:{transform.transform_id}",
                    f"Transform '{transform.transform_id}' is not declared in ModelCapabilities.transforms "
                    f"for model '{self.model_id}'. Applying on a best-effort basis.",
                )

            try:
                transform.on_infer_start(context=self.context)
            except TransformRequirementError as exc:
                if is_declared:
                    raise TransformError(
                        f"Transform '{transform.transform_id}' failed during infer startup for model '{self.model_id}': {exc}"
                    ) from exc

                # Record requirement failure so it's safely bypassed in apply()
                self._failed_startup_transforms.add(id(transform))
                self.context.warn_once(
                    f"startup-req-failed-transform:{transform.transform_id}",
                    f"Undeclared transform '{transform.transform_id}' is missing requirements and will be skipped. Cause: {exc}",
                )
            except Exception as exc:
                # Real crash in startup logic (e.g. TypeError), fail hard
                raise TransformError(
                    f"Transform '{transform.transform_id}' crashed during infer startup for model '{self.model_id}': {exc}"
                ) from exc
