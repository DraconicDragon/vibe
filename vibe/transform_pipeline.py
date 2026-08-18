"""Transforms execution and lifecycle management."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from vibe.exceptions import SessionError, TransformError, TransformRequirementError
from vibe.features import FeatureSpec
from vibe.result_transforms import CleanTags, ResultTransform, TransformContext
from vibe.results import ModelResult

logger = logging.getLogger(__name__)


class TransformPipeline:
    def __init__(
        self,
        model_id: str,
        context: TransformContext,
        supported_features: Sequence[FeatureSpec],
    ):
        self.model_id = model_id
        self.context = context
        # Keep the object alongside its id so Python cannot recycle an id for
        # a later transform instance and accidentally skip that new instance.
        self._failed_startup_transforms: dict[int, ResultTransform] = {}

        # Extract supported transform classes from feature specs
        self.supported_transform_types: list[type[ResultTransform]] = [
            f.config_type
            for f in supported_features
            if f.binding == "result_transform" and isinstance(f.config_type, type)
        ]
        self._declared_order = {
            f.id: index
            for index, f in enumerate(supported_features)
            if f.binding == "result_transform"
        }
        self._declared_type_order = {
            f.config_type: index
            for index, f in enumerate(supported_features)
            if f.binding == "result_transform"
        }

    def is_supported(self, transform: ResultTransform) -> bool:
        return any(isinstance(transform, supported) for supported in self.supported_transform_types)

    @staticmethod
    def _validate_unique_transforms(transforms: Sequence[ResultTransform]) -> None:
        seen_ids: set[str] = set()
        for transform in transforms:
            transform_id = getattr(transform, "transform_id", type(transform).__name__)
            if transform_id in seen_ids:
                raise SessionError(f"Duplicate transform '{transform_id}' detected in inference request.")
            seen_ids.add(transform_id)

    def _ordered_transforms(self, transforms: Sequence[ResultTransform]) -> list[ResultTransform]:
        """Order by priority, then stable model declaration order for declared features."""
        fallback_order = len(self._declared_order)
        indexed = list(enumerate(transforms))
        indexed.sort(
            key=lambda item: (
                -item[1].priority,
                self._declared_order.get(
                    item[1].transform_id,
                    self._declared_type_order.get(type(item[1]), fallback_order),
                ),
                item[0],
            )
        )
        return [transform for _, transform in indexed]

    def apply(self, result: ModelResult, transforms: list[ResultTransform] | None) -> ModelResult:
        if not transforms:
            return result

        self._validate_unique_transforms(transforms)

        # Highest priority first; equal-priority declared features use model declaration order.
        ordered_transforms = self._ordered_transforms(transforms)

        clean_tags_idx = next((i for i, t in enumerate(ordered_transforms) if isinstance(t, CleanTags)), -1)
        if clean_tags_idx != -1 and clean_tags_idx != len(ordered_transforms) - 1:
            self.context.warn_once(
                "clean-tags-order",
                "CleanTags is not the last result transform in execution order. "
                "This may cause tag matching errors with subsequent transforms.",
            )

        current = result
        for transform in ordered_transforms:
            # Skip transforms that already failed during the startup hook
            if self._failed_startup_transforms.get(id(transform)) is transform:
                continue

            is_declared = self.is_supported(transform)
            if not is_declared:
                self.context.warn_once(
                    f"unsupported-transform:{transform.transform_id}",
                    f"Transform '{transform.transform_id}' is not declared in ModelCapabilities.features "
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
                    f"Undeclared transform '{transform.transform_id}' is missing requirements and will be skipped: {exc}",
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

        self._validate_unique_transforms(transforms)

        for transform in self._ordered_transforms(transforms):
            is_declared = self.is_supported(transform)
            if not is_declared:
                self.context.warn_once(
                    f"unsupported-transform:{transform.transform_id}",
                    f"Transform '{transform.transform_id}' is not declared in ModelCapabilities.features "
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
                self._failed_startup_transforms[id(transform)] = transform
                self.context.warn_once(
                    f"startup-req-failed-transform:{transform.transform_id}",
                    f"Undeclared transform '{transform.transform_id}' is missing requirements and will be skipped: {exc}",
                )
            except Exception as exc:
                raise TransformError(
                    f"Transform '{transform.transform_id}' crashed during infer startup for model '{self.model_id}': {exc}"
                ) from exc
