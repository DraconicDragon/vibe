"""Thresholding transforms: global/category filtering and per-tag optimal thresholds."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Self

import numpy as np
from pydantic import BaseModel, Field

from vibe.exceptions import TransformError, TransformRequirementError
from vibe.metadata import ModelDescriptor, OutputKind
from vibe.results import TagEntry, TagResult, build_category_arrays, build_sorted_tag_entries
from vibe.session import ModelSession
from vibe.transforms.base import ResultTransform


class ScoreThresholdsConfig(BaseModel):
    """Configuration schema for ScoreThresholds."""

    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        title="Base Threshold",
        description="Default minimum score required for tags.",
    )
    category_thresholds: dict[str, float] = Field(
        default_factory=dict,
        title="Category Thresholds",
        description="Per-category threshold overrides (e.g. character: 0.75).",
    )


class TagLevelThresholdsConfig(BaseModel):
    """Configuration schema for TagLevelThresholds."""

    offset: float = Field(
        default=0.0,
        title="Threshold Offset",
        description="Fixed value added to each tag's calibrated threshold.",
    )
    relative_offset: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        title="Relative Offset",
        description="Relative adjustment (-1.0 to 1.0) applied to each threshold.",
    )
    fallback: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Fallback Threshold",
        description="Fallback threshold when a tag has no pre-calibrated value.",
    )


@dataclass(frozen=True, slots=True)
class ScoreThresholds(ResultTransform[TagResult]):
    """Filter tags using a base score threshold and optional category overrides."""

    id: ClassVar[str] = "score_thresholds"
    display_name: ClassVar[str] = "Score Thresholds"
    description: ClassVar[str] = "Filter tags using global and category score thresholds."
    config_model: ClassVar[type[BaseModel] | None] = ScoreThresholdsConfig

    threshold: float
    category_thresholds: Mapping[str, float] = field(default_factory=dict)
    _plan_cache: dict[
        tuple[int, int],
        tuple[Mapping[str, Sequence[int]], tuple[tuple[str, np.ndarray, float], ...]],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not (0.0 <= self.threshold <= 1.0):
            raise TransformError(f"threshold must be between 0.0 and 1.0, got {self.threshold}")

        if self.category_thresholds:
            normalized: dict[str, float] = {}
            for category, value in self.category_thresholds.items():
                category_key = category.value if isinstance(category, Enum) else str(category)
                if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                    raise TransformError(
                        f"Threshold for category '{category_key}' must be in [0.0, 1.0], got {value}"
                    )
                normalized[category_key] = float(value)
            object.__setattr__(self, "category_thresholds", normalized)

    @classmethod
    def from_session(
        cls,
        session: ModelSession,
        fallback_threshold: float,
        *,
        category_thresholds: Mapping[str, float] | None = None,
    ) -> Self:
        """
        Construct ScoreThresholds using model recommendations, backed by a mandatory fallback.

        Args:
            session: Active ModelSession.
            fallback_threshold: Mandatory safety floor for any category without a recommended threshold.
            category_thresholds: Explicit category overrides (e.g. {"character": 0.85}).
        """
        if not session.is_tagger:
            raise TransformRequirementError(
                f"ScoreThresholds requires a tagger, but '{session.model_id}' outputs {session.metadata.output.kind}."
            )

        recommendation = session.tagger.recommendation
        base_threshold = (
            recommendation.global_threshold
            if recommendation and recommendation.global_threshold is not None
            else fallback_threshold
        )

        final_categories = dict(recommendation.category_thresholds) if recommendation else {}
        if category_thresholds:
            final_categories.update(category_thresholds)

        return cls(threshold=base_threshold, category_thresholds=final_categories)

    @classmethod
    def is_supported(cls, target: ModelDescriptor | ModelSession) -> bool:
        if isinstance(target, ModelSession):
            return target.is_tagger
        return target.output.kind == OutputKind.TAGS

    def _get_compiled_plan(
        self,
        category_indices: Mapping[str, Sequence[int]],
        usable_count: int,
    ) -> tuple[tuple[str, np.ndarray, float], ...]:
        key = (id(category_indices), usable_count)
        cached = self._plan_cache.get(key)
        if cached is not None and cached[0] is category_indices:
            return cached[1]

        category_arrays = build_category_arrays(category_indices, usable_count)
        plan = tuple(
            (category, indices, self.category_thresholds.get(category, self.threshold))
            for category, indices in category_arrays
        )

        if len(self._plan_cache) >= 4:
            self._plan_cache.pop(next(iter(self._plan_cache)))
        self._plan_cache[key] = (category_indices, plan)
        return plan

    def __call__(self, result: TagResult) -> TagResult:
        if result.is_array_backed:
            assert result._scores is not None and result._tag_names is not None and result._category_indices is not None
            scores = result._scores
            names = result._tag_names
            usable_count = min(len(scores), len(names))
            if usable_count == 0:
                return TagResult(categories={}, extras=dict(result.extras))

            filtered_categories: dict[str, list[TagEntry]] = {}
            for category, indices, threshold in self._get_compiled_plan(result._category_indices, usable_count):
                category_scores = scores[indices]
                mask = category_scores >= threshold
                if np.any(mask):
                    filtered_categories[category] = build_sorted_tag_entries(
                        names,
                        indices[mask],
                        category_scores[mask],
                    )

            return TagResult(categories=filtered_categories, extras=dict(result.extras))

        filtered_categories = {}
        for category, entries in result.categories.items():
            threshold = self.category_thresholds.get(category, self.threshold)
            kept = [entry for entry in entries if entry.score >= threshold]
            if kept:
                filtered_categories[category] = kept

        return TagResult(categories=filtered_categories, extras=dict(result.extras))


@dataclass(frozen=True, slots=True)
class TagLevelThresholds(ResultTransform[TagResult]):
    """Filter tags using individual calibrated per-tag decision thresholds."""

    id: ClassVar[str] = "tag_level_thresholds"
    display_name: ClassVar[str] = "Tag Level Thresholds"
    description: ClassVar[str] = "Filter tags using model-provided per-tag calibrated decision thresholds."
    config_model: ClassVar[type[BaseModel] | None] = TagLevelThresholdsConfig

    threshold_map: Mapping[str, float]
    offset: float = 0.0
    relative_offset: float = 0.0
    fallback: float | None = None
    _plan_cache: dict[
        tuple[int, int, int],
        tuple[
            Sequence[str],
            Mapping[str, Sequence[int]],
            np.ndarray,
            np.ndarray | None,
            tuple[tuple[str, np.ndarray], ...],
        ],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.offset):
            raise TransformError(f"offset must be a finite number, got {self.offset}")
        if not math.isfinite(self.relative_offset) or not (-1.0 <= self.relative_offset <= 1.0):
            raise TransformError(f"relative_offset must be in [-1.0, 1.0], got {self.relative_offset}")
        if self.fallback is not None and (not math.isfinite(self.fallback) or not (0.0 <= self.fallback <= 1.0)):
            raise TransformError(f"fallback must be in [0.0, 1.0], got {self.fallback}")

    @classmethod
    def from_session(
        cls,
        session: ModelSession,
        *,
        offset: float = 0.0,
        relative_offset: float = 0.0,
        fallback: float | None = None,
    ) -> Self:
        """Construct TagLevelThresholds directly from the session's ThresholdTable."""
        if not session.is_tagger:
            raise TransformRequirementError(
                f"TagLevelThresholds requires a tagger, but '{session.model_id}' outputs {session.metadata.output.kind}."
            )

        if session.tagger.thresholds is None:
            raise TransformRequirementError(
                f"Model '{session.model_id}' does not provide per-tag calibrated thresholds (ThresholdProvider). "
                "Use ScoreThresholds instead."
            )

        return cls(
            threshold_map=session.tagger.thresholds.values,
            offset=offset,
            relative_offset=relative_offset,
            fallback=fallback,
        )

    @classmethod
    def is_supported(cls, target: ModelDescriptor | ModelSession) -> bool:
        if isinstance(target, ModelSession):
            return target.is_tagger and target.tagger.thresholds is not None
        return "ThresholdProvider" in target.capabilities

    def _get_compiled_plan(
        self,
        names: Sequence[str],
        category_indices: Mapping[str, Sequence[int]],
        usable_count: int,
    ) -> tuple[np.ndarray, np.ndarray | None, tuple[tuple[str, np.ndarray], ...]]:
        key = (id(names), id(category_indices), usable_count)
        cached = self._plan_cache.get(key)
        if cached is not None and cached[0] is names and cached[1] is category_indices:
            return cached[2], cached[3], cached[4]

        thresh_vec = np.empty(usable_count, dtype=np.float64)
        none_indices: list[int] = []
        rel_scale = 1.0 + self.relative_offset
        has_rel = self.relative_offset != 0.0

        for index in range(usable_count):
            threshold = self.threshold_map.get(names[index], self.fallback)
            if threshold is None:
                thresh_vec[index] = -np.inf
                none_indices.append(index)
            else:
                value = threshold + self.offset
                if has_rel:
                    value *= rel_scale
                thresh_vec[index] = value

        none_mask: np.ndarray | None = None
        if none_indices:
            none_mask = np.zeros(usable_count, dtype=bool)
            none_mask[none_indices] = True

        category_arrays = build_category_arrays(category_indices, usable_count)
        entry = (names, category_indices, thresh_vec, none_mask, category_arrays)
        if len(self._plan_cache) >= 4:
            self._plan_cache.pop(next(iter(self._plan_cache)))
        self._plan_cache[key] = entry
        return thresh_vec, none_mask, category_arrays

    def __call__(self, result: TagResult) -> TagResult:
        if result.is_array_backed:
            assert result._scores is not None and result._tag_names is not None and result._category_indices is not None
            scores = result._scores
            names = result._tag_names
            usable_count = min(len(scores), len(names))
            if usable_count == 0:
                return TagResult(categories={}, extras=dict(result.extras))

            thresh_vec, none_mask, category_arrays = self._get_compiled_plan(
                names,
                result._category_indices,
                usable_count,
            )
            global_mask = scores[:usable_count] >= thresh_vec
            if none_mask is not None:
                global_mask |= none_mask

            return result.select_by_mask(global_mask, category_arrays=category_arrays)

        filtered_categories = {}
        for category, entries in result.categories.items():
            kept = []
            for entry in entries:
                threshold = self.threshold_map.get(entry.tag, self.fallback)
                if threshold is not None:
                    threshold += self.offset
                    if self.relative_offset != 0.0:
                        threshold *= 1.0 + self.relative_offset

                if threshold is None or entry.score >= threshold:
                    kept.append(entry)

            if kept:
                filtered_categories[category] = kept

        return TagResult(categories=filtered_categories, extras=dict(result.extras))