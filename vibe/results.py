"""Result types returned by model inference."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeGuard

import numpy as np

from vibe.metadata import OutputKind
from vibe.settings import serialize_value

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BaseModelResult(ABC):
    """Abstract base class for all inference result objects."""

    output_type: OutputKind = field(init=False)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a standardized dictionary structure."""


# region Result Dataclasses


@dataclass(slots=True)
class TagEntry:
    """A single predicted tag with its confidence score."""

    tag: str
    score: float
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tag": self.tag, "score": self.score}
        if self.extras:
            d["extras"] = self.extras
        return d


@dataclass(slots=True)
class ScoreEntry:
    """A single score component, used inside MultiScoreResult."""

    label: str
    score: float
    score_min: float
    score_max: float
    normalized_score: float
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "score": self.score,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "normalized_score": self.normalized_score,
        }
        if self.extras:
            d["extras"] = self.extras
        return d


@dataclass(slots=True)
class TagResult(BaseModelResult):
    """
    Structured result for tagger model outputs.

    Backed either by a materialized dictionary of TagEntries, or by raw contiguous
    NumPy arrays that lazily instantiate TagEntries on-demand for maximum throughput.
    """

    output_type: Literal[OutputKind.TAGS] = field(default=OutputKind.TAGS, init=False)
    _categories: dict[str, list[TagEntry]] | None = field(default=None)
    _scores: np.ndarray | None = field(default=None)
    _tag_names: Sequence[str] | None = field(default=None)
    _category_indices: Mapping[str, Sequence[int]] | None = field(default=None)
    extras: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        categories: dict[str, list[TagEntry]] | None = None,
        *,
        extras: dict[str, Any] | None = None,
        _scores: np.ndarray | None = None,
        _tag_names: Sequence[str] | None = None,
        _category_indices: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.output_type = OutputKind.TAGS
        self._categories = categories
        self._scores = _scores
        self._tag_names = _tag_names
        self._category_indices = _category_indices
        self.extras = extras if extras is not None else {}

    @classmethod
    def from_arrays(
        cls,
        *,
        tag_names: Sequence[str],
        scores: np.ndarray,
        category_indices: Mapping[str, Sequence[int]],
        extras: dict[str, Any] | None = None,
    ) -> TagResult:
        """High-performance factory: wraps raw score arrays with zero upfront TagEntry allocations."""
        return cls(
            categories=None,
            extras=extras,
            _scores=scores,
            _tag_names=tag_names,
            _category_indices=category_indices,
        )

    def _materialize_categories(self) -> dict[str, list[TagEntry]]:
        """Lazily construct TagEntry lists on first access."""
        if self._categories is not None:
            return self._categories

        if self._scores is None or self._tag_names is None or self._category_indices is None:
            self._categories = {}
            return self._categories

        scores = self._scores
        names = self._tag_names
        usable_count = min(len(scores), len(names))
        materialized: dict[str, list[TagEntry]] = {}

        for cat_name, indices in self._category_indices.items():
            if not indices:
                continue

            valid_indices = [idx for idx in indices if idx < usable_count]
            if not valid_indices:
                continue

            idx_arr = np.array(valid_indices, dtype=np.int32)
            cat_scores = scores[idx_arr]
            sort_order = np.argsort(-cat_scores)  # Fast descending sort in C
            sorted_idx = idx_arr[sort_order]
            sorted_scores = cat_scores[sort_order]

            materialized[cat_name] = [
                TagEntry(tag=names[i], score=float(s)) for i, s in zip(sorted_idx, sorted_scores, strict=False)
            ]

        self._categories = materialized
        return self._categories

    @property
    def categories(self) -> dict[str, list[TagEntry]]:
        """Return tags grouped by category, lazily materializing on-demand."""
        return self._materialize_categories()

    @property
    def tags(self) -> list[TagEntry]:
        """All TagEntry objects flattened across categories, sorted by score descending."""
        all_entries: list[TagEntry] = []
        for entries in self.categories.values():
            all_entries.extend(entries)
        return sorted(all_entries, key=lambda entry: entry.score, reverse=True)

    def category(self, name: str) -> list[TagEntry]:
        """Return tags belonging to a specific category, or an empty list if missing."""
        return self.categories.get(name, [])

    def tag_names(self) -> list[str]:
        """Return all tag names flattened across categories, sorted by score descending."""
        if self._categories is None and self._scores is not None and self._tag_names is not None:
            scores = self._scores
            names = self._tag_names
            usable = min(len(scores), len(names))
            sort_order = np.argsort(-scores[:usable])
            return [names[i] for i in sort_order]

        return [entry.tag for entry in self.tags]

    def as_score_dict(self) -> dict[str, float]:
        """
        Return a flat {tag: score} dictionary sorted descending by score.
        Directly compiled from NumPy arrays when unmaterialized.
        """
        if self._categories is None and self._scores is not None and self._tag_names is not None:
            scores = self._scores
            names = self._tag_names
            usable = min(len(scores), len(names))
            sort_order = np.argsort(-scores[:usable])
            return {names[i]: float(scores[i]) for i in sort_order}

        scores_dict: dict[str, float] = {}
        for entry in self.tags:
            if entry.tag not in scores_dict:
                scores_dict[entry.tag] = entry.score
        return scores_dict

    def as_category_score_dict(self) -> dict[str, dict[str, float]]:
        """
        Return tags grouped by category as {category: {tag: score}}, sorted descending by score.
        """
        if (
            self._categories is None
            and self._scores is not None
            and self._category_indices is not None
            and self._tag_names is not None
        ):
            result: dict[str, dict[str, float]] = {}
            scores = self._scores
            names = self._tag_names
            usable = min(len(scores), len(names))

            for cat_name, indices in self._category_indices.items():
                valid_indices = [idx for idx in indices if idx < usable]
                if not valid_indices:
                    continue
                idx_arr = np.array(valid_indices, dtype=np.int32)
                cat_scores = scores[idx_arr]
                sort_order = np.argsort(-cat_scores)
                sorted_idx = idx_arr[sort_order]
                sorted_scores = cat_scores[sort_order]

                result[cat_name] = {names[i]: float(s) for i, s in zip(sorted_idx, sorted_scores, strict=False)}
            return result

        res: dict[str, dict[str, float]] = {}
        for cat, entries in self.categories.items():
            sorted_entries = sorted(entries, key=lambda e: e.score, reverse=True)
            cat_dict: dict[str, float] = {}
            for entry in sorted_entries:
                if entry.tag not in cat_dict:
                    cat_dict[entry.tag] = entry.score
            res[cat] = cat_dict
        return res

    def filter(self, predicate: Callable[[str, float, str], bool]) -> TagResult:
        """
        Return a new TagResult containing only tags that satisfy the predicate.

        The predicate receives `(tag: str, score: float, category: str) -> bool`.
        Evaluates directly over NumPy arrays with zero throwaway TagEntry allocations.
        """
        # Path A: Fast path directly from NumPy arrays
        if (
            self._categories is None
            and self._scores is not None
            and self._tag_names is not None
            and self._category_indices is not None
        ):
            scores = self._scores
            names = self._tag_names
            usable_count = min(len(scores), len(names))
            filtered: dict[str, list[TagEntry]] = {}

            for cat_name, indices in self._category_indices.items():
                valid_indices = [idx for idx in indices if idx < usable_count]
                if not valid_indices:
                    continue

                idx_arr = np.array(valid_indices, dtype=np.int32)
                cat_scores = scores[idx_arr]
                sort_order = np.argsort(-cat_scores)
                sorted_idx = idx_arr[sort_order]
                sorted_scores = cat_scores[sort_order]

                # ONLY create TagEntry if the predicate passes!
                kept: list[TagEntry] = []
                for i, s in zip(sorted_idx, sorted_scores, strict=False):
                    tag_name = names[i]
                    score_val = float(s)
                    if predicate(tag_name, score_val, cat_name):
                        kept.append(TagEntry(tag=tag_name, score=score_val))

                if kept:
                    filtered[cat_name] = kept

            return TagResult(categories=filtered, extras=dict(self.extras))

        # Path B: Fallback for already-materialized results
        filtered_mat: dict[str, list[TagEntry]] = {}
        for cat, entries in self.categories.items():
            kept = [e for e in entries if predicate(e.tag, e.score, cat)]
            if kept:
                filtered_mat[cat] = kept
        return TagResult(categories=filtered_mat, extras=dict(self.extras))

    def to_dict(self) -> dict[str, Any]:
        """Serialize result, using fast C-level extraction if unmaterialized."""
        if self._categories is not None:
            categories_dict: dict[str, list[dict[str, Any]]] = {}
            for cat, entries in self._categories.items():
                sorted_entries = sorted(entries, key=lambda entry: entry.score, reverse=True)
                categories_dict[cat] = [entry.to_dict() for entry in sorted_entries]
        elif self._scores is not None and self._tag_names is not None and self._category_indices is not None:
            categories_dict = {}
            scores = self._scores
            names = self._tag_names
            usable_count = min(len(scores), len(names))

            for cat_name, indices in self._category_indices.items():
                valid_indices = [idx for idx in indices if idx < usable_count]
                if not valid_indices:
                    continue
                idx_arr = np.array(valid_indices, dtype=np.int32)
                cat_scores = scores[idx_arr]
                sort_order = np.argsort(-cat_scores)
                sorted_idx = idx_arr[sort_order]
                sorted_scores = cat_scores[sort_order]

                categories_dict[cat_name] = [
                    {"tag": names[i], "score": float(s)} for i, s in zip(sorted_idx, sorted_scores, strict=False)
                ]
        else:
            categories_dict = {}

        d: dict[str, Any] = {
            "output_type": self.output_type.value,
            "categories": categories_dict,
        }
        if self.extras:
            d["extras"] = self.extras
        return d


@dataclass(slots=True)
class ScoreResult(BaseModelResult):
    """
    Result from a single-value scoring model (e.g. aesthetic scorer).
    """

    output_type: Literal[OutputKind.SCORE] = field(default=OutputKind.SCORE, init=False)
    score: float
    score_min: float
    score_max: float
    normalized_score: float
    label: str = "score"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "output_type": self.output_type.value,
            "score": self.score,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "normalized_score": self.normalized_score,
            "label": self.label,
        }
        if self.extras:
            d["extras"] = self.extras
        return d


@dataclass(slots=True)
class MultiScoreResult(BaseModelResult):
    """
    Result from a model that returns multiple scores.
    """

    output_type: Literal[OutputKind.MULTI_SCORE] = field(default=OutputKind.MULTI_SCORE, init=False)
    entries: list[ScoreEntry]
    normalized_score: float
    extras: dict[str, Any] = field(default_factory=dict)

    def entry(self, label: str) -> ScoreEntry | None:
        """Return the ScoreEntry for a label, or None if missing."""
        return next((e for e in self.entries if e.label == label), None)

    def as_score_dict(self) -> dict[str, float]:
        """Return a flat {label: raw_score} dict preserving list order."""
        return {e.label: e.score for e in self.entries}

    def as_normalized_dict(self) -> dict[str, float]:
        """Return a flat {label: normalized_score} dict preserving list order."""
        return {e.label: e.normalized_score for e in self.entries}

    def to_dict(self) -> dict[str, Any]:
        d = {
            "output_type": self.output_type.value,
            "entries": [entry.to_dict() for entry in self.entries],
            "normalized_score": self.normalized_score,
        }
        if self.extras:
            d["extras"] = self.extras
        return d


# endregion


@dataclass
class InferenceResultItem:
    """
    Metadata wrapper for one input image and its model prediction.
    """

    index: int
    result: BaseModelResult
    input_ref: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "result": self.result.to_dict(),
        }
        if self.input_ref is not None:
            data["input_ref"] = serialize_value(self.input_ref)
        return data


@dataclass
class InferenceResult:
    """
    Batch envelope returned by session.infer() for one or more images.
    """

    total_inputs: int | None = None
    items: list[InferenceResultItem] = field(default_factory=list)
    memory: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[InferenceResultItem]:
        return iter(self.items)

    def results(self) -> list[BaseModelResult]:
        return [item.result for item in self.items]

    def first(self) -> BaseModelResult:
        if not self.items:
            raise IndexError("Inference batch result is empty.")
        return self.items[0].result

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "total_inputs": self.total_inputs,
            "items": [item.to_dict() for item in self.items],
        }
        if self.memory is not None:
            data["memory"] = self.memory
        return data


# Union type for type hints throughout the codebase
ModelResult = TagResult | ScoreResult | MultiScoreResult


# region Type Narrowing Help


def is_tag_result(result: BaseModelResult) -> TypeGuard[TagResult]:
    """Check if result is a TagResult."""
    return result.output_type == OutputKind.TAGS


def is_score_result(result: BaseModelResult) -> TypeGuard[ScoreResult]:
    """Check if result is a ScoreResult."""
    return result.output_type == OutputKind.SCORE


def is_multi_score_result(result: BaseModelResult) -> TypeGuard[MultiScoreResult]:
    """Check if result is a MultiScoreResult."""
    return result.output_type == OutputKind.MULTI_SCORE


# endregion Type Narrowing Help
