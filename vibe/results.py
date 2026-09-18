"""Result types returned by model inference."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeGuard

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

    Contains tags grouped by category, with flat accessors and filtering utilities.
    """

    output_type: Literal[OutputKind.TAGS] = field(default=OutputKind.TAGS, init=False)
    categories: dict[str, list[TagEntry]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

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
        return [entry.tag for entry in self.tags]

    def as_score_dict(self) -> dict[str, float]:
        """
        Return a flat {tag: score} dictionary sorted descending by score.
        Deduplicates tags by preserving the highest score.
        """
        scores: dict[str, float] = {}
        for entry in self.tags:
            if entry.tag not in scores:
                scores[entry.tag] = entry.score
        return scores

    def as_category_score_dict(self) -> dict[str, dict[str, float]]:
        """
        Return tags grouped by category as {category: {tag: score}}, sorted descending by score.
        """
        result: dict[str, dict[str, float]] = {}
        for cat, entries in self.categories.items():
            sorted_entries = sorted(entries, key=lambda e: e.score, reverse=True)
            cat_dict: dict[str, float] = {}
            for entry in sorted_entries:
                if entry.tag not in cat_dict:
                    cat_dict[entry.tag] = entry.score
            result[cat] = cat_dict
        return result

    def filter(self, predicate: Callable[[TagEntry, str], bool]) -> TagResult:
        """
        Return a new TagResult containing only TagEntries that satisfy the predicate.

        The predicate receives `(entry: TagEntry, category: str) -> bool`.
        """
        filtered: dict[str, list[TagEntry]] = {}
        for cat, entries in self.categories.items():
            kept = [e for e in entries if predicate(e, cat)]
            if kept:
                filtered[cat] = kept
        return TagResult(categories=filtered, extras=dict(self.extras))

    def to_dict(self) -> dict[str, Any]:
        categories_dict: dict[str, list[dict[str, Any]]] = {}
        for cat, entries in self.categories.items():
            sorted_entries = sorted(entries, key=lambda entry: entry.score, reverse=True)
            categories_dict[cat] = [entry.to_dict() for entry in sorted_entries]

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
