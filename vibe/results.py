"""Result types returned by model inference."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeGuard

logger = logging.getLogger(__name__)


class OutputType(str, Enum):
    """The kind of output a model produces."""

    TAGS = "tags"
    SCORE = "score"
    MULTI_SCORE = "multi_score"


@dataclass(slots=True)
class BaseModelResult(ABC):
    """Abstract base class for all inference result objects."""

    output_type: OutputType = field(init=False)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a standardized dictionary structure."""


# region Result Dataclasses


@dataclass(slots=True)
class TagEntry:
    """A single predicted tag with its confidence score."""

    tag: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "score": self.score,
        }


@dataclass(slots=True)
class ScoreEntry:
    """A single score component, used inside MultiScoreResult."""

    label: str
    score: float
    score_min: float
    score_max: float
    normalized_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "normalized_score": self.normalized_score,
        }


@dataclass(slots=True)
class TagResult(BaseModelResult):
    """
    Structured result for tagger model outputs.

    Contains tags grouped by category.
    """

    output_type: Literal[OutputType.TAGS] = field(default=OutputType.TAGS, init=False)
    tags: dict[str, list[TagEntry]] = field(default_factory=dict)
    character_copyright_mapping: dict[str, list[str]] | None = None

    def category(self, name: str) -> list[TagEntry]:
        """Return tags of one category by name, or an empty list if missing."""
        return self.tags.get(name, [])

    def tag_names(self) -> list[str]:
        """Return all tag names flattened across categories in a list."""
        names: list[str] = []
        for entries in self.tags.values():
            names.extend(entry.tag for entry in entries)
        return names

    def as_score_dict(self) -> dict[str, float]:
        """Return a flat {tag: score} dict sorted by score descending. Deduplicates by keeping the highest score."""
        all_entries: list[TagEntry] = []
        for entries in self.tags.values():
            all_entries.extend(entries)

        sorted_entries = sorted(all_entries, key=lambda entry: entry.score, reverse=True)
        scores: dict[str, float] = {}
        for entry in sorted_entries:
            if entry.tag not in scores:
                scores[entry.tag] = entry.score
        return scores

    def to_dict(self) -> dict[str, Any]:
        tags_dict: dict[str, dict[str, float]] = {}
        for category, entries in self.tags.items():
            sorted_entries = sorted(entries, key=lambda entry: entry.score, reverse=True)
            tags_dict[category] = {entry.tag: entry.score for entry in sorted_entries}

        d: dict[str, Any] = {
            "output_type": self.output_type.value,
            "tags": tags_dict,
        }

        if self.character_copyright_mapping is not None:
            d["character_copyright_mapping"] = self.character_copyright_mapping

        return d


@dataclass(slots=True)
class ScoreResult(BaseModelResult):
    """
    Result from a single-value scoring model (e.g. aesthetic scorer).
    """

    output_type: Literal[OutputType.SCORE] = field(default=OutputType.SCORE, init=False)
    score: float
    score_min: float
    score_max: float
    normalized_score: float
    label: str = "score"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_type": self.output_type.value,
            "score": self.score,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "normalized_score": self.normalized_score,
            "label": self.label,
        }


@dataclass(slots=True)
class MultiScoreResult(BaseModelResult):
    """
    Result from a model that returns multiple scores.
    """

    output_type: Literal[OutputType.MULTI_SCORE] = field(default=OutputType.MULTI_SCORE, init=False)
    entries: list[ScoreEntry]
    normalized_score: float

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
        return {
            "output_type": self.output_type.value,
            "entries": [entry.to_dict() for entry in self.entries],
            "normalized_score": self.normalized_score,
        }


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
            data["input_ref"] = self.input_ref
        return data


@dataclass
class InferenceResult:
    """
    Batch envelope returned by session.infer() for one or more images.
    """

    total_inputs: int
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
    return result.output_type == OutputType.TAGS


def is_score_result(result: BaseModelResult) -> TypeGuard[ScoreResult]:
    """Check if result is a ScoreResult."""
    return result.output_type == OutputType.SCORE


def is_multi_score_result(result: BaseModelResult) -> TypeGuard[MultiScoreResult]:
    """Check if result is a MultiScoreResult."""
    return result.output_type == OutputType.MULTI_SCORE


# endregion Type Narrowing Help
