"""
Result types returned by model inference.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeGuard, cast

logger = logging.getLogger(__name__)


class OutputType(str, Enum):
    """The kind of output a model produces."""

    TAGS = "tags"
    SCORE = "score"
    MULTI_SCORE = "multi_score"


@dataclass
class BaseModelResult(ABC):
    """Abstract base class for all inference result objects."""

    output_type: OutputType = field(init=False)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


# region Result Dataclasses


@dataclass
class TagEntry:
    """A single predicted tag with its confidence score."""

    tag: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "score": self.score,
        }


@dataclass
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
        """Return a flat {tag: score} dict sorted by score descending."""
        all_entries: list[TagEntry] = []
        for entries in self.tags.values():
            all_entries.extend(entries)

        sorted_entries = sorted(all_entries, key=lambda entry: entry.score, reverse=True)
        scores: dict[str, float] = {}
        for entry in sorted_entries:
            if entry.tag in scores:
                # Duplicate tag: first occurrence already has the highest score
                # shouldn't happen but checking and logging anyway
                logger.warning(
                    f"Duplicate tag '{entry.tag}' (score: {entry.score:.3f}) in result; keeping first occurrence"
                )
                continue
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


@dataclass
class ScoreResult(BaseModelResult):
    """
    Result from a single-value scoring model (e.g. aesthetic scorer).

    Attributes:
        score:              The predicted score.
        score_min:          The minimum of the model's output range (informational).
        score_max:          The maximum of the model's output range (informational).
        normalized_score:   Optional normalized score in [0, 1] computed by result processors.
        label:              Human-readable label for what the score means. Default: "score".
    """

    output_type: Literal[OutputType.SCORE] = field(default=OutputType.SCORE, init=False)
    score: float
    score_min: float
    score_max: float
    normalized_score: float | None = None
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

    Acts like a dictionary mapping both integer indices and string labels to their respective float scores.
    """

    output_type: Literal[OutputType.MULTI_SCORE] = field(default=OutputType.MULTI_SCORE, init=False)
    scores: dict[int, float] | list[float] = field(default_factory=dict)
    label_map: dict[int, str] | None = None
    label_order: list[str] | None = None
    score_min: float | None = None
    score_max: float | None = None
    normalized_score: float | None = None

    def __post_init__(self) -> None:
        self.scores = self._normalize_scores(self.scores)

        if self.label_map is None:
            self.label_map = {index: f"score_{index + 1}" for index in self.scores.keys()}  # type: ignore[union-attr]
        else:
            self.label_map = {int(index): str(label) for index, label in self.label_map.items()}

        if self.label_order is not None:
            self.label_order = [str(label) for label in self.label_order]

    def __getitem__(self, key: int | str) -> float:
        if isinstance(key, str):
            label_map = self._label_to_index_map()
            if key not in label_map:
                raise KeyError(key)
            return self.as_index_score_dict()[label_map[key]]
        return self.as_index_score_dict()[key]

    def get(self, key: int | str, default: float | None = None) -> float | None:
        if isinstance(key, str):
            label_map = self._label_to_index_map()
            index = label_map.get(key)
            if index is None:
                return default
            return self.as_index_score_dict().get(index, default)
        return self.as_index_score_dict().get(key, default)

    def items(self):
        """Yield (label, score) pairs."""
        return self.as_label_score_dict().items()

    def as_index_score_dict(self) -> dict[int, float]:
        return cast(dict[int, float], self.scores)

    def as_label_score_dict(self) -> dict[str, float]:
        """Return a {label: score} dictionary, ordered by label_order if specified."""
        scores = self.as_index_score_dict()
        label_map = self.label_map
        label_order = self.label_order
        assert label_map is not None

        label_scores = {label_map[index]: score for index, score in scores.items()}
        if label_order is None:
            return label_scores

        ordered_scores: dict[str, float] = {}
        for label in label_order:
            if label in label_scores and label not in ordered_scores:
                ordered_scores[label] = label_scores[label]
        for label, score in label_scores.items():
            if label not in ordered_scores:
                ordered_scores[label] = score
        return ordered_scores

    def _normalize_scores(self, scores: dict[int, float] | dict[str, float] | list[float]) -> dict[int, float]:
        if isinstance(scores, list):
            return {index: float(score) for index, score in enumerate(scores)}

        if scores and not all(isinstance(index, int) for index in scores.keys()):
            return {index: float(score) for index, score in enumerate(scores.values())}

        normalized: dict[int, float] = {}
        for index, score in sorted(scores.items(), key=lambda item: item[0]):
            normalized[int(index)] = float(score)
        return normalized

    def _label_to_index_map(self) -> dict[str, int]:
        assert self.label_map is not None
        return {label: index for index, label in self.label_map.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_type": self.output_type.value,
            "scores": self.scores,
            "label_map": self.label_map,
            "label_order": self.label_order,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "normalized_score": self.normalized_score,
        }


# endregion Result Dataclasses


@dataclass
class InferenceResultItem:
    """
    Metadata wrapper for one input image and its model prediction.

    Contains the input's position in the batch, the actual prediction
    (ModelResult), and an optional reference back to the input source.
    Returned as items within InferenceResult.
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

    Each input image produces one InferenceResultItem in the items list.
    Supports both single and batch operations via the same structure.
    Provides convenience methods: first() for single-image workflows,
    results() to extract all BaseModelResult objects, and iteration.
    """

    total_inputs: int
    items: list[InferenceResultItem] = field(default_factory=list)
    memory: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
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
    """Check if result is a TagResult (narrows type for type checkers)."""
    return result.output_type == OutputType.TAGS


def is_score_result(result: BaseModelResult) -> TypeGuard[ScoreResult]:
    """Check if result is a ScoreResult (narrows type for type checkers)."""
    return result.output_type == OutputType.SCORE


def is_multi_score_result(result: BaseModelResult) -> TypeGuard[MultiScoreResult]:
    """Check if result is a MultiScoreResult (narrows type for type checkers)."""
    return result.output_type == OutputType.MULTI_SCORE


# endregion Type Narrowing Help
