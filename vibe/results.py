"""
Result types returned by model inference.

Every result is serialisable to a plain dict via .to_dict().
Consumers should check result.output_type before accessing type-specific fields.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, TypeGuard

logger = logging.getLogger(__name__)


class OutputType(str, Enum):
    """The kind of output a model produces."""

    TAGS = "tags"
    SCORE = "score"
    MULTI_SCORE = "multi_score"


# region Result Dataclasses

# todo: turn asdicts into manual to_dict, less processing and simpler and fine anyway since these arent expected to change when done


@dataclass
class TagEntry:
    """A single predicted tag with its confidence score."""

    tag: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TagResult:
    """
    Structured result for tagger model outputs.

    Subclasses should expose category lists such as rating, general, and
    character through categories().
    """

    output_type: Literal[OutputType.TAGS] = field(default=OutputType.TAGS, init=False)
    character_mapping: dict[str, list[str]] | None = None

    def categories(self) -> dict[str, list[TagEntry]]:
        """Return category name -> entries."""
        raise NotImplementedError("TagResult subclasses must implement categories().")

    def category(self, name: str) -> list[TagEntry]:
        """Return tags of one category by name, or an empty list if missing."""
        return self.categories().get(name, [])

    def tag_names(self) -> list[str]:
        """Return all tag names flattened across categories."""
        names: list[str] = []
        for entries in self.categories().values():
            names.extend(entry.tag for entry in entries)
        return names

    def as_score_dict(self) -> dict[str, float]:
        """Return a flat {tag: score} dict sorted by score descending."""
        all_entries: list[TagEntry] = []
        for entries in self.categories().values():
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
        d: dict[str, Any] = {
            "output_type": self.output_type.value,
        }
        for category, entries in self.categories().items():
            sorted_entries = sorted(entries, key=lambda entry: entry.score, reverse=True)
            d[category] = [entry.to_dict() for entry in sorted_entries]
        if self.character_mapping is None:
            return d
        d["character_mapping"] = {
            key: [str(value) for value in values] for key, values in self.character_mapping.items()
        }
        return d


@dataclass
class ScoreResult:
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
        d = asdict(self)
        d["output_type"] = self.output_type.value
        return d


@dataclass(slots=True)
class MultiScoreResult:
    """
    Result from a model that returns multiple scores.

    Attributes
    ----------
    output_type
        Result type identifier used for dispatch and serialization.
    scores
        Mapping from label to score.
    label_order
        Optional label ordering that reflects the model's native output order.
    score_min
        Optional lower bound of the score range.
    score_max
        Optional upper bound of the score range.
    normalized_score
        Optional normalized score in [0, 1] computed by result processors.
    """

    output_type: Literal[OutputType.MULTI_SCORE] = field(default=OutputType.MULTI_SCORE, init=False)
    scores: dict[str, float] = field(default_factory=dict)
    label_order: list[str] = field(default_factory=list)
    score_min: float | None = None
    score_max: float | None = None
    normalized_score: float | None = None

    def __getitem__(self, label: str) -> float:
        return self.scores[label]

    def get(self, label: str, default: float | None = None) -> float | None:
        return self.scores.get(label, default)

    def items(self):
        return self.scores.items()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_type"] = self.output_type.value
        return data


@dataclass
class InferenceResultItem:
    """
    Metadata wrapper for one input image and its model prediction.

    Contains the input's position in the batch, the actual prediction
    (ModelResult), and an optional reference back to the input source.
    Returned as items within InferenceResult."""

    index: int
    result: "ModelResult"
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
    results() to extract all ModelResult objects, and iteration.
    """

    total_inputs: int
    items: list[InferenceResultItem] = field(default_factory=list)
    memory: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def results(self) -> list["ModelResult"]:
        return [item.result for item in self.items]

    def first(self) -> "ModelResult":
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


# endregion Result Dataclasses


# Union type for type hints throughout the codebase
ModelResult = TagResult | ScoreResult | MultiScoreResult


# region Type Narrowing Help


def is_tag_result(result: ModelResult) -> TypeGuard[TagResult]:
    """Check if result is a TagResult (narrows type for type checkers)."""
    return result.output_type == OutputType.TAGS


def is_score_result(result: ModelResult) -> TypeGuard[ScoreResult]:
    """Check if result is a ScoreResult (narrows type for type checkers)."""
    return result.output_type == OutputType.SCORE


def is_multi_score_result(result: ModelResult) -> TypeGuard[MultiScoreResult]:
    """Check if result is a MultiScoreResult (narrows type for type checkers)."""
    return result.output_type == OutputType.MULTI_SCORE


# endregion Type Narrowing Help
