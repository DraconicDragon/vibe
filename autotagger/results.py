"""
Result types returned by model inference.

Every result is serialisable to a plain dict via .to_dict().
Consumers should check result.output_type before accessing type-specific fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, TypeGuard


class OutputType(str, Enum):
    """The kind of output a model produces."""

    TAGS = "tags"
    SCORE = "score"
    MULTI_SCORE = "multi_score"


# region Result Dataclasses


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
    Base result type for tagger models with named score categories.

    Concrete tagger results should subclass this and expose explicit fields for
    each category (for example: rating/general/character).
    """

    output_type: Literal[OutputType.TAGS] = field(default=OutputType.TAGS, init=False)
    character_mapping: dict[str, list[str]] | None = None

    def categories(self) -> dict[str, list[TagEntry]]:
        """Return category name -> entries for this result."""
        raise NotImplementedError("TagResult subclasses must implement categories().")

    def category(self, name: str) -> list[TagEntry]:
        """Return one category by name, or an empty list if missing."""
        return self.categories().get(name, [])

    def tag_names(self) -> list[str]:
        """Convenience: flattened tag names from all categories."""
        names: list[str] = []
        for entries in self.categories().values():
            names.extend(entry.tag for entry in entries)
        return names

    def as_score_dict(self) -> dict[str, float]:
        """Convenience: {tag: score} for all tags in all categories."""
        scores: dict[str, float] = {}
        for entries in self.categories().values():
            for entry in entries:
                scores[entry.tag] = entry.score
        return scores

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "output_type": self.output_type.value,
        }
        for category, entries in self.categories().items():
            d[category] = [entry.to_dict() for entry in entries]
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
        score:      The predicted score.
        score_min:  The minimum of the model's output range (informational).
        score_max:  The maximum of the model's output range (informational).
        label:      Optional human-readable label for what the score means.
    """

    output_type: Literal[OutputType.SCORE] = OutputType.SCORE
    score: float = 0.0
    score_min: float = 0.0
    score_max: float = 1.0
    label: str = "score"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_type"] = self.output_type.value
        return d


@dataclass
class MultiScoreResult:
    """
    Result from a model that outputs multiple named scores
    (e.g. good/normal/bad).

    Attributes:
        scores:     Ordered dict of {label: value}.
        score_min:  Minimum of each score's range.
        score_max:  Maximum of each score's range.
    """

    # todo: no labels like scoreresult? where does scoreresult get label from? unify more to be like scoreresult?

    output_type: Literal[OutputType.MULTI_SCORE] = OutputType.MULTI_SCORE
    scores: dict[str, float] = field(default_factory=dict)
    score_min: float = 0.0
    score_max: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_type"] = self.output_type.value
        return d


# endregion Result Dataclasses


# Union type for type hints throughout the codebase
InferenceResult = TagResult | ScoreResult | MultiScoreResult


# region Type Narrowing Help


def is_tag_result(result: InferenceResult) -> TypeGuard[TagResult]:
    """Check if result is a TagResult (narrows type for type checkers)."""
    return result.output_type == OutputType.TAGS


def is_score_result(result: InferenceResult) -> TypeGuard[ScoreResult]:
    """Check if result is a ScoreResult (narrows type for type checkers)."""
    return result.output_type == OutputType.SCORE


def is_multi_score_result(result: InferenceResult) -> TypeGuard[MultiScoreResult]:
    """Check if result is a MultiScoreResult (narrows type for type checkers)."""
    return result.output_type == OutputType.MULTI_SCORE


# endregion Type Narrowing Help
