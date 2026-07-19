"""Result types returned by model inference."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, ItemsView, Iterator, Literal, TypeGuard

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
        pass


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
                logger.warning(
                    "Duplicate tag '%s' (score: %.3f) in result; keeping first occurrence",
                    entry.tag,
                    entry.score,
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


@dataclass(slots=True)
class ScoreResult(BaseModelResult):
    """
    Result from a single-value scoring model (e.g. aesthetic scorer).
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

    Accepts lists, int-keyed dicts, or string-keyed dicts (where strings act as labels).
    """

    output_type: Literal[OutputType.MULTI_SCORE] = field(default=OutputType.MULTI_SCORE, init=False)
    scores_input: InitVar[dict[int, float] | dict[str, float] | list[float]]
    label_map: dict[int, str] | None = None
    label_order: list[str] | None = None
    score_min: float | None = None
    score_max: float | None = None
    normalized_score: float | None = None  # Optional summary score representing the entire set

    # Strict physical representations
    scores: dict[int, float] = field(init=False)
    _label_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self, scores_input: dict[int, float] | dict[str, float] | list[float]) -> None:
        norm_scores, inferred_labels = self._normalize_scores(scores_input)
        self.scores = norm_scores

        self.label_map = self._build_and_validate_label_map(
            norm_scores=self.scores,
            explicit_map=self.label_map,
            inferred_map=inferred_labels,
        )

        if self.label_map is not None:
            self._label_to_index = {label: index for index, label in self.label_map.items()}
        else:
            self._label_to_index = {}

        if self.label_order is not None:
            self.label_order = [str(label) for label in self.label_order]
            self._validate_label_order(self.label_map, self.label_order)

    def _normalize_scores(
        self, raw_scores: dict[int, float] | dict[str, float] | list[float]
    ) -> tuple[dict[int, float], dict[int, str] | None]:
        """Convert arbitrary score inputs into a strict dict[int, float]."""
        if isinstance(raw_scores, list):
            return {i: float(v) for i, v in enumerate(raw_scores)}, None

        if isinstance(raw_scores, dict):
            if not raw_scores:
                return {}, None

            keys = list(raw_scores.keys())
            is_int = all(isinstance(k, int) for k in keys)
            is_str = all(isinstance(k, str) for k in keys)

            if not (is_int or is_str):
                raise ValueError("MultiScoreResult 'scores' dictionary must have all integer keys or all string keys.")

            if is_str:
                norm_scores: dict[int, float] = {}
                inferred_labels: dict[int, str] = {}
                for i, (k, v) in enumerate(raw_scores.items()):
                    norm_scores[i] = float(v)
                    inferred_labels[i] = str(k)
                return norm_scores, inferred_labels

            return {int(k): float(v) for k, v in raw_scores.items()}, None

        raise TypeError(f"Unsupported scores type for MultiScoreResult: {type(raw_scores)}")

    def _build_and_validate_label_map(
        self,
        norm_scores: dict[int, float],
        explicit_map: dict[int, str] | None,
        inferred_map: dict[int, str] | None,
    ) -> dict[int, str] | None:
        """Determine the final label map and ensure it perfectly matches the scores without duplicates."""
        if explicit_map is not None:
            if inferred_map is not None:
                raise ValueError("Cannot provide an explicit 'label_map' when 'scores' is a string-keyed dictionary.")
            final_map = {int(k): str(v) for k, v in explicit_map.items()}
        elif inferred_map is not None:
            final_map = inferred_map
        else:
            # can use integer indices results[0] etc
            return None

        score_keys = set(norm_scores.keys())
        map_keys = set(final_map.keys())

        # Validate index completeness
        missing = score_keys - map_keys
        if missing:
            raise ValueError(f"label_map is missing labels for indices: {missing}")

        extra = map_keys - score_keys
        if extra:
            raise ValueError(f"label_map contains extra indices not present in scores: {extra}")

        # Validate label uniqueness
        seen = set()
        duplicates = set()
        for label in final_map.values():
            if label in seen:
                duplicates.add(label)
            seen.add(label)

        if duplicates:
            raise ValueError(f"label_map contains duplicate labels: {duplicates}")

        return final_map

    def _validate_label_order(self, final_map: dict[int, str] | None, order: list[str]) -> None:
        """Ensure all requested labels in label_order actually exist."""
        if final_map is None:
            raise ValueError("Cannot specify label_order when no labels or label_map are available.")
        known_labels = set(final_map.values())
        unknown = set(order) - known_labels
        if unknown:
            raise ValueError(f"label_order contains unknown labels: {unknown}")

    def __contains__(self, key: int | str) -> bool:
        if isinstance(key, str):
            return key in self._label_to_index
        return key in self.scores

    def __getitem__(self, key: int | str) -> float:
        if isinstance(key, str):
            index = self._label_to_index.get(key)
            if index is None:
                raise KeyError(f"Label '{key}' not found in MultiScoreResult.")
            return self.scores[index]
        return self.scores[key]

    def get(self, key: int | str, default: float | None = None) -> float | None:
        if isinstance(key, str):
            index = self._label_to_index.get(key)
            if index is None:
                return default
            return self.scores.get(index, default)
        return self.scores.get(key, default)

    def items(self) -> ItemsView[str, float] | ItemsView[int, float]:
        """Yield (label, score) or (index, score) pairs based on label availability."""
        if self.label_map is not None:
            return self.as_label_score_dict().items()
        return self.scores.items()

    def as_index_score_dict(self) -> dict[int, float]:
        """Return the strictly normalized integer-indexed dictionary."""
        return self.scores

    def as_label_score_dict(self) -> dict[str, float]:
        """Return a {label: score} dictionary, ordered by label_order if specified."""
        if self.label_map is None:
            raise ValueError("No label mapping was parsed or supplied for this MultiScoreResult.")

        label_scores = {self.label_map[index]: score for index, score in self.scores.items()}
        if not self.label_order:
            return label_scores

        ordered_scores: dict[str, float] = {}
        for label in self.label_order:
            if label in label_scores:
                ordered_scores[label] = label_scores[label]
        for label, score in label_scores.items():
            if label not in ordered_scores:
                ordered_scores[label] = score
        return ordered_scores

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
