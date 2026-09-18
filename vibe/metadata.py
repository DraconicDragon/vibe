"""
Public metadata schemas describing capabilities and requirements of a model.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from vibe.settings import SettingGroupSpec, serialize_value

if TYPE_CHECKING:
    from vibe.backends.base import ModelVariant


class Modality(StrEnum):
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"


class OutputKind(StrEnum):
    TAGS = "tags"
    SCORE = "score"
    MULTI_SCORE = "multi_score"
    SCORE_VECTOR = "score_vector"
    EMBEDDING = "embedding"
    RAW = "raw"


class LabelSource(StrEnum):
    MODEL = "model"
    USER = "user"
    NONE = "none"


class ScoreSemantics(StrEnum):
    PROBABILITY = "probability"
    SIMILARITY = "similarity"
    LOGITS = "logits"
    UNCALIBRATED = "uncalibrated"


class StandardConsumerSettingId(StrEnum):
    """Standard identifiers for consumer-facing recommended settings."""

    TAG_FILTER = "tagger.score_filter"


@dataclass(frozen=True, kw_only=True)
class ModelIdentity:
    """Core identification strings for a registered model."""

    model_id: str
    display_name: str
    description: str


class TagFilterRecommendation(BaseModel):
    """Standardized semantic recommendation for downstream thresholding filters."""

    global_threshold: float | None = None
    category_thresholds: dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True, kw_only=True)
class InputSpec:
    modalities: tuple[Modality, ...]
    text_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modalities": [m.value for m in self.modalities],
            "text_source": self.text_source,
        }


@dataclass(frozen=True, kw_only=True)
class OutputSpec:
    kind: OutputKind
    label_source: LabelSource = LabelSource.NONE
    score_semantics: ScoreSemantics = ScoreSemantics.UNCALIBRATED
    categories: tuple[str | Enum, ...] = ()
    output_extras: Mapping[str, str] = field(default_factory=dict)
    entry_extras: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "label_source": self.label_source.value,
            "score_semantics": self.score_semantics.value,
            "categories": [c.value if isinstance(c, Enum) else str(c) for c in self.categories],
            "output_extras": dict(self.output_extras),
            "entry_extras": dict(self.entry_extras),
        }


@dataclass(frozen=True, kw_only=True)
class ConsumerSettingSpec:
    """Metadata for settings that third-party tools should consume, isolated from model runtime execution."""

    id: str
    display_name: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)
    default: Any = None
    recommended: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id.value if isinstance(self.id, Enum) else str(self.id),
            "display_name": self.display_name,
            "description": self.description,
            "schema": dict(self.schema),
            "default": serialize_value(self.default),
            "recommended": serialize_value(self.recommended),
        }


# Standard Domain Objects for Typed Attributes


@dataclass(frozen=True, kw_only=True)
class LabelInfo:
    index: int
    name: str
    category: str
    threshold: float | None = None
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "category": self.category.value if isinstance(self.category, Enum) else str(self.category),
            "threshold": self.threshold,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, kw_only=True)
class LabelCatalog:
    labels: tuple[LabelInfo, ...]
    categories: tuple[str | Enum, ...]

    @functools.cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(label.name for label in self.labels)

    @functools.cached_property
    def by_name(self) -> Mapping[str, LabelInfo]:
        return {label.name: label for label in self.labels}

    @functools.cached_property
    def by_category(self) -> Mapping[str, tuple[LabelInfo, ...]]:
        cat_map: dict[str, list[LabelInfo]] = {
            (cat.value if isinstance(cat, Enum) else str(cat)): [] for cat in self.categories
        }
        for label in self.labels:
            cat_key = label.category.value if isinstance(label.category, Enum) else str(label.category)
            cat_map.setdefault(cat_key, []).append(label)
        return {cat: tuple(items) for cat, items in cat_map.items()}

    @functools.cached_property
    def by_index(self) -> Mapping[int, LabelInfo]:
        return {label.index: label for label in self.labels}

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": [label.to_dict() for label in self.labels],
            "categories": [c.value if isinstance(c, Enum) else str(c) for c in self.categories],
        }


@dataclass(frozen=True, kw_only=True)
class ThresholdTable:
    values: Mapping[str, float]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "source": self.source,
        }


@dataclass(frozen=True, kw_only=True)
class CalibrationTable:
    """Mapping from raw uncalibrated scores to normalized percentiles."""

    x: Sequence[float] | Any
    y: Sequence[float] | Any
    source: str

    def to_dict(self) -> dict[str, Any]:
        def _to_list(val: Any) -> list[float]:
            return val.tolist() if hasattr(val, "tolist") else list(val)

        return {
            "x": _to_list(self.x),
            "y": _to_list(self.y),
            "source": self.source,
        }


@dataclass(frozen=True, kw_only=True)
class ModelProfile:
    """A collection of capabilities and requirements representing a specific archetype of model."""

    model_types: tuple[str, ...]
    input_spec: InputSpec
    output_spec: OutputSpec
    consumer_settings: tuple[ConsumerSettingSpec, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ModelDescriptor:
    schema_version: str
    identity: ModelIdentity
    family_name: str
    model_types: tuple[str, ...]
    capabilities: tuple[str, ...]
    input: InputSpec
    output: OutputSpec
    settings: tuple[SettingGroupSpec, ...]
    consumer_settings: tuple[ConsumerSettingSpec, ...]
    default_repo_id: str | None
    variants: tuple[ModelVariant, ...]

    def get_consumer_setting(self, setting_id: str | Enum) -> ConsumerSettingSpec | None:
        """Find a consumer setting specification by its string ID or enum."""
        sid = setting_id.value if isinstance(setting_id, Enum) else str(setting_id)
        for cs in self.consumer_settings:
            if cs.id == sid:
                return cs
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": {
                "model_id": self.identity.model_id,
                "display_name": self.identity.display_name,
                "description": self.identity.description,
            },
            "family_name": self.family_name,
            "model_types": list(self.model_types),
            "capabilities": list(self.capabilities),
            "input": self.input.to_dict(),
            "output": self.output.to_dict(),
            "settings": [s.to_dict() for s in self.settings],
            "consumer_settings": [cs.to_dict() for cs in self.consumer_settings],
            "default_repo_id": self.default_repo_id,
            "variants": [
                {
                    "variant_id": v.variant_id,
                    "backend": v.backend.value,
                    "description": v.description,
                    "repo_id": v.repo_id,
                    "hf_subdir": v.hf_subdir,
                    "artifacts": [
                        {
                            "id": a.id,
                            "name": a.name,
                            "role": a.role.value,
                            "required": a.required,
                            "repo_id": a.repo_id,
                            "hf_subdir": a.hf_subdir,
                        }
                        for a in v.artifacts
                    ],
                }
                for v in self.variants
            ],
        }
