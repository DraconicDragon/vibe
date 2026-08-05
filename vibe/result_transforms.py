"""Typed result transforms applied at inference time."""

from __future__ import annotations

import dataclasses
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

from vibe.backends.base import ArtifactMap
from vibe.backends.char_ip_mapping import apply_character_ip_mapping, resolve_character_ip_mapping
from vibe.plugins.shared.tagger_shared import load_tag_metadata
from vibe.registry import transform_registry
from vibe.results import ModelResult, TagEntry, TagResult
from vibe.tag_categories import TagCategory

logger = logging.getLogger(__name__)

KAOMOJIS = {
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
}

TIn = TypeVar("TIn", bound=ModelResult)
TOut = TypeVar("TOut", bound=ModelResult)


# region Context & Metadata


@dataclass
class TransformContext:
    model_id: str
    artifacts: ArtifactMap
    source: str
    auto_download: bool
    cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _warned_keys: set[str] = field(default_factory=set, repr=False, compare=False)

    def get_cached_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        if key not in self.cache:
            self.cache[key] = loader_fn()
        return self.cache[key]

    def warn_once(self, key: str, message: str) -> None:
        """Log a warning message exactly once for the given key in this context (session)."""
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        logger.warning(message)


@dataclass(frozen=True)
class ParamInfo:
    name: str
    type: str | None
    default: Any
    required: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "required": self.required,
            "description": self.description,
        }


@dataclass(frozen=True)
class TransformInfo:
    transform_id: str
    display_name: str
    description: str
    params: list[ParamInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "display_name": self.display_name,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
        }


# endregion


# region Base Transform


class ResultTransform(ABC, Generic[TIn, TOut]):
    """Base class for result transforms."""

    transform_id: ClassVar[str]
    display_name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    priority: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if not getattr(cls, "transform_id", None):
            raise ValueError(f"{cls.__name__} must define a 'transform_id' ClassVar.")
        transform_registry.register(cls)

    def __call__(self, result: TIn, *, context: TransformContext) -> TOut:
        """Convenience caller forwarding directly to apply()."""
        return self.apply(result, context=context)

    @classmethod
    def describe(cls) -> TransformInfo:
        """Dynamically build transform metadata from dataclass fields."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"ResultTransform subclass '{cls.__name__}' must be decorated with @dataclass.")

        params: list[ParamInfo] = []

        for f in dataclasses.fields(cls):
            if f.metadata.get("internal", False):
                continue

            is_required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            default_val = None if is_required else (f.default if f.default is not dataclasses.MISSING else None)

            params.append(
                ParamInfo(
                    name=f.name,
                    type=getattr(f.type, "__name__", str(f.type)) if f.type else None,
                    default=default_val,
                    required=is_required,
                    description=f.metadata.get("description", ""),
                )
            )

        return TransformInfo(
            transform_id=cls.transform_id,
            display_name=cls.display_name,
            description=cls.description,
            params=params,
        )

    def on_infer_start(self, *, context: TransformContext) -> None:
        """Hook called once per infer call before any outputs are processed."""

    def to_config_dict(self) -> dict[str, Any]:
        """Serialize non-internal dataclass fields for third-party inspection."""
        if not dataclasses.is_dataclass(self):
            return {}

        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(cast(Any, self))
            if not f.metadata.get("internal", False)
        }

    @abstractmethod
    def apply(self, result: TIn, *, context: TransformContext) -> TOut:
        pass


# endregion


# region Transforms


@dataclass(frozen=True)
class CharacterIPMapping(ResultTransform[TagResult, TagResult]):
    transform_id: ClassVar[str] = "character_ip_mapping"
    display_name: ClassVar[str] = "Character IP Mapping"
    description: ClassVar[str] = "Maps copyright tags to character tags from tag results."
    priority: ClassVar[int] = 0

    mapping_file: str | None = field(
        default=None,
        metadata={"description": "Path to custom mapping JSON. If omitted, uses model bundle or HF fallback."},
    )
    _mapping_cache: dict[str, dict[str, list[str]]] = field(
        default_factory=dict, repr=False, compare=False, metadata={"internal": True}
    )

    def apply(self, result: TagResult, *, context: TransformContext) -> TagResult:
        character_entries = result.category(TagCategory.CHARACTER)
        if not character_entries:
            return result

        mapping = self._get_mapping(context)
        if not mapping:
            return result

        mapped = apply_character_ip_mapping([entry.tag for entry in character_entries], mapping)
        if mapped:
            result.extras["character_copyright_mapping"] = mapped
        return result

    def _get_mapping(self, context: TransformContext) -> dict[str, list[str]]:
        cache_key = self.mapping_file or ""

        if cache_key in self._mapping_cache:
            return self._mapping_cache[cache_key]

        cache = resolve_character_ip_mapping(
            manual_path=self.mapping_file,
            allow_download=context.auto_download,
        )
        self._mapping_cache[cache_key] = cache
        return cache


@dataclass(frozen=True)
class CleanTags(ResultTransform[TagResult, TagResult]):
    transform_id: ClassVar[str] = "clean_tags"
    display_name: ClassVar[str] = "Clean Tags"
    description: ClassVar[str] = "Replaces underscores with spaces while preserving kaomoji tags."
    priority: ClassVar[int] = 100  # Should always execute last to not cause issues for other transforms

    def apply(self, result: TagResult, *, context: TransformContext) -> TagResult:
        if not isinstance(result, TagResult):
            return result

        for entries in result.tags.values():
            entries[:] = [TagEntry(tag=_clean_tag_text(entry.tag), score=entry.score) for entry in entries]

        # todo: have transforms see this as another tag category and process it with options to ignore extras or specific one(s)

        mapping = result.extras.get("character_copyright_mapping")
        if mapping is not None:
            result.extras["character_copyright_mapping"] = {
                _clean_tag_text(character): [_clean_tag_text(ip) for ip in ips] for character, ips in mapping.items()
            }

        return result


@dataclass(frozen=True)
class ScoreThresholds(ResultTransform[TagResult, TagResult]):
    transform_id: ClassVar[str] = "score_thresholds"
    display_name: ClassVar[str] = "Score Thresholds"
    description: ClassVar[str] = "Filters tags using global and/or per-category score thresholds."
    priority: ClassVar[int] = 0

    threshold: float = field(default=0.0, metadata={"description": "Global minimum score required."})
    category_thresholds: dict[str | TagCategory, float] | None = field(
        default=None, metadata={"description": "Per-category threshold overrides."}
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0.")
        if self.category_thresholds:
            for cat, val in self.category_thresholds.items():
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"Threshold for category '{cat}' must be between 0.0 and 1.0.")

    def apply(self, result: TagResult, *, context: TransformContext) -> TagResult:
        cat_thresh = self.category_thresholds or {}

        for category, entries in result.tags.items():
            threshold = cat_thresh.get(category, self.threshold)
            filtered = [entry for entry in entries if entry.score >= threshold]

            logger.debug(
                "ScoreThresholds applied category=%s threshold=%.3f kept=%d dropped=%d",
                category,
                threshold,
                len(filtered),
                len(entries) - len(filtered),
            )
            entries[:] = filtered

        return result


@dataclass(frozen=True)
class TagLevelThresholds(ResultTransform[TagResult, TagResult]):
    transform_id: ClassVar[str] = "tag_level_thresholds"
    display_name: ClassVar[str] = "Tag Level Thresholds"
    description: ClassVar[str] = "Filters tags using per-tag thresholds from selected_tags.csv."
    priority: ClassVar[int] = 0

    threshold_column: str = field(default="best_threshold", metadata={"description": "Name of the CSV column."})
    threshold_offset: float = field(
        default=0.0, metadata={"description": "Fixed value added to every tag's threshold."}
    )
    threshold_relative_offset: float = field(
        default=0.0, metadata={"description": "Relative adjustment applied to each threshold."}
    )
    threshold_fallback: float | None = field(
        default=None, metadata={"description": "Fallback threshold when a tag has no per-tag value."}
    )

    _threshold_cache: dict[str, dict[str, float]] = field(
        default_factory=dict, repr=False, compare=False, metadata={"internal": True}
    )
    _threshold_stats_cache: dict[str, tuple[int, int, bool]] = field(
        default_factory=dict, repr=False, compare=False, metadata={"internal": True}
    )

    def __post_init__(self) -> None:
        if self.threshold_offset != 0.0 and self.threshold_relative_offset != 0.0:
            raise ValueError("Use only one of threshold_offset or threshold_relative_offset.")
        if not (-1.0 <= self.threshold_relative_offset <= 1.0):
            raise ValueError("threshold_relative_offset must be in [-1.0, 1.0].")
        if self.threshold_fallback is not None and not (0.0 <= self.threshold_fallback <= 1.0):
            raise ValueError("threshold_fallback must be in [0.0, 1.0].")

    def apply(self, result: TagResult, *, context: TransformContext) -> TagResult:
        if not isinstance(result, TagResult):
            return result

        csv_path = context.artifacts.get_optional("tag_list")
        if csv_path is None:
            context.warn_once(
                key=f"tag-level-thresholds:missing-csv:{context.source}",
                message="TagLevelThresholds requires artifact 'tag_list' (CSV), but it was not found.",
            )
            return result

        threshold_map = self._threshold_map_for_csv(csv_path)
        total_tags, with_thresh, col_present = self._threshold_stats_for_csv(csv_path)

        if not col_present:
            raise RuntimeError(f"CSV at '{csv_path}' missing '{self.threshold_column}' column.")

        missing_count = max(total_tags - with_thresh, 0)
        if missing_count > 0:
            context.warn_once(
                key=f"tag-level-thresholds:partial:{csv_path}",
                message=(
                    f"CSV '{csv_path}' has partial '{self.threshold_column}' data. "
                    f"{missing_count}/{total_tags} tags are missing it."
                ),
            )

        for entries in result.tags.values():
            filtered = []
            for entry in entries:
                threshold = threshold_map.get(entry.tag, self.threshold_fallback)
                if threshold is not None:
                    threshold += self.threshold_offset
                    if self.threshold_relative_offset != 0.0:
                        threshold *= 1.0 + self.threshold_relative_offset
                if threshold is None or entry.score >= threshold:
                    filtered.append(entry)
            entries[:] = filtered

        return result

    def _threshold_map_for_csv(self, csv_path: Path) -> dict[str, float]:
        cache_key = str(csv_path)
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]

        metadata = load_tag_metadata(csv_path, threshold_column=self.threshold_column)
        threshold_map = {
            tag: thr
            for tag, thr in zip(metadata.raw_tag_names, metadata.per_tag_thresholds, strict=False)
            if thr is not None
        }

        self._threshold_stats_cache[cache_key] = (
            len(metadata.raw_tag_names),
            len(threshold_map),
            metadata.threshold_column_present,
        )
        self._threshold_cache[cache_key] = threshold_map
        return threshold_map

    def _threshold_stats_for_csv(self, csv_path: Path) -> tuple[int, int, bool]:
        cache_key = str(csv_path)
        if cache_key not in self._threshold_stats_cache:
            self._threshold_map_for_csv(csv_path)
        return self._threshold_stats_cache.get(cache_key, (0, 0, False))


# endregion


def _clean_tag_text(tag: str) -> str:
    if tag in KAOMOJIS:
        return tag
    return tag.replace("_", " ")
