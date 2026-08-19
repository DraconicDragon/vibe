"""Typed result transforms applied at inference time."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Generic, TypeVar, cast, get_args, get_origin

from vibe.backends.base import ArtifactMap
from vibe.backends.char_ip_mapping import apply_character_ip_mapping, resolve_character_ip_mapping
from vibe.exceptions import SessionError, TransformRequirementError
from vibe.features import FeatureSpec, ValueSchema, transform_meta
from vibe.registry import transform_registry
from vibe.results import BaseModelResult, ModelResult, TagEntry, TagResult
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


# region Plugin-Provided Runtime Data


@dataclass(frozen=True)
class PluginData:
    """Base marker for typed, plugin-computed data handed to transforms at runtime."""


@dataclass(frozen=True)
class TagThresholds(PluginData):
    """Per-tag decision thresholds, precomputed by a plugin (F1-optimal, CSV-sourced, etc)."""

    values: dict[str, float]


# endregion

# NOTE: pre py3.12 syntax
TIn = TypeVar("TIn", bound=ModelResult)
TOut = TypeVar("TOut", bound=ModelResult)
TData = TypeVar("TData", bound="PluginData")

# region Context & Metadata


@dataclass
class TransformContext:
    model_id: str
    artifacts: ArtifactMap
    source: str
    auto_download: bool
    token: str | None = None
    _plugin_data: dict[type[PluginData], PluginData] = field(default_factory=dict, repr=False)
    cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _warned_keys: set[str] = field(default_factory=set, repr=False, compare=False)

    def get_plugin_data(self, kind: type[TData]) -> TData | None:
        """Fetch optional plugin data, returning None if not provided."""
        found = self._plugin_data.get(kind)
        return found if isinstance(found, kind) else None

    def require_plugin_data(self, kind: type[TData]) -> TData:
        """
        Fetch required plugin data.
        Raises TransformRequirementError if the model did not provide it,
        triggering the pipeline's fallback policy.
        """
        found = self.get_plugin_data(kind)
        if found is None:
            raise TransformRequirementError(
                f"Model '{self.model_id}' did not provide the required {kind.__name__} data."
            )
        return found

    def get_cached_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        if key not in self.cache:
            self.cache[key] = loader_fn()
        return self.cache[key]

    def warn_once(self, key: str, message: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        logger.warning(message)


# endregion


# region Base Transform


class ResultTransform(ABC, Generic[TIn, TOut]):
    """Base class for result transforms."""

    transform_id: ClassVar[str]
    display_name: ClassVar[str]
    description: ClassVar[str]
    priority: ClassVar[int] = 0
    requires_result_type: ClassVar[type[BaseModelResult]] = BaseModelResult
    output_extras: ClassVar[dict[str, str]] = {}
    entry_extras: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        for attr in ("transform_id", "display_name", "description"):
            val = getattr(cls, attr, None)
            if not val or not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"Concrete transform subclass '{cls.__name__}' must define a non-empty '{attr}' string."
                )

        # Automatically deduce expected result type from generic type signature (e.g., TagResult)
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is ResultTransform or (isinstance(origin, type) and issubclass(origin, ResultTransform)):
                args = get_args(base)
                if args and isinstance(args[0], type) and issubclass(args[0], BaseModelResult):
                    cls.requires_result_type = args[0]
                break

        transform_registry.register(cls)

    def accepts_result(self, result: BaseModelResult) -> bool:
        """Return True if the result matches the required output type for this transform."""
        return isinstance(result, self.requires_result_type)

    def __call__(self, result: TIn, *, context: TransformContext) -> TOut:
        """Convenience caller forwarding directly to apply()."""
        return self.apply(result, context=context)

    @classmethod
    def describe(cls) -> FeatureSpec:
        """Return the unified feature descriptor for this transform class.

        Transform metadata used to be exposed through the separate
        ``TransformInfo``/``TransformOptionSpec`` hierarchy.  Keeping this
        method preserves the introspection entry point while making
        ``FeatureSpec`` the single metadata representation.
        """
        return FeatureSpec.from_transform(cls)

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


# region Concrete Transforms


@dataclass(frozen=True)
class CharacterIPMapping(ResultTransform[TagResult, TagResult]):
    transform_id: ClassVar[str] = "character_ip_mapping"
    display_name: ClassVar[str] = "Character IP Mapping"
    description: ClassVar[str] = "Maps copyright tags to character tags from tag results."
    priority: ClassVar[int] = 0
    output_extras: ClassVar[dict[str, str]] = {
        "character_copyright_mapping": "Mapping of detected character tags to their corresponding copyright/IP tags."
    }

    mapping_file: str | None = field(
        default=None,
        metadata=transform_meta(
            description="Path to custom mapping JSON. If omitted, uses model bundle or HF fallback.",
            schema=ValueSchema(kind="string", nullable=True),
        ),
    )

    def on_infer_start(self, *, context: TransformContext) -> None:
        """Pre-flight check: validate and pre-cache mapping before running inference."""
        try:
            mapping = self._get_mapping(context)
        except Exception as exc:
            # A failure to load is a requirement failure
            raise TransformRequirementError(f"Failed to load character IP mapping data: {exc}") from exc

        if not mapping:
            raise TransformRequirementError(
                f"CharacterIPMapping on model '{context.model_id}' failed to load any valid mapping data."
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
        cache_key = f"char_ip_mapping:{self.mapping_file or 'default'}"
        return context.get_cached_or_load(
            cache_key,
            lambda: resolve_character_ip_mapping(
                manual_path=self.mapping_file,
                allow_download=context.auto_download,
                token=context.token,
            ),
        )


@dataclass(frozen=True)
class CleanTags(ResultTransform[TagResult, TagResult]):
    transform_id: ClassVar[str] = "clean_tags"
    display_name: ClassVar[str] = "Clean Tags"
    description: ClassVar[str] = "Replaces underscores with spaces while preserving kaomoji tags."
    priority: ClassVar[int] = -100  # Low priority default, should execute last to not cause issues for other transforms

    def apply(self, result: TagResult, *, context: TransformContext) -> TagResult:
        if not isinstance(result, TagResult):
            return result

        for entries in result.tags.values():
            entries[:] = [
                TagEntry(
                    tag=_clean_tag_text(entry.tag),
                    score=entry.score,
                    extras=entry.extras,  # Preserve entry metadata
                )
                for entry in entries
            ]

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

    threshold: float = field(
        default=0.0,
        metadata=transform_meta(
            description="Global minimum score required.",
            min_val=0.0,
            max_val=1.0,
            step=0.01,
            schema=ValueSchema(kind="float"),
        ),
    )
    category_thresholds: dict[str | TagCategory, float] | None = field(
        default=None,
        metadata=transform_meta(
            description="Per-category threshold overrides.",
            schema=ValueSchema(
                kind="mapping",
                key_schema=ValueSchema(kind="string"),
                value_schema=ValueSchema(kind="float"),
                allow_custom_keys=True,
                nullable=True,
                known_keys_source="output_categories",
            ),
        ),
    )

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise SessionError("threshold must be a number between 0.0 and 1.0.")
        if not math.isfinite(float(self.threshold)) or not (0.0 <= self.threshold <= 1.0):
            raise SessionError("threshold must be between 0.0 and 1.0.")

        if self.category_thresholds is not None:
            normalized_map: dict[str, float] = {}
            seen_canonical: dict[str, Any] = {}

            for cat, val in self.category_thresholds.items():
                canonical_cat = str(cat.value if isinstance(cat, Enum) else cat)
                if canonical_cat in seen_canonical and seen_canonical[canonical_cat] != cat:
                    raise SessionError(
                        f"Duplicate normalized category collision in category_thresholds: '{cat}' conflicts with '{seen_canonical[canonical_cat]}'."
                    )
                seen_canonical[canonical_cat] = cat

                if isinstance(val, bool):
                    raise SessionError(f"Threshold for category '{canonical_cat}' must be a number.")
                try:
                    float_val = float(val)
                except (TypeError, ValueError) as exc:
                    raise SessionError(f"Threshold for category '{canonical_cat}' must be a number.") from exc
                if not math.isfinite(float_val) or not (0.0 <= float_val <= 1.0):
                    raise SessionError(f"Threshold for category '{canonical_cat}' must be between 0.0 and 1.0.")
                normalized_map[canonical_cat] = float_val

            object.__setattr__(self, "category_thresholds", normalized_map)

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
    description: ClassVar[str] = "Filters tags using per-tag thresholds provided by the model plugin."
    priority: ClassVar[int] = 0

    threshold_offset: float = field(
        default=0.0,
        metadata=transform_meta(
            description="Fixed value added to every tag's threshold.",
            min_val=-1.0,
            max_val=1.0,
            step=0.01,
            schema=ValueSchema(kind="float"),
        ),
    )
    threshold_relative_offset: float = field(
        default=0.0,
        metadata=transform_meta(
            description="Relative adjustment applied to each threshold.",
            min_val=-1.0,
            max_val=1.0,
            step=0.01,
            schema=ValueSchema(kind="float"),
        ),
    )
    threshold_fallback: float | None = field(
        default=None,
        metadata=transform_meta(
            description="Fallback threshold when a tag has no per-tag value.",
            min_val=0.0,
            max_val=1.0,
            step=0.01,
            schema=ValueSchema(kind="float", nullable=True),
        ),
    )

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
            for value in (self.threshold_offset, self.threshold_relative_offset)
        ):
            raise SessionError("Threshold offsets must be finite numbers.")
        if self.threshold_offset != 0.0 and self.threshold_relative_offset != 0.0:
            raise SessionError("Use only one of threshold_offset or threshold_relative_offset.")
        if not (-1.0 <= self.threshold_relative_offset <= 1.0):
            raise SessionError("threshold_relative_offset must be in [-1.0, 1.0].")
        if self.threshold_fallback is not None and (
            isinstance(self.threshold_fallback, bool)
            or not isinstance(self.threshold_fallback, (int, float))
            or not math.isfinite(float(self.threshold_fallback))
            or not (0.0 <= self.threshold_fallback <= 1.0)
        ):
            raise SessionError("threshold_fallback must be in [0.0, 1.0].")

    def on_infer_start(self, *, context: TransformContext) -> None:
        # Fails with TransformRequirementError automatically if data is missing
        context.require_plugin_data(TagThresholds)

    def apply(self, result: TagResult, *, context: TransformContext) -> TagResult:
        # Data is guaranteed to exist here if on_infer_start passed
        data = context.require_plugin_data(TagThresholds)
        threshold_map = data.values

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


# endregion


def _clean_tag_text(tag: str) -> str:
    if tag in KAOMOJIS:
        return tag
    return tag.replace("_", " ")
