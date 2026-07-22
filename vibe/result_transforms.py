"""Typed result transforms applied at inference time."""

from __future__ import annotations

import dataclasses
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, TypeVar, Union, cast

import numpy as np

from vibe.backends.base import ArtifactMap
from vibe.backends.char_ip_mapping import apply_character_ip_mapping, resolve_character_ip_mapping
from vibe.plugins.shared.tagger_shared import load_tag_metadata
from vibe.registry import transform_registry
from vibe.results import ModelResult, MultiScoreResult, ScoreResult, TagEntry, TagResult

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
        fields_dict: dict[str, dataclasses.Field[Any]] = getattr(cls, "__dataclass_fields__", {})

        for f in fields_dict.values():
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
        pass

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

        character_entries = result.category("character")
        if not character_entries:
            result.character_copyright_mapping = None
            return result

        mapping = self._get_mapping(context)
        if not mapping:
            result.character_copyright_mapping = None
            return result

        mapped = apply_character_ip_mapping([entry.tag for entry in character_entries], mapping)
        result.character_copyright_mapping = mapped or None
        return result

    def _get_mapping(self, context: TransformContext) -> dict[str, list[str]]:
        cache_key = self.mapping_file or "default"
        if cache_key in self._mapping_cache:
            return self._mapping_cache[cache_key]

        tag_list_path = context.artifacts.get_optional("tag_list")
        model_dir = tag_list_path.parent if tag_list_path else Path.cwd()

        cache = resolve_character_ip_mapping(
            model_dir=model_dir,
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
    priority: ClassVar[int] = 100  # Will always execute last due to pipeline sorting

    def apply(self, result: TagResult, *, context: TransformContext) -> TagResult:
        if not isinstance(result, TagResult):
            return result

        for entries in result.tags.values():
            entries[:] = [TagEntry(tag=_clean_tag_text(entry.tag), score=entry.score) for entry in entries]

        if result.character_copyright_mapping is not None:
            result.character_copyright_mapping = {
                _clean_tag_text(character): [_clean_tag_text(ip) for ip in ips]
                for character, ips in result.character_copyright_mapping.items()
            }

        return result


@dataclass(frozen=True)
class ScoreThresholds(ResultTransform[TagResult, TagResult]):
    transform_id: ClassVar[str] = "score_thresholds"
    display_name: ClassVar[str] = "Score Thresholds"
    description: ClassVar[str] = "Filters tags using global and/or per-category score thresholds."
    priority: ClassVar[int] = 0

    threshold: float = field(default=0.0, metadata={"description": "Global minimum score required."})
    category_thresholds: dict[str, float] | None = field(
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

        for category, entries in result.tags.items():
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


@dataclass(frozen=True)
class MultiScoreToScore(ResultTransform[MultiScoreResult, ScoreResult]):
    transform_id: ClassVar[str] = "multi_score_to_score"
    display_name: ClassVar[str] = "Multi-Score to Score"
    description: ClassVar[str] = "Collapses a MultiScoreResult into a single normalized ScoreResult."
    priority: ClassVar[int] = 0

    use_samples_percentile: bool = field(
        default=False, metadata={"description": "Use bundled samples.npz to map score to a percentile."}
    )
    label: str = field(default="score", metadata={"description": "Label attached to the output ScoreResult."})

    def apply(self, result: MultiScoreResult, *, context: TransformContext) -> ScoreResult:
        if not isinstance(result, MultiScoreResult):
            raise TypeError(f"Expected MultiScoreResult, got {type(result)}")

        if not result.scores:
            context.warn_once(f"MultiScoreToScore:empty-scores:{self.label}", "Empty scores; returning zero.")
            return ScoreResult(score=0.0, score_min=0.0, score_max=1.0, label=self.label, normalized_score=0.0)

        norm_transform = NormalizedScore(use_samples_percentile=self.use_samples_percentile)
        normalized_result = norm_transform.apply(result, context=context)

        score = (
            normalized_result.normalized_score
            if isinstance(normalized_result, MultiScoreResult) and normalized_result.normalized_score is not None
            else 0.0
        )

        return ScoreResult(
            score=score,
            score_min=0.0,
            score_max=1.0,
            label=self.label,
            normalized_score=score,
        )


@dataclass(frozen=True)
class NormalizedScore(ResultTransform[Union[ScoreResult, MultiScoreResult], Union[ScoreResult, MultiScoreResult]]):
    transform_id: ClassVar[str] = "normalized_score"
    display_name: ClassVar[str] = "Normalized Score"
    description: ClassVar[str] = "Attaches a normalized score in [0, 1]."
    priority: ClassVar[int] = 0

    use_samples_percentile: bool = field(
        default=False, metadata={"description": "Use bundled samples.npz for percentile normalization."}
    )

    def apply(
        self, result: Union[ScoreResult, MultiScoreResult], *, context: TransformContext
    ) -> Union[ScoreResult, MultiScoreResult]:
        if isinstance(result, ScoreResult):
            result.normalized_score = self._normalize_scalar(result.score, result.score_min, result.score_max)
            return result

        if not isinstance(result, MultiScoreResult):
            return result

        result.normalized_score = self._normalize_multiscore(result, context=context)
        return result

    def _normalize_scalar(self, score: float, score_min: float, score_max: float) -> float:
        if score_max <= score_min:
            return 0.0
        return float(np.clip((score - score_min) / (score_max - score_min), 0.0, 1.0))

    def _normalize_multiscore(self, result: MultiScoreResult, *, context: TransformContext) -> float:
        if not result.scores:
            return 0.0

        weighted_mean = self._weighted_mean(result)
        samples_path = context.artifacts.get_optional("samples")

        if self.use_samples_percentile:
            if samples_path is None:
                raise FileNotFoundError("NormalizedScore: use_samples_percentile=True but artifact 'samples' missing.")
            x, y = self._get_samples_table(samples_path, context=context)
            return self._interp_percentile(weighted_mean, x, y)

        if samples_path is not None:
            context.warn_once(
                f"NormalizedScore:samples-detected-not-used:{samples_path}",
                "Optional 'samples' artifact detected but ignored. Set use_samples_percentile=True.",
            )

        max_v = float(max(len(result.scores) - 1, 1))
        return float(np.clip((weighted_mean - 0.0) / max_v, 0.0, 1.0)) if max_v > 0.0 else 0.0

    def _weighted_mean(self, result: MultiScoreResult) -> float:
        scores = result.as_index_score_dict()
        weighted_mean = 0.0

        if result.label_order is not None and result.label_map is not None:
            label_scores = result.as_label_score_dict()
            ordered_values = [label_scores[label] for label in result.label_order if label in label_scores]
            total = len(ordered_values)
            for index, value in enumerate(ordered_values):
                weighted_mean += (total - 1 - index) * float(value)
            return weighted_mean

        for index, value in enumerate(scores.values()):
            weighted_mean += index * float(value)
        return weighted_mean

    def _get_samples_table(self, path: Path, context: TransformContext) -> tuple[np.ndarray, np.ndarray]:
        cache_key = f"normalized_score:samples:{path}"
        return context.get_cached_or_load(cache_key, lambda: self._load_samples_file(path))

    @staticmethod
    def _interp_percentile(value: float, x: np.ndarray, y: np.ndarray) -> float:
        value = float(np.clip(value, x[0], x[-1]))
        idx = np.searchsorted(x, value)
        if idx >= len(x) - 1:
            return float(y[-1])

        x0, y0 = x[idx], y[idx]
        x1, y1 = x[idx + 1], y[idx + 1]
        return float(y0) if x1 == x0 else float((value - x0) / (x1 - x0) * (y1 - y0) + y0)

    @staticmethod
    def _load_samples_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            arr = np.asarray(data["arr_0"], dtype=np.float32)
            x, y = np.asarray(arr[0], dtype=np.float32), np.asarray(arr[1], dtype=np.float32)

        order = np.argsort(x)
        x, y = x[order], y[order]
        x = np.concatenate(([0.0], x, [x[-1] + 1e-6])).astype(np.float32, copy=False)
        y = np.concatenate(([0.0], y, [1.0])).astype(np.float32, copy=False)
        return x, y


# endregion


def _clean_tag_text(tag: str) -> str:
    if tag in KAOMOJIS:
        return tag
    return tag.replace("_", " ")
