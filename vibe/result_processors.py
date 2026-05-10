"""Typed result processors applied at inference time."""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, Union, get_type_hints

import numpy as np

from vibe.backends.base import FileRole
from vibe.backends.char_ip_mapping import (
    apply_character_ip_mapping,
    resolve_character_ip_mapping,
)
from vibe.loader import FileMap
from vibe.plugins.shared.tagger_shared import load_tag_metadata
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


@dataclass
class ResultProcessorContext:
    file_map: FileMap
    source: str
    auto_download: bool


@dataclass(frozen=True)
class ParamInfo:
    name: str
    type: Any
    default: Any
    required: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": _type_to_string(self.type),
            "default": self.default,
            "required": self.required,
            "description": self.description,
        }


@dataclass(frozen=True)
class ProcessorInfo:
    processor_id: str
    display_name: str
    description: str
    params: list[ParamInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "processor_id": self.processor_id,
            "display_name": self.display_name,
            "description": self.description,
            "params": [param.to_dict() for param in self.params],
        }


class Param:
    """Declare a processor parameter with its description.

    Place as a class-level annotation on a ResultProcessor subclass.
    The type and default come from the matching __init__ parameter —
    this only carries the human-readable description for API/UI consumers.

    Example:
        class MyProcessor(ResultProcessor):
            threshold = Param("Controls sensitivity. Higher = stricter.")
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __init__(self, description: str) -> None:
        self.description = description

    def __repr__(self) -> str:
        return f"Param({self.description!r})"


class ResultProcessor(ABC, Generic[TIn, TOut]):
    """Base class for result processors."""

    _abstract = True
    display_name: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Skip validation on the abstract base itself and any intermediate marker subclasses.
        if cls is ResultProcessor or inspect.isabstract(cls):
            return

        # Lightweight helper processors are allowed to omit public metadata.
        if not cls.display_name or not cls.description:
            return

        # Collect Param declarations from this class (not inherited ones)
        param_decls: dict[str, Param] = {k: v for k, v in vars(cls).items() if isinstance(v, Param)}

        # Build ProcessorInfo by merging Param descriptions with __init__ signature
        sig = inspect.signature(cls.__init__)
        try:
            hints = get_type_hints(cls.__init__)
        except Exception:
            hints = {}

        params: list[ParamInfo] = []
        for name, p in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue
            # keyword-only marker
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue

            decl = param_decls.get(name)
            params.append(
                ParamInfo(
                    name=name,
                    type=hints.get(name),
                    default=p.default if p.default is not inspect.Parameter.empty else None,
                    required=p.default is inspect.Parameter.empty,
                    description=decl.description if decl is not None else "",
                )
            )

        cls._processor_info = ProcessorInfo(
            processor_id=cls.__name__,
            display_name=cls.display_name,
            description=cls.description,
            params=params,
        )

    @classmethod
    def describe(cls) -> ProcessorInfo:
        """Return metadata about this processor for API/UI consumers."""
        return cls._processor_info

    def on_infer_start(self, *, context: ResultProcessorContext) -> None:
        """Hook called once per infer call before any outputs are processed."""
        del context

    @abstractmethod
    def process(
        self,
        result: TIn,
        *,
        context: ResultProcessorContext,
    ) -> TOut:
        raise NotImplementedError


class CharacterIPMapping(ResultProcessor[TagResult, TagResult]):
    display_name = "Character IP Mapping"
    description = "Maps copyright tags to character tags from tag results."

    mapping_file = Param(
        "Path to a custom character-to-IP mapping JSON file. If omitted, the mapping bundled with the model is used."
    )

    def __init__(self, mapping_file: str | Path | None = None) -> None:
        self._mapping_file = str(mapping_file) if mapping_file is not None else None
        self._mapping_cache: dict[str, list[str]] | None = None

    def process(
        self,
        result: TagResult,
        *,
        context: ResultProcessorContext,
    ) -> TagResult:
        if not isinstance(result, TagResult):
            return result

        character_entries = result.category("character")
        if not character_entries:
            result.character_mapping = None
            return result

        mapping = self._get_mapping(context)
        if not mapping:
            result.character_mapping = None
            return result

        mapped = apply_character_ip_mapping([entry.tag for entry in character_entries], mapping)
        result.character_mapping = mapped or None
        return result

    def _get_mapping(self, context: ResultProcessorContext) -> dict[str, list[str]]:
        if self._mapping_cache is not None:
            return self._mapping_cache

        model_dir = self._resolve_model_dir(context.file_map)
        self._mapping_cache = resolve_character_ip_mapping(
            model_dir=model_dir,
            manual_path=self._mapping_file,
            allow_download=context.auto_download,
        )
        return self._mapping_cache

    def _resolve_model_dir(self, file_map: FileMap) -> Path:
        tag_list_path = _resolve_file_path(file_map, preferred_names=("selected_tags.csv",), roles=(FileRole.TAG_LIST,))
        if tag_list_path is not None:
            return tag_list_path.parent

        values = file_map.values()
        first_path = values[0] if values else None
        if first_path is not None:
            return first_path.parent

        return Path.cwd()


class CleanTags(ResultProcessor[TagResult, TagResult]):
    display_name = "Clean Tags"
    description = "Replaces underscores with spaces, preserving kaomojis."

    def process(
        self,
        result: TagResult,
        *,
        context: ResultProcessorContext,
    ) -> TagResult:
        del context
        if not isinstance(result, TagResult):
            return result

        for entries in result.categories().values():
            entries[:] = [TagEntry(tag=_clean_tag_text(entry.tag), score=entry.score) for entry in entries]

        if result.character_mapping is not None:
            result.character_mapping = {
                _clean_tag_text(character): [_clean_tag_text(ip) for ip in ips]
                for character, ips in result.character_mapping.items()
            }

        return result


class TagLevelThresholds(ResultProcessor[TagResult, TagResult]):
    display_name = "Tag Level Thresholds"
    description = (
        "Filters tags using per-tag thresholds from selected_tags.csv. "
    )

    threshold_column = Param(
        "Name of the CSV column containing per-tag threshold values. Defaults to 'best_threshold'."
    )
    threshold_offset = Param(
        "Fixed value added to every tag's threshold. "
        "Negative = more tags pass, positive = fewer. "
        "Cannot be combined with threshold_relaxation."
    )
    threshold_relaxation = Param(
        "Proportional reduction applied to each tag's threshold. "
        "E.g. 0.1 reduces a threshold of 0.80 to 0.72 and 0.20 to 0.18. "
        "Must be in [0.0, 1.0]. Cannot be combined with threshold_offset."
    )

    def __init__(
        self,
        *,
        threshold_column: str = "best_threshold",
        threshold_offset: float = 0.0,
        threshold_relaxation: float = 0.0,
    ) -> None:
        if threshold_offset != 0.0 and threshold_relaxation != 0.0:
            raise ValueError("Use only one of threshold_offset or threshold_relaxation.")
        if threshold_relaxation < 0.0 or threshold_relaxation >= 1.0:
            raise ValueError("threshold_relaxation must be between 0.0 and 1.0 (non-inclusive).")
        self._threshold_column = threshold_column
        self._threshold_offset = threshold_offset
        self._threshold_relaxation = threshold_relaxation
        self._threshold_cache: dict[str, dict[str, float]] = {}
        self._threshold_stats_cache: dict[str, tuple[int, int, bool]] = {}
        self._warned_this_call: set[str] = set()

    def on_infer_start(self, *, context: ResultProcessorContext) -> None:
        del context
        self._warned_this_call.clear()

    def process(
        self,
        result: TagResult,
        *,
        context: ResultProcessorContext,
    ) -> TagResult:
        if not isinstance(result, TagResult):
            return result

        csv_path = _resolve_file_path(
            context.file_map,
            preferred_names=("selected_tags.csv",),
            roles=(FileRole.TAG_LIST,),
            suffixes=(".csv",),
        )
        if csv_path is None:
            self._warn(
                key=f"missing-csv:{context.source}",
                message=(
                    "TagLevelThresholds could not find a tag-list CSV in model files; "
                    "cannot apply tag-level thresholds."
                ),
            )
            return result

        threshold_map = self._threshold_map_for_csv(csv_path)
        total_tags, with_threshold_count, threshold_column_present = self._threshold_stats_for_csv(csv_path)
        missing_count = max(total_tags - with_threshold_count, 0)
        missing_pct = (missing_count / total_tags * 100.0) if total_tags > 0 else 0.0

        if not threshold_column_present:
            self._warn(
                key=f"threshold-column-missing:{csv_path}",
                message=(
                    f"selected_tags.csv at '{csv_path}' has no '{self._threshold_column}' column; "
                    f"0/{total_tags} tags have threshold data and {missing_count}/{total_tags} "
                    f"tags ({missing_pct:.1f}%) are missing it. Tag-level-threshold filtering is skipped."
                ),
            )
            return result

        if not threshold_map:
            self._warn(
                key=f"threshold-values-missing:{csv_path}",
                message=(
                    f"selected_tags.csv at '{csv_path}' has a '{self._threshold_column}' column, "
                    f"but 0/{total_tags} tags have usable threshold data and {missing_count}/{total_tags} "
                    f"tags ({missing_pct:.1f}%) are missing it. Tag-level-threshold filtering is skipped."
                ),
            )
            return result

        if missing_count > 0:
            self._warn(
                key=f"threshold-values-partial:{csv_path}",
                message=(
                    f"selected_tags.csv at '{csv_path}' has partial '{self._threshold_column}' data: "
                    f"{with_threshold_count}/{total_tags} tags have thresholds, while {missing_count}/{total_tags} "
                    f"tags ({missing_pct:.1f}%) are missing it and will remain unfiltered."
                ),
            )

        for category, entries in result.categories().items():
            before = len(entries)
            filtered: list[TagEntry] = []
            for entry in entries:
                threshold = threshold_map.get(entry.tag)
                if threshold is not None:
                    threshold += self._threshold_offset
                    if self._threshold_relaxation != 0.0:
                        threshold *= 1.0 - self._threshold_relaxation
                if threshold is None or entry.score >= threshold:
                    filtered.append(entry)
            entries[:] = filtered
            logger.debug(
                "TagLevelThresholds applied category=%s kept=%d dropped=%d",
                category,
                len(filtered),
                before - len(filtered),
            )

        return result

    def _threshold_map_for_csv(self, csv_path: Path) -> dict[str, float]:
        cache_key = str(csv_path)
        cached = self._threshold_cache.get(cache_key)
        if cached is not None:
            return cached

        metadata = load_tag_metadata(csv_path, threshold_column=self._threshold_column)
        threshold_map: dict[str, float] = {}
        for tag_name, threshold in zip(metadata.raw_tag_names, metadata.per_tag_thresholds, strict=False):
            if threshold is None:
                continue
            threshold_map[tag_name] = threshold

        total_tags = len(metadata.raw_tag_names)
        with_threshold_count = len(threshold_map)
        self._threshold_stats_cache[cache_key] = (
            total_tags,
            with_threshold_count,
            metadata.threshold_column_present,
        )
        self._threshold_cache[cache_key] = threshold_map
        return threshold_map

    def _threshold_stats_for_csv(self, csv_path: Path) -> tuple[int, int, bool]:
        cache_key = str(csv_path)
        stats = self._threshold_stats_cache.get(cache_key)
        if stats is not None:
            return stats

        # Populate both caches when stats are first requested.
        self._threshold_map_for_csv(csv_path)
        stats = self._threshold_stats_cache.get(cache_key)
        if stats is not None:
            return stats
        return (0, 0, False)

    def _warn(self, *, key: str, message: str) -> None:
        namespaced = f"tag-level-thresholds:{key}"
        if namespaced in self._warned_this_call:
            return
        self._warned_this_call.add(namespaced)
        logger.warning(message)


class MultiScoreToScore(ResultProcessor[MultiScoreResult, ScoreResult]):
    display_name = "Multi-Score to Score"
    description = (
        "Collapses a MultiScoreResult into a single normalized ScoreResult. "
        "Useful for models that output multiple ranked scores (e.g. aesthetic models)."
    )

    use_samples_percentile = Param(
        "If True, uses a bundled samples file to map the score to a percentile "
        "rather than using simple min/max normalization. "
        "Requires a samples.npz file to be present in the model files."
    )
    label = Param("Label attached to the output ScoreResult. Defaults to 'score'.")

    def __init__(
        self,
        *,
        use_samples_percentile: bool = False,
        label: str = "score",
    ) -> None:
        self._label = label
        self._normalized_score = NormalizedScore(use_samples_percentile=use_samples_percentile)

    def process(
        self,
        result: MultiScoreResult,
        *,
        context: ResultProcessorContext,
    ) -> ScoreResult:
        if not isinstance(result, MultiScoreResult):
            raise TypeError(f"Expected MultiScoreResult, got {type(result)}")

        if not result.scores:
            logger.warning("MultiScoreToScore received empty scores; returning zero score.")
            return ScoreResult(score=0.0, score_min=0.0, score_max=1.0, label=self._label, normalized_score=0.0)

        normalized_result = self._normalized_score.process(result, context=context)
        score = (
            normalized_result.normalized_score
            if isinstance(normalized_result, MultiScoreResult) and normalized_result.normalized_score is not None
            else 0.0
        )
        return ScoreResult(
            score=score,
            score_min=0.0,
            score_max=1.0,
            label=self._label,
            normalized_score=score,
        )


class NormalizedScore(ResultProcessor[Union[ScoreResult, MultiScoreResult], Union[ScoreResult, MultiScoreResult]]):
    display_name = "Normalized Score"
    description = (
        "Attaches a normalized score in [0, 1] to a ScoreResult or MultiScoreResult. "
        "Supports both simple min/max normalization and percentile-based normalization "
        "via a bundled samples file."
    )

    use_samples_percentile = Param(
        "If True, uses a bundled samples.npz file to convert the raw score into a "
        "percentile rather than using min/max normalization. "
        "Requires a compatible samples.npz/csv file to be present in the model files."
    )

    def __init__(self, *, use_samples_percentile: bool = False) -> None:
        self._use_samples_percentile = use_samples_percentile
        self._samples_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._warned_paths: set[str] = set()

    def process(
        self,
        result: Union[ScoreResult, MultiScoreResult],
        *,
        context: ResultProcessorContext,
    ) -> Union[ScoreResult, MultiScoreResult]:
        if isinstance(result, ScoreResult):
            result.normalized_score = self._normalize_scalar(result.score, result.score_min, result.score_max)
            return result
        if not isinstance(result, MultiScoreResult):
            return result

        normalized = self._normalize_multiscore(result, context=context)
        result.normalized_score = normalized
        return result

    def _normalize_scalar(self, score: float, score_min: float, score_max: float) -> float:
        if score_max <= score_min:
            return 0.0
        return float(np.clip((score - score_min) / (score_max - score_min), 0.0, 1.0))

    def _normalize_multiscore(self, result: MultiScoreResult, *, context: ResultProcessorContext) -> float:
        if not result.scores:
            return 0.0

        weighted_mean = self._weighted_mean(result)
        samples_path = self._resolve_samples_path(context.file_map)

        if self._use_samples_percentile:
            if samples_path is None:
                raise FileNotFoundError(
                    "NormalizedScore: use_samples_percentile=True but no samples file was resolved."
                )
            x, y = self._get_samples_table(samples_path)
            return self._interp_percentile(weighted_mean, x, y)

        if samples_path is not None:
            key = str(samples_path)
            if key not in self._warned_paths:
                logger.warning(
                    f"Optional samples file '{samples_path}' detected but not used. "
                    "Set use_samples_percentile=True to apply percentile normalization."
                )
                self._warned_paths.add(key)

        max_v = float(max(len(result.scores) - 1, 1))
        if max_v <= 0.0:
            return 0.0
        return float(np.clip((weighted_mean - 0.0) / max_v, 0.0, 1.0))

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

    def _get_samples_table(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(path)
        cached = self._samples_cache.get(key)
        if cached is not None:
            return cached

        with np.load(path, allow_pickle=False) as data:
            if "arr_0" not in data:
                raise ValueError(f"Samples file '{path}' missing required arr_0 data.")

            arr = np.asarray(data["arr_0"], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 2:
                raise ValueError(f"Samples file '{path}' must contain a 2D array with at least two rows.")

            x = np.asarray(arr[0], dtype=np.float32)
            y = np.asarray(arr[1], dtype=np.float32)

        if x.size != y.size:
            raise ValueError(f"Samples file '{path}' score and percentile arrays must be the same length.")

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x = np.concatenate(([0.0], x, [x[-1] + 1e-6])).astype(np.float32, copy=False)
        y = np.concatenate(([0.0], y, [1.0])).astype(np.float32, copy=False)

        self._samples_cache[key] = (x, y)
        return x, y

    @staticmethod
    def _interp_percentile(value: float, x: np.ndarray, y: np.ndarray) -> float:
        value = float(np.clip(value, x[0], x[-1]))
        idx = np.searchsorted(x, value)

        if idx >= len(x) - 1:
            return float(y[-1])

        x0, y0 = x[idx], y[idx]
        x1, y1 = x[idx + 1], y[idx + 1]

        if x1 == x0:
            return float(y0)

        return float((value - x0) / (x1 - x0) * (y1 - y0) + y0)

    def _resolve_samples_path(self, file_map: FileMap) -> Path | None:
        return _resolve_file_path(
            file_map,
            preferred_names=("samples.npz", "samples.csv"),
            roles=(FileRole.MAPPING,),
            suffixes=(".npz", ".csv"),
        )


def _resolve_file_path(
    file_map: FileMap,
    *,
    preferred_names: tuple[str, ...] = (),
    roles: tuple[FileRole, ...] = (),
    suffixes: tuple[str, ...] = (),
) -> Path | None:
    resolved = file_map.as_path_dict()

    for name in preferred_names:
        path = resolved.get(name)
        if path is not None:
            return path

    if roles:
        role_names = {role.value for role in roles}
        for key, path in resolved.items():
            if key in role_names:
                return path

    if suffixes:
        normalized_suffixes = tuple(suffix.lower() for suffix in suffixes)
        for path in resolved.values():
            if path.suffix.lower() in normalized_suffixes:
                return path

    return None


def _type_to_string(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "__name__", str(value))


def _clean_tag_text(tag: str) -> str:
    if tag in KAOMOJIS:
        return tag
    return tag.replace("_", " ")
