"""Typed result processors applied at inference time."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class ResultProcessorContext:
    file_map: FileMap
    source: str
    auto_download: bool


class ResultProcessor:
    """Base class for result processors."""

    def on_infer_start(self, *, context: ResultProcessorContext) -> None:
        """Hook called once per infer/infer_batches/infer_async call before processing outputs."""
        del context

    def process(
        self,
        result: ModelResult,
        *,
        context: ResultProcessorContext,
    ) -> ModelResult:
        del context
        return result


class CharacterIPMapping(ResultProcessor):
    """Attach character -> copyright/IP mappings to tag results."""

    def __init__(self, mapping_file: str | Path | None = None) -> None:
        self._mapping_file = str(mapping_file) if mapping_file is not None else None
        self._mapping_cache: dict[str, list[str]] | None = None

    def process(
        self,
        result: ModelResult,
        *,
        context: ResultProcessorContext,
    ) -> ModelResult:
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


class CleanTags(ResultProcessor):
    """Normalize underscore-delimited tags while preserving kaomojis."""

    def process(
        self,
        result: ModelResult,
        *,
        context: ResultProcessorContext,
    ) -> ModelResult:
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


class TagLevelThresholds(ResultProcessor):
    """Filter tags using per-tag thresholds from selected_tags.csv.

    Use `threshold_offset` for a fixed adjustment, or `threshold_relaxation` for
    a proportional adjustment that scales with each tag's own threshold.
    Example: with `threshold_relaxation=0.1`, a threshold of `0.80` becomes
    `0.72` and a threshold of `0.20` becomes `0.18`.
    """

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
        result: ModelResult,
        *,
        context: ResultProcessorContext,
    ) -> ModelResult:
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

    def _threshold_map_for_csv(
        self,
        csv_path: Path,
    ) -> dict[str, float]:
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


class MultiScoreToScore(ResultProcessor):
    """Convert MultiScoreResult into a single ScoreResult.

    - Computes a weighted mean over labels.
        - When `use_samples_percentile=False` (default), falls back to simple
            min/max scaling for normalization using the result's score bounds.
        - When `use_samples_percentile=True`, requires a resolved auxiliary
            samples file and uses it to normalize the weighted mean to a percentile.

    Args:
        use_samples_percentile:
            If True, use the auxiliary samples file for percentile normalization.
        label:
            Label for the resulting ScoreResult.
    """

    def __init__(
        self,
        *,
        use_samples_percentile: bool = False,
        label: str = "score",
    ) -> None:
        self._use_samples_percentile = use_samples_percentile
        self._label = label

        # cache: samples_path -> (x, y) mark table
        self._samples_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._warned_paths: set[str] = set()

    def process(
        self,
        result: ModelResult,
        *,
        context: ResultProcessorContext,
    ) -> ModelResult:
        if not isinstance(result, MultiScoreResult):
            return result

        if not result.scores:
            logger.warning("MultiScoreToScore received empty scores; returning zero score.")
            return ScoreResult(score=0.0, score_min=0.0, score_max=1.0, label=self._label)

        # weighted mean
        labels = result.label_order or list(result.scores.keys())
        values = result.scores

        weighted_mean = 0.0
        for i, label in enumerate(labels):
            v = values.get(label)
            if v is None:
                continue
            weighted_mean += i * float(v)

        # percentile normalization
        percentile: float | None = None

        samples_path = self._resolve_samples_path(context.file_map)

        if self._use_samples_percentile:
            if samples_path is None:
                raise FileNotFoundError(
                    "MultiScoreToScore: use_samples_percentile=True but no samples file was resolved."
                )
            x, y = self._get_samples_table(samples_path)
            percentile = self._interp_percentile(weighted_mean, x, y)

        else:
            if samples_path is not None:
                key = str(samples_path)
                if key not in self._warned_paths:
                    logger.warning(
                        f"Samples file '{samples_path}' detected but not used. "
                        f"Enable use_samples_percentile=True to apply percentile normalization."
                    )
                    self._warned_paths.add(key)

            min_v = 0.0
            max_v = float(max(len(labels) - 1, 1))

            if max_v > min_v:
                # at this point, without samples file, percentile here just means normalized score
                percentile = (weighted_mean - min_v) / (max_v - min_v)
            else:
                percentile = 0.0

        percentile = float(np.clip(percentile, 0.0, 1.0))

        return ScoreResult(
            score=percentile,
            score_min=0.0,
            score_max=1.0,
            label=self._label,
        )

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


def _clean_tag_text(tag: str) -> str:
    if tag in KAOMOJIS:
        return tag
    return tag.replace("_", " ")
