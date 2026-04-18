"""Typed result processors applied at inference time."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibe.backends.char_ip_mapping import (
    apply_character_ip_mapping,
    resolve_character_ip_mapping,
)
from vibe.loader import FileMap
from vibe.plugins.shared.tagger_shared import load_tag_metadata
from vibe.results import ModelResult, TagEntry, TagResult

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
    model_id: str
    warning_keys: set[str]


class ResultProcessor:
    """Base class for result processors."""

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
        selected_tags = file_map.get("selected_tags.csv")
        if selected_tags is not None:
            return selected_tags.parent

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
    """Apply CSV `best_threshold` filtering to AnimeTimm tagger outputs."""

    def __init__(self, *, threshold_column: str = "best_threshold") -> None:
        self._threshold_column = threshold_column
        self._threshold_cache: dict[str, dict[str, float]] = {}

    def process(
        self,
        result: ModelResult,
        *,
        context: ResultProcessorContext,
    ) -> ModelResult:
        if not isinstance(result, TagResult):
            return result

        if not context.model_id.startswith("wdv4-"):
            self._warn_once(
                context,
                key=f"unsupported-model:{context.model_id}",
                message=(
                    "TagLevelThresholds was requested for model "
                    f"'{context.model_id}', but tag-level-thresholds are currently supported "
                    "for AnimeTimm WDV4 models only. Skipping filtering."
                ),
            )
            return result

        csv_path = context.file_map.get("selected_tags.csv")
        if csv_path is None:
            self._warn_once(
                context,
                key=f"missing-csv:{context.model_id}",
                message=(
                    f"Model '{context.model_id}' has no selected_tags.csv in file_map; "
                    "cannot apply tag-level-thresholds."
                ),
            )
            return result

        threshold_map = self._threshold_map_for_csv(csv_path, context)
        if not threshold_map:
            self._warn_once(
                context,
                key=f"no-thresholds:{csv_path}",
                message=(
                    f"No usable '{self._threshold_column}' values were found in {csv_path}; "
                    "tag-level-threshold filtering is skipped."
                ),
            )
            return result

        for category, entries in result.categories().items():
            before = len(entries)
            filtered: list[TagEntry] = []
            for entry in entries:
                threshold = threshold_map.get(entry.tag)
                if threshold is None or entry.score >= threshold:
                    filtered.append(entry)
            entries[:] = filtered
            logger.debug(
                "TagLevelThresholds applied model_id=%s category=%s kept=%d dropped=%d",
                context.model_id,
                category,
                len(filtered),
                before - len(filtered),
            )

        return result

    def _threshold_map_for_csv(
        self,
        csv_path: Path,
        context: ResultProcessorContext,
    ) -> dict[str, float]:
        cache_key = str(csv_path)
        cached = self._threshold_cache.get(cache_key)
        if cached is not None:
            return cached

        metadata = load_tag_metadata(csv_path, threshold_column=self._threshold_column)
        threshold_map: dict[str, float] = {}
        missing_count = 0
        for tag_name, threshold in zip(metadata.raw_tag_names, metadata.per_tag_thresholds, strict=False):
            if threshold is None:
                missing_count += 1
                continue
            threshold_map[tag_name] = threshold

        self._threshold_cache[cache_key] = threshold_map

        self._log_once(
            context,
            key=f"threshold-available:{cache_key}",
            level=logging.INFO,
            message=(
                f"Tag-level-threshold metadata loaded for model '{context.model_id}': "
                f"{len(threshold_map)} tags have '{self._threshold_column}' values."
            ),
        )

        if missing_count > 0:
            self._warn_once(
                context,
                key=f"threshold-missing:{cache_key}",
                message=(
                    f"Model '{context.model_id}' is missing '{self._threshold_column}' for "
                    f"{missing_count} tag(s); those tags are left unfiltered."
                ),
            )

        return threshold_map

    def _warn_once(self, context: ResultProcessorContext, *, key: str, message: str) -> None:
        self._log_once(context, key=key, level=logging.WARNING, message=message)

    def _log_once(
        self,
        context: ResultProcessorContext,
        *,
        key: str,
        level: int,
        message: str,
    ) -> None:
        namespaced = f"tag-level-thresholds:{key}"
        if namespaced in context.warning_keys:
            return
        context.warning_keys.add(namespaced)
        logger.log(level, message)


def _clean_tag_text(tag: str) -> str:
    if tag in KAOMOJIS:
        return tag
    return tag.replace("_", " ")
