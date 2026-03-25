"""Result processor pipeline for session-level postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autotagger.backends.char_ip_mapping import (
    apply_character_ip_mapping,
    resolve_character_ip_mapping,
)
from autotagger.loader import FileMap, ModelSource
from autotagger.results import InferenceResult, TagEntry, TagResult

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


# region Context


@dataclass
class ResultProcessorContext:
    file_map: FileMap
    source: ModelSource
    auto_download: bool
    character_mapping_path: str | None


class ResultProcessor:
    def process(
        self,
        result: InferenceResult,
        *,
        params: dict[str, Any],
        context: ResultProcessorContext,
    ) -> InferenceResult:
        return result


# endregion Context


# region Processors


class CharacterIPMappingProcessor(ResultProcessor):
    """Attach character/IP mappings to tag results when requested."""

    def __init__(self) -> None:
        self._mapping_cache: dict[str, list[str]] | None = None

    def process(
        self,
        result: InferenceResult,
        *,
        params: dict[str, Any],
        context: ResultProcessorContext,
    ) -> InferenceResult:
        if not isinstance(result, TagResult):
            return result
        if not params.get("return_character_mapping", False):
            return result

        mapping = self._get_mapping(context)
        if not mapping:
            result.character_mapping = None
            return result

        mapped = apply_character_ip_mapping(result.tag_names(), mapping)
        result.character_mapping = mapped or None
        return result

    def _get_mapping(self, context: ResultProcessorContext) -> dict[str, list[str]]:
        if self._mapping_cache is not None:
            return self._mapping_cache

        model_dir = self._resolve_model_dir(context.file_map)
        self._mapping_cache = resolve_character_ip_mapping(
            model_dir=model_dir,
            manual_path=context.character_mapping_path,
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


class TagCleaningProcessor(ResultProcessor):
    """Optionally clean tag text while preserving kaomojis."""

    def process(
        self,
        result: InferenceResult,
        *,
        params: dict[str, Any],
        context: ResultProcessorContext,
    ) -> InferenceResult:
        del context
        if not isinstance(result, TagResult):
            return result
        if not params.get("clean_tags", False):
            return result

        result.tags = [TagEntry(tag=_clean_tag_text(entry.tag), score=entry.score) for entry in result.tags]

        if result.all_scores is not None:
            result.all_scores = [
                TagEntry(tag=_clean_tag_text(entry.tag), score=entry.score) for entry in result.all_scores
            ]

        if result.character_mapping is not None:
            result.character_mapping = {
                _clean_tag_text(character): [_clean_tag_text(ip) for ip in ips]
                for character, ips in result.character_mapping.items()
            }

        return result


def _clean_tag_text(tag: str) -> str:
    if tag in KAOMOJIS:
        return tag
    return tag.replace("_", " ")


# endregion Processors
