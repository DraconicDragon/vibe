"""Typed result processors applied at inference time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vibe.backends.char_ip_mapping import (
    apply_character_ip_mapping,
    resolve_character_ip_mapping,
)
from vibe.loader import FileMap
from vibe.results import ModelResult, TagEntry, TagResult

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


def _clean_tag_text(tag: str) -> str:
    if tag in KAOMOJIS:
        return tag
    return tag.replace("_", " ")
