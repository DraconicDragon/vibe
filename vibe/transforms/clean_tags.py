"""Text sanitization transforms: replacing underscores while preserving kaomojis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel

from vibe.metadata import ModelDescriptor, OutputKind
from vibe.results import TagEntry, TagResult
from vibe.session import ModelSession
from vibe.transforms.base import ResultTransform

DEFAULT_KAOMOJIS: frozenset[str] = frozenset(
    {
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
)


@dataclass(frozen=True, slots=True)
class CleanTags(ResultTransform[TagResult]):
    """
    Replaces underscores with spaces across all tags while preserving kaomojis.
    Also cleans character-to-copyright mapping keys and values if present in extras.
    """

    id: ClassVar[str] = "clean_tags"
    display_name: ClassVar[str] = "Clean Tags"
    description: ClassVar[str] = "Replaces underscores with spaces while preserving kaomojis."
    config_model: ClassVar[type[BaseModel] | None] = None

    kaomojis: frozenset[str] = field(default=DEFAULT_KAOMOJIS)

    @classmethod
    def is_supported(cls, target: ModelDescriptor | ModelSession) -> bool:
        if isinstance(target, ModelSession):
            return target.is_tagger
        return target.output.kind == OutputKind.TAGS

    def clean_text(self, text: str) -> str:
        """Replace underscores with spaces unless the text is a protected kaomoji."""
        return text if text in self.kaomojis else text.replace("_", " ")

    def __call__(self, result: TagResult) -> TagResult:
        cleaned_categories: dict[str, list[TagEntry]] = {}

        for category, entries in result.categories.items():
            cleaned_categories[category] = [
                TagEntry(
                    tag=self.clean_text(entry.tag),
                    score=entry.score,
                    extras=dict(entry.extras),
                )
                for entry in entries
            ]

        extras = dict(result.extras)
        if "character_copyright_mapping" in extras and isinstance(extras["character_copyright_mapping"], dict):
            mapping = extras["character_copyright_mapping"]
            extras["character_copyright_mapping"] = {
                self.clean_text(character): [self.clean_text(ip) for ip in ips] for character, ips in mapping.items()
            }

        return TagResult(categories=cleaned_categories, extras=extras)