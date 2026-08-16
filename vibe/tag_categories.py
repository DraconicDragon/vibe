"""Shared category enums and mappings for tag and score outputs."""

from __future__ import annotations

from enum import Enum, IntEnum

# region Canonical Category Taxonomy


class TagCategory(str, Enum):
    """Canonical category names output by vibe tagger models."""

    GENERAL = "general"
    ARTIST = "artist"
    CONTRIBUTOR = "contributor"
    COPYRIGHT = "copyright"
    CHARACTER = "character"
    SPECIES = "species"
    META = "meta"
    INVALID = "invalid"
    RATING = "rating"
    LORE = "lore"

    def __str__(self) -> str:
        return self.value


# endregion


# region Dataset Specific Mappings


class DanbooruTagCategory(IntEnum):
    """Integer category IDs used in Danbooru / WD-family CSV tag metadata."""

    GENERAL = 0
    ARTIST = 1
    INVALID = 2
    COPYRIGHT = 3
    CHARACTER = 4
    META = 5
    RATING = 9


class E621TagCategory(IntEnum):
    """Integer category IDs used in e621 tag metadata."""

    # NOTE: e621 based models likely use META category for rating tags

    GENERAL = 0
    ARTIST = 1
    CONTRIBUTOR = 2
    COPYRIGHT = 3
    CHARACTER = 4
    SPECIES = 5
    INVALID = 6
    META = 7
    LORE = 8


# todo: implement model
# class AestheticBucketSKAR(StrEnum):
#     """Named buckets used by Shio-Koube's aesthetic models.
#         - ConvNext-aesthetic-rater
#         - Anime-rater-2
#     """

#     GOOD = "good"
#     NORMAL = "normal"
#     BAD = "bad"

# Mappings from dataset integer IDs to canonical TagCategory enums
DANBOORU_CATEGORY_LABELS: dict[int, TagCategory] = {
    DanbooruTagCategory.GENERAL: TagCategory.GENERAL,
    DanbooruTagCategory.ARTIST: TagCategory.ARTIST,
    DanbooruTagCategory.INVALID: TagCategory.INVALID,
    DanbooruTagCategory.COPYRIGHT: TagCategory.COPYRIGHT,
    DanbooruTagCategory.CHARACTER: TagCategory.CHARACTER,
    DanbooruTagCategory.META: TagCategory.META,
    DanbooruTagCategory.RATING: TagCategory.RATING,
}

E621_CATEGORY_LABELS: dict[int, TagCategory] = {
    E621TagCategory.GENERAL: TagCategory.GENERAL,
    E621TagCategory.ARTIST: TagCategory.ARTIST,
    E621TagCategory.CONTRIBUTOR: TagCategory.CONTRIBUTOR,
    E621TagCategory.COPYRIGHT: TagCategory.COPYRIGHT,
    E621TagCategory.CHARACTER: TagCategory.CHARACTER,
    E621TagCategory.SPECIES: TagCategory.SPECIES,
    E621TagCategory.INVALID: TagCategory.INVALID,
    E621TagCategory.META: TagCategory.META,
    E621TagCategory.LORE: TagCategory.LORE,
}

# endregion
