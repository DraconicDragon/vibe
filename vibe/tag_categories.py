"""Shared category enums/constants for tag and score outputs."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class DanbooruTagCategory(IntEnum):
    """Common Danbooru-style category IDs used by WD-family CSV files."""

    GENERAL = 0
    ARTIST = 1
    COPYRIGHT = 3
    CHARACTER = 4
    META = 5
    RATING = 9


class E621TagCategory(IntEnum):
    """Category IDs used by e621 tags."""

    GENERAL = 0
    ARTIST = 1
    CONTRIBUTOR = 2
    COPYRIGHT = 3
    CHARACTER = 4
    SPECIES = 5
    INVALID = 6
    META = 7
    LORE = 8


class AestheticBucketSKAR(StrEnum):
    """Named buckets used by Shio-Koube's aesthetic models.
        - ConvNext-aesthetic-rater
        - Anime-rater-2  
    """

    GOOD = "good"
    NORMAL = "normal"
    BAD = "bad"


DANBOORU_CATEGORY_LABELS: dict[int, str] = {
    DanbooruTagCategory.GENERAL: "general",
    DanbooruTagCategory.ARTIST: "artist",
    DanbooruTagCategory.COPYRIGHT: "copyright",
    DanbooruTagCategory.CHARACTER: "character",
    DanbooruTagCategory.META: "meta",
    DanbooruTagCategory.RATING: "rating",
}

E621_CATEGORY_LABELS: dict[int, str] = {
    E621TagCategory.GENERAL: "general",
    E621TagCategory.ARTIST: "artist",
    E621TagCategory.COPYRIGHT: "copyright",
    E621TagCategory.CHARACTER: "character",
    E621TagCategory.SPECIES: "species",
    E621TagCategory.INVALID: "invalid",
    E621TagCategory.META: "meta",
    E621TagCategory.LORE: "lore",
}
