from __future__ import annotations

from vibe.result_processors import ScoreThresholds
from vibe.results import TagEntry, TagResult


def _SimpleTagResult(
    *,
    general: list[TagEntry] | None = None,
    character: list[TagEntry] | None = None,
    artist: list[TagEntry] | None = None,
    rating: list[TagEntry] | None = None,
) -> TagResult:
    return TagResult(
        tags={
            "general": general or [],
            "character": character or [],
            "artist": artist or [],
            "rating": rating or [],
        }
    )


def test_score_thresholds_filters_by_global_threshold() -> None:
    processor = ScoreThresholds(threshold=0.50)

    result = _SimpleTagResult(
        general=[
            TagEntry(tag="keep", score=0.50),
            TagEntry(tag="drop", score=0.49),
        ],
        character=[TagEntry(tag="char_keep", score=0.80)],
        artist=[TagEntry(tag="artist_drop", score=0.10)],
        rating=[],
    )

    out = processor.process(result, context=None)  # type: ignore[arg-type]

    assert [entry.tag for entry in out.tags["general"]] == ["keep"]  # ty:ignore[unresolved-attribute]
    assert [entry.tag for entry in out.tags["character"]] == ["char_keep"]  # ty:ignore[unresolved-attribute]
    assert out.tags["artist"] == []  # ty:ignore[unresolved-attribute]
    assert out.tags["rating"] == []  # ty:ignore[unresolved-attribute]


def test_score_thresholds_applies_per_category_thresholds_over_global() -> None:
    processor = ScoreThresholds(
        threshold=0.50,
        category_thresholds={
            "general": 0.20,
            "artist": 0.90,
        },
    )

    result = _SimpleTagResult(
        general=[
            TagEntry(tag="general_keep", score=0.20),
            TagEntry(tag="general_drop", score=0.19),
        ],
        character=[
            TagEntry(tag="character_keep", score=0.50),
            TagEntry(tag="character_drop", score=0.49),
        ],
        artist=[
            TagEntry(tag="artist_drop", score=0.89),
            TagEntry(tag="artist_keep", score=0.90),
        ],
        rating=[],
    )

    out = processor.process(result, context=None)  # type: ignore[arg-type]

    assert [entry.tag for entry in out.tags["general"]] == ["general_keep"]  # ty:ignore[unresolved-attribute]
    assert [entry.tag for entry in out.tags["character"]] == ["character_keep"]  # ty:ignore[unresolved-attribute]
    assert [entry.tag for entry in out.tags["artist"]] == ["artist_keep"]  # ty:ignore[unresolved-attribute]
    assert out.tags["rating"] == []  # ty:ignore[unresolved-attribute]


def test_score_thresholds_rejects_out_of_range_thresholds() -> None:
    try:
        ScoreThresholds(threshold=-0.01)
    except ValueError as exc:
        assert "threshold must be between 0.0 and 1.0" in str(exc)
    else:
        raise AssertionError("Expected ScoreThresholds to reject an out-of-range global threshold.")


def test_score_thresholds_rejects_out_of_range_category_thresholds() -> None:
    try:
        ScoreThresholds(category_thresholds={"general": 1.01})
    except ValueError as exc:
        assert "Threshold for category 'general' must be between 0.0 and 1.0" in str(exc)
    else:
        raise AssertionError("Expected ScoreThresholds to reject an out-of-range category threshold.")
