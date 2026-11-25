"""Safeguard unit tests for vibe-result-transforms."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from vibe.metadata import OutputKind, TagFilterRecommendation
from vibe.results import InferenceResult, InferenceResultItem, TagEntry, TagResult
from vibe.session import ModelSession
from vibe.transforms import (
    CharacterIPMapping,
    CleanTags,
    ScoreThresholds,
    TagLevelThresholds,
    TransformError,
    TransformPipeline,
    TransformRequirementError,
    list_transforms,
)

# region 1. Test Fixtures & Helpers


def make_sample_tag_result() -> TagResult:
    """Creates a deterministic TagResult covering multiple categories and edge cases."""
    return TagResult(
        categories={
            "general": [
                TagEntry(tag="1girl", score=0.95),
                TagEntry(tag="blue_hair", score=0.60),
                TagEntry(tag="smile", score=0.30),
                TagEntry(tag="^_^", score=0.85),  # Kaomoji safeguard
            ],
            "character": [
                TagEntry(tag="hatsune_miku", score=0.90),
                TagEntry(tag="rem_(re:zero)", score=0.40),
            ],
        },
        extras={},
    )


# endregion


# region 2. ScoreThresholds Tests


def test_score_thresholds_global_filter() -> None:
    result = make_sample_tag_result()
    transform = ScoreThresholds(threshold=0.50)
    filtered = transform(result)

    general_tags = {e.tag: e.score for e in filtered.categories.get("general", [])}
    character_tags = {e.tag: e.score for e in filtered.categories.get("character", [])}

    # Kept (>= 0.50)
    assert "1girl" in general_tags
    assert "blue_hair" in general_tags
    assert "^_^" in general_tags
    assert "hatsune_miku" in character_tags

    # Dropped (< 0.50)
    assert "smile" not in general_tags
    assert "rem_(re:zero)" not in character_tags


def test_score_thresholds_category_override() -> None:
    result = make_sample_tag_result()
    # Global fallback is 0.50, but character category requires 0.85
    transform = ScoreThresholds(threshold=0.50, category_thresholds={"character": 0.85})
    filtered = transform(result)

    character_tags = [e.tag for e in filtered.categories.get("character", [])]
    general_tags = [e.tag for e in filtered.categories.get("general", [])]

    assert character_tags == ["hatsune_miku"]  # 0.90 >= 0.85, rem dropped (0.40 < 0.85)
    assert "blue_hair" in general_tags  # 0.60 >= 0.50 global


def test_score_thresholds_invalid_values() -> None:
    with pytest.raises(TransformError):
        ScoreThresholds(threshold=1.5)  # > 1.0

    with pytest.raises(TransformError):
        ScoreThresholds(threshold=0.5, category_thresholds={"general": -0.1})  # < 0.0


def test_score_thresholds_from_session_with_recommendations() -> None:
    mock_session = MagicMock()
    mock_session.is_tagger = True
    mock_session.tagger.recommendation = TagFilterRecommendation(
        global_threshold=0.53,
        category_thresholds={"character": 0.75},
    )

    # fallback_threshold is required as safety floor, but model recommendations take precedence!
    tf = ScoreThresholds.from_session(mock_session, fallback_threshold=0.35)

    assert tf.threshold == 0.53
    assert tf.category_thresholds["character"] == 0.75


def test_score_thresholds_from_session_without_recommendations_uses_fallback() -> None:
    mock_session = MagicMock()
    mock_session.is_tagger = True
    mock_session.tagger.recommendation = None  # Model has no recommendations

    tf = ScoreThresholds.from_session(mock_session, fallback_threshold=0.40)

    # Safely uses the fallback
    assert tf.threshold == 0.40
    assert tf.category_thresholds == {}


def test_score_thresholds_from_session_missing_fallback_raises_type_error() -> None:
    mock_session = MagicMock()
    mock_session.is_tagger = True

    with pytest.raises(TypeError):
        # Missing required fallback_threshold argument!
        ScoreThresholds.from_session(mock_session)  # ty:ignore[missing-argument]


def test_score_thresholds_from_session_user_category_override() -> None:
    mock_session = MagicMock()
    mock_session.is_tagger = True
    mock_session.tagger.recommendation = TagFilterRecommendation(
        global_threshold=0.53,
        category_thresholds={"character": 0.75},
    )

    # User explicitly overrides character to 0.90
    tf = ScoreThresholds.from_session(
        mock_session,
        fallback_threshold=0.35,
        category_thresholds={"character": 0.90},
    )

    assert tf.threshold == 0.53
    assert tf.category_thresholds["character"] == 0.90


def test_score_thresholds_from_session_fails_if_not_tagger() -> None:
    mock_session = MagicMock()
    mock_session.is_tagger = False
    mock_session.model_id = "test-scorer"
    mock_session.metadata.output.kind = OutputKind.SCORE

    with pytest.raises(TransformRequirementError):
        ScoreThresholds.from_session(mock_session, fallback_threshold=0.35)


# endregion


# region 3. TagLevelThresholds Tests


def test_tag_level_thresholds_exact_and_fallback() -> None:
    result = make_sample_tag_result()
    # 1girl has high threshold (0.90), blue_hair has 0.70. Others fall back to 0.50
    threshold_map = {"1girl": 0.90, "blue_hair": 0.70}
    transform = TagLevelThresholds(threshold_map=threshold_map, fallback=0.50)
    filtered = transform(result)

    general_tags = {e.tag for e in filtered.categories.get("general", [])}

    assert "1girl" in general_tags  # 0.95 >= 0.90
    assert "blue_hair" not in general_tags  # 0.60 < 0.70
    assert "^_^" in general_tags  # 0.85 >= fallback 0.50
    assert "smile" not in general_tags  # 0.30 < fallback 0.50


def test_tag_level_thresholds_offset() -> None:
    result = make_sample_tag_result()
    # 1girl is 0.95. Base threshold 0.90 + offset 0.10 = 1.00 -> should drop
    threshold_map = {"1girl": 0.90}
    transform = TagLevelThresholds(threshold_map=threshold_map, offset=0.10, fallback=0.0)
    filtered = transform(result)

    general_tags = {e.tag for e in filtered.categories.get("general", [])}
    assert "1girl" not in general_tags


def test_tag_level_thresholds_relative_offset() -> None:
    result = make_sample_tag_result()
    # blue_hair is 0.60. Base threshold 0.50 * (1 + 0.30) = 0.65 -> should drop
    threshold_map = {"blue_hair": 0.50}
    transform = TagLevelThresholds(threshold_map=threshold_map, relative_offset=0.30, fallback=0.0)
    filtered = transform(result)

    general_tags = {e.tag for e in filtered.categories.get("general", [])}
    assert "blue_hair" not in general_tags


def test_tag_level_thresholds_invalid_bounds() -> None:
    with pytest.raises(TransformError):
        TagLevelThresholds(threshold_map={}, relative_offset=1.5)  # > 1.0

    with pytest.raises(TransformError):
        TagLevelThresholds(threshold_map={}, fallback=-0.1)  # < 0.0


def test_tag_level_thresholds_from_session_success() -> None:
    mock_session = MagicMock()
    mock_session.is_tagger = True
    mock_session.tagger.thresholds.values = {"1girl": 0.80}

    tf = TagLevelThresholds.from_session(mock_session, fallback=0.50)
    assert tf.threshold_map == {"1girl": 0.80}
    assert tf.fallback == 0.50


def test_tag_level_thresholds_from_session_fails_if_no_threshold_table() -> None:
    mock_session = MagicMock()
    mock_session.is_tagger = True
    mock_session.model_id = "wd-eva02"
    mock_session.tagger.thresholds = None  # Model provides no per-tag table

    with pytest.raises(TransformRequirementError, match="does not provide per-tag calibrated thresholds"):
        TagLevelThresholds.from_session(mock_session)


def test_tag_level_thresholds_from_session_fails_if_not_tagger() -> None:
    mock_session = MagicMock()
    mock_session.is_tagger = False
    mock_session.model_id = "test-scorer"
    mock_session.metadata.output.kind = OutputKind.SCORE

    with pytest.raises(TransformRequirementError):
        TagLevelThresholds.from_session(mock_session)


# endregion


# region 4. CleanTags Tests


def test_clean_tags_preserves_kaomoji_and_cleans_underscores() -> None:
    result = make_sample_tag_result()
    transform = CleanTags()
    cleaned = transform(result)

    general_tags = [e.tag for e in cleaned.categories.get("general", [])]
    character_tags = [e.tag for e in cleaned.categories.get("character", [])]

    assert "blue hair" in general_tags  # Underscore replaced
    assert "^_^" in general_tags  # Kaomoji preserved!
    assert "hatsune miku" in character_tags
    assert "rem (re:zero)" in character_tags


def test_clean_tags_sanitizes_character_ip_mapping_extras() -> None:
    result = make_sample_tag_result()
    result.extras["character_copyright_mapping"] = {
        "hatsune_miku": ["vocaloid_series"],
    }

    transform = CleanTags()
    cleaned = transform(result)

    cleaned_mapping = cleaned.extras["character_copyright_mapping"]
    assert "hatsune miku" in cleaned_mapping
    assert cleaned_mapping["hatsune miku"] == ["vocaloid series"]


# endregion


# region 5. CharacterIPMapping Tests


def test_character_ip_mapping_adds_extras() -> None:
    result = make_sample_tag_result()
    mapping_data = {"hatsune_miku": ["vocaloid"]}

    transform = CharacterIPMapping(mapping=mapping_data)
    mapped = transform(result)

    assert "character_copyright_mapping" in mapped.extras
    assert mapped.extras["character_copyright_mapping"] == {"hatsune_miku": ["vocaloid"]}


def test_character_ip_mapping_unmapped_character_is_safe() -> None:
    result = make_sample_tag_result()
    # Mapping does not match any character in the result
    transform = CharacterIPMapping(mapping={"unrelated_character": ["some_ip"]})
    mapped = transform(result)

    # Doesn't fail, doesn't pollute extras
    assert "character_copyright_mapping" not in mapped.extras


def test_character_ip_mapping_no_characters_in_result() -> None:
    # Result has general tags only
    result = TagResult(categories={"general": [TagEntry(tag="1girl", score=0.9)]})
    transform = CharacterIPMapping(mapping={"hatsune_miku": ["vocaloid"]})
    mapped = transform(result)

    assert "character_copyright_mapping" not in mapped.extras


# endregion


# region 6. TransformPipeline Tests


def test_pipeline_chains_single_result() -> None:
    result = make_sample_tag_result()
    pipeline = TransformPipeline(
        [
            ScoreThresholds(threshold=0.50),  # Drops smile & rem
            CleanTags(),  # blue_hair -> blue hair
        ]
    )

    final_result = pipeline(result)

    general_tags = [e.tag for e in final_result.categories.get("general", [])]
    assert "blue hair" in general_tags
    assert "smile" not in general_tags


def test_pipeline_handles_inference_result_batch() -> None:
    item1 = InferenceResultItem(index=0, input_ref="img1.jpg", result=make_sample_tag_result())
    item2 = InferenceResultItem(index=1, input_ref="img2.jpg", result=make_sample_tag_result())
    batch = InferenceResult(total_inputs=2, items=[item1, item2], memory={"peak_rss": 100})

    pipeline = TransformPipeline(
        [
            CleanTags(),
        ]
    )

    transformed_batch = pipeline(batch)

    assert len(transformed_batch) == 2
    assert transformed_batch.total_inputs == 2
    assert transformed_batch.memory == {"peak_rss": 100}

    # Verify all items in the batch were processed
    for item in transformed_batch.items:
        tags = [e.tag for e in item.result.categories.get("general", [])]  # ty:ignore[unresolved-attribute]
        assert "blue hair" in tags


# endregion


# region 7. Discovery & UI Schemas Tests


def test_list_transforms_returns_all_transforms() -> None:
    transforms = list_transforms()
    assert ScoreThresholds in transforms
    assert TagLevelThresholds in transforms
    assert CleanTags in transforms
    assert CharacterIPMapping in transforms


def test_transform_get_schema_produces_openapi_spec() -> None:
    schema = ScoreThresholds.get_schema()
    assert schema is not None
    assert "properties" in schema
    assert "threshold" in schema["properties"]
    assert schema["properties"]["threshold"]["type"] == "number"

    tag_level_schema = TagLevelThresholds.get_schema()
    assert tag_level_schema is not None
    assert "properties" in tag_level_schema
    assert "relative_offset" in tag_level_schema["properties"]


def test_transform_is_supported_checks() -> None:
    mock_session = MagicMock(spec=ModelSession)
    mock_session.is_tagger = True
    mock_session.tagger.thresholds = None

    # TagLevelThresholds requires thresholds table
    assert not TagLevelThresholds.is_supported(mock_session)

    # ScoreThresholds and CleanTags only need a tagger
    assert ScoreThresholds.is_supported(mock_session)
    assert CleanTags.is_supported(mock_session)


# endregion


def _assert_same_categories(left: TagResult, right: TagResult) -> None:
    assert left.categories.keys() == right.categories.keys()
    for category, left_entries in left.categories.items():
        right_entries = right.categories[category]
        assert len(left_entries) == len(right_entries)
        for left_entry, right_entry in zip(left_entries, right_entries, strict=True):
            assert left_entry.tag == right_entry.tag
            assert left_entry.score == right_entry.score or (
                math.isnan(left_entry.score) and math.isnan(right_entry.score)
            )


def test_array_backed_transform_fast_paths_match_materialized_results() -> None:
    names = tuple(f"tag_{index}" for index in range(12))
    scores = np.array([0.5, 0.9, 0.4, 0.2, 0.8, np.nan, 0.7, 0.3, 0.6, 0.1, 0.25, 0.95], dtype=np.float32)
    categories = {"rating": [0, 1, 2], "general": list(range(3, 12))}

    transforms = (
        ScoreThresholds(threshold=0.45, category_thresholds={"rating": 0.6}),
        TagLevelThresholds(
            threshold_map={"tag_0": 0.5, "tag_10": 0.25},
            offset=0.05,
            relative_offset=-0.1,
            fallback=None,
        ),
    )
    for transform in transforms:
        fast_result = transform(TagResult.from_arrays(tag_names=names, scores=scores, category_indices=categories))
        materialized_input = TagResult.from_arrays(tag_names=names, scores=scores, category_indices=categories)
        _ = materialized_input.categories
        _assert_same_categories(fast_result, transform(materialized_input))

    predicate = lambda tag, score, category: (category == "rating" and score >= 0.4) or score >= 0.8
    fast_filtered = TagResult.from_arrays(tag_names=names, scores=scores, category_indices=categories).filter(predicate)
    materialized_input = TagResult.from_arrays(tag_names=names, scores=scores, category_indices=categories)
    _ = materialized_input.categories
    _assert_same_categories(fast_filtered, materialized_input.filter(predicate))
