from __future__ import annotations

from vibe.results import (
    InferenceResult,
    InferenceResultItem,
    MultiScoreResult,
    ScoreResult,
    TagEntry,
    TagResult,
)


def test_score_result_to_dict_serialization() -> None:
    result = ScoreResult(
        label="aesthetic",
        score=0.42,
        score_min=0.0,
        score_max=1.0,
        normalized_score=0.42,
    )

    serialized = result.to_dict()

    assert serialized == {
        "output_type": "score",
        "score": 0.42,
        "score_min": 0.0,
        "score_max": 1.0,
        "label": "aesthetic",
        "normalized_score": 0.42,
    }


def test_multi_score_result_to_dict_serialization() -> None:
    result = MultiScoreResult(
        scores=[0.7, 0.1],
        label_map={0: "good", 1: "bad"},
        label_order=["bad", "good"],
        score_min=0.0,
        score_max=1.0,
    )

    serialized = result.to_dict()

    assert serialized == {
        "output_type": "multi_score",
        "scores": {0: 0.7, 1: 0.1},
        "label_map": {0: "good", 1: "bad"},
        "label_order": ["bad", "good"],
        "score_min": 0.0,
        "score_max": 1.0,
        "normalized_score": None,
    }


def test_tag_result_to_dict_sorts_categories_by_score() -> None:
    # TagResult's to_dict has a custom feature: it converts TagEntry lists
    # to flat dictionaries and sorts tags in descending score order.
    result = TagResult(
        tags={
            "general": [
                TagEntry(tag="low_score", score=0.1),
                TagEntry(tag="high_score", score=0.9),
            ]
        },
        character_copyright_mapping={"char_a": ["copyright_x"]},
    )

    serialized = result.to_dict()

    assert serialized["output_type"] == "tags"
    # Verify descending sort order was enforced during serialization
    assert list(serialized["tags"]["general"].keys()) == ["high_score", "low_score"]
    assert serialized["tags"]["general"]["high_score"] == 0.9
    assert serialized["character_copyright_mapping"] == {"char_a": ["copyright_x"]}


def test_inference_batch_result_nested_serialization() -> None:
    score = ScoreResult(score=1.0, score_min=0.0, score_max=1.0)
    item = InferenceResultItem(index=0, result=score, input_ref="image_a.png")
    batch = InferenceResult(total_inputs=1, items=[item], memory={"rss_bytes": 1024})

    serialized = batch.to_dict()

    assert serialized["total_inputs"] == 1
    assert serialized["memory"] == {"rss_bytes": 1024}
    assert len(serialized["items"]) == 1

    serialized_item = serialized["items"][0]
    assert serialized_item["index"] == 0
    assert serialized_item["input_ref"] == "image_a.png"
    assert serialized_item["result"]["output_type"] == "score"
    assert serialized_item["result"]["score"] == 1.0
