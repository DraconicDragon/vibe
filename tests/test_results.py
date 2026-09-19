from __future__ import annotations

from vibe.metadata import OutputKind
from vibe.results import (
    InferenceResult,
    InferenceResultItem,
    MultiScoreResult,
    ScoreEntry,
    ScoreResult,
    TagEntry,
    TagResult,
    is_multi_score_result,
    is_score_result,
    is_tag_result,
)


def test_tag_result_aggregation_and_sorting() -> None:
    result = TagResult(
        categories={
            "general": [
                TagEntry(tag="solo", score=0.85),
                TagEntry(tag="1girl", score=0.95),
            ],
            "character": [
                TagEntry(tag="hatsune_miku", score=0.90),
            ],
        },
        extras={"execution_time": 0.042},
    )

    assert is_tag_result(result)
    assert not is_score_result(result)
    assert result.output_type == OutputKind.TAGS

    # Flattened tags must be sorted descending by score
    assert [e.tag for e in result.tags] == ["1girl", "hatsune_miku", "solo"]
    assert result.tag_names() == ["1girl", "hatsune_miku", "solo"]

    # Category accessor
    general_entries = result.category("general")
    assert [e.tag for e in general_entries] == ["solo", "1girl"]

    # Score dictionary accessors
    assert result.as_score_dict() == {
        "1girl": 0.95,
        "hatsune_miku": 0.90,
        "solo": 0.85,
    }


def test_tag_result_filtering() -> None:
    result = TagResult(
        categories={
            "general": [
                TagEntry(tag="1girl", score=0.95),
                TagEntry(tag="solo", score=0.40),
            ],
        }
    )

    filtered = result.filter(lambda entry, cat: entry.score >= 0.50)
    assert [e.tag for e in filtered.tags] == ["1girl"]
    assert "solo" not in filtered.as_score_dict()


def test_tag_result_to_dict_serialization() -> None:
    result = TagResult(
        categories={
            "general": [TagEntry(tag="1girl", score=0.95, extras={"id": 1})],
        },
        extras={"latency_ms": 12.5},
    )
    data = result.to_dict()

    assert data["output_type"] == "tags"
    assert "general" in data["categories"]
    assert data["categories"]["general"][0] == {"tag": "1girl", "score": 0.95, "extras": {"id": 1}}
    assert data["extras"] == {"latency_ms": 12.5}


def test_score_result_serialization() -> None:
    result = ScoreResult(
        score=8.5,
        score_min=0.0,
        score_max=10.0,
        normalized_score=0.85,
        label="aesthetic",
        extras={"percentile": 0.92},
    )

    assert is_score_result(result)
    assert result.output_type == OutputKind.SCORE

    data = result.to_dict()
    assert data["output_type"] == "score"
    assert data["score"] == 8.5
    assert data["normalized_score"] == 0.85
    assert data["extras"] == {"percentile": 0.92}


def test_multi_score_result_accessors_and_serialization() -> None:
    entries = [
        ScoreEntry(label="masterpiece", score=0.7, score_min=0.0, score_max=1.0, normalized_score=0.7),
        ScoreEntry(label="worst", score=0.1, score_min=0.0, score_max=1.0, normalized_score=0.1),
    ]
    result = MultiScoreResult(entries=entries, normalized_score=0.6)

    assert is_multi_score_result(result)
    assert result.entry("masterpiece") is entries[0]
    assert result.entry("nonexistent") is None
    assert result.as_score_dict() == {"masterpiece": 0.7, "worst": 0.1}

    data = result.to_dict()
    assert data["output_type"] == "multi_score"
    assert len(data["entries"]) == 2
    assert data["entries"][0]["label"] == "masterpiece"


def test_inference_result_batch_envelope() -> None:
    item1 = InferenceResultItem(
        index=0, result=ScoreResult(score=1.0, score_min=0, score_max=1, normalized_score=1.0), input_ref="img1.png"
    )
    item2 = InferenceResultItem(
        index=1, result=ScoreResult(score=0.5, score_min=0, score_max=1, normalized_score=0.5), input_ref="img2.png"
    )
    batch = InferenceResult(total_inputs=2, items=[item1, item2], memory={"rss_delta": 1024})

    assert len(batch) == 2
    assert batch.first() is item1.result
    assert [item.input_ref for item in batch] == ["img1.png", "img2.png"]

    data = batch.to_dict()
    assert data["total_inputs"] == 2
    assert len(data["items"]) == 2
    assert data["items"][0]["input_ref"] == "img1.png"
    assert data["memory"] == {"rss_delta": 1024}
