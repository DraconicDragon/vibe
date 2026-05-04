from __future__ import annotations

import pytest

from vibe.results import MultiScoreResult, ScoreResult


def test_score_result_requires_label() -> None:
    with pytest.raises(TypeError):
        ScoreResult()  # ty:ignore[missing-argument]


def test_score_result_serializes_label() -> None:
    result = ScoreResult(label="aesthetic", score=0.42, score_min=0.0, score_max=1.0)

    assert result.to_dict() == {
        "output_type": "score",
        "score": 0.42,
        "score_min": 0.0,
        "score_max": 1.0,
        "label": "aesthetic",
        "normalized_score": None,
    }


def test_multi_score_result_uses_mapping_keys_as_labels() -> None:
    result = MultiScoreResult(
        scores=[0.7, 0.1],
        label_map={0: "good", 1: "bad"},
        label_order=["bad", "good"],
        score_min=0.0,
        score_max=1.0,
    )

    assert result.to_dict() == {
        "output_type": "multi_score",
        "scores": {0: 0.7, 1: 0.1},
        "label_map": {0: "good", 1: "bad"},
        "label_order": ["bad", "good"],
        "score_min": 0.0,
        "score_max": 1.0,
        "normalized_score": None,
    }


def test_multi_score_result_generates_default_label_metadata() -> None:
    result = MultiScoreResult(scores=[0.7, 0.1])

    assert result.label_map == {0: "score_1", 1: "score_2"}
    assert result.label_order is None
    assert result.to_dict() == {
        "output_type": "multi_score",
        "scores": {0: 0.7, 1: 0.1},
        "label_map": {0: "score_1", 1: "score_2"},
        "label_order": None,
        "score_min": None,
        "score_max": None,
        "normalized_score": None,
    }
