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
    result = MultiScoreResult(scores={"good": 0.7, "bad": 0.1}, score_min=0.0, score_max=1.0)

    assert result.to_dict() == {
        "output_type": "multi_score",
        "scores": {"good": 0.7, "bad": 0.1},
        "label_order": [],
        "score_min": 0.0,
        "score_max": 1.0,
        "normalized_score": None,
    }
