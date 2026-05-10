from __future__ import annotations

from types import SimpleNamespace

import pytest

from vibe.result_processors import NormalizedScore, ResultProcessorContext
from vibe.results import MultiScoreResult, ScoreResult


def _context() -> ResultProcessorContext:
    return ResultProcessorContext(file_map=SimpleNamespace(as_path_dict=lambda: {}), source="test", auto_download=False)


def test_normalized_score_attaches_to_score_result() -> None:
    result = ScoreResult(score=3.0, score_min=1.0, score_max=5.0, label="aesthetic")

    processed = NormalizedScore().process(result, context=_context())

    assert isinstance(processed, ScoreResult)
    assert processed.normalized_score == pytest.approx(0.5)


def test_normalized_score_attaches_to_multi_score_result() -> None:
    result = MultiScoreResult(scores=[0.2, 0.8], label_map={0: "good", 1: "bad"})

    processed = NormalizedScore().process(result, context=_context())

    assert isinstance(processed, MultiScoreResult)
    assert processed.normalized_score == pytest.approx(0.8)


def test_processor_describe_is_json_friendly() -> None:
    info = NormalizedScore.describe()

    data = info.to_dict()

    assert data["processor_id"] == "NormalizedScore"
    assert data["display_name"] == "Normalized Score"
    assert data["params"][0]["name"] == "use_samples_percentile"
    assert data["params"][0]["type"] == "bool"
