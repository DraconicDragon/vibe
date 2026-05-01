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
    result = MultiScoreResult(scores={"good": 0.2, "bad": 0.8}, label_order=["good", "bad"])

    processed = NormalizedScore().process(result, context=_context())

    assert isinstance(processed, MultiScoreResult)
    assert processed.normalized_score == pytest.approx(0.8)
