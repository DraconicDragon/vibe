from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vibe.loader import FileMap
from vibe.result_processors import ResultProcessorContext, TagLevelThresholds
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


def _write_selected_tags_csv(path: Path) -> None:
    path.write_text(
        "tag_id,name,category,best_threshold\n"
        "1,1girl,0,0.45\n"
        "2,artist_name,1,0.55\n"
        "3,char_name,4,0.75\n"
        "4,safe,9,0.60\n"
        "5,missing_threshold,0,\n",
        encoding="utf-8",
    )


def _write_selected_tags_csv_without_threshold_column(path: Path) -> None:
    path.write_text(
        "tag_id,name,category\n1,1girl,0\n2,artist_name,1\n3,char_name,4\n4,safe,9\n",
        encoding="utf-8",
    )


def _make_context(*, csv_path: Path) -> ResultProcessorContext:
    return ResultProcessorContext(
        file_map=FileMap({"selected_tags.csv": csv_path}),
        source="hf:animetimm/caformer_b36.dbv4-full",
        auto_download=False,
    )


def test_tag_level_thresholds_filters_tags_by_best_threshold(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    processor = TagLevelThresholds()
    context = _make_context(csv_path=csv_path)

    result = _SimpleTagResult(
        general=[TagEntry(tag="1girl", score=0.30), TagEntry(tag="missing_threshold", score=0.05)],
        artist=[TagEntry(tag="artist_name", score=0.70)],
        character=[TagEntry(tag="char_name", score=0.60)],
        rating=[TagEntry(tag="safe", score=0.90)],
    )

    processor.on_infer_start(context=context)
    out = processor.process(result, context=context)

    assert [entry.tag for entry in out.tags["general"]] == ["missing_threshold"]
    assert [entry.tag for entry in out.tags["artist"]] == ["artist_name"]
    assert out.tags["character"] == []
    assert [entry.tag for entry in out.tags["rating"]] == ["safe"]


def test_tag_level_thresholds_applies_threshold_offset(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    processor = TagLevelThresholds(threshold_offset=-0.01)
    context = _make_context(csv_path=csv_path)

    result = _SimpleTagResult(
        general=[TagEntry(tag="1girl", score=0.44)],
        artist=[TagEntry(tag="artist_name", score=0.54)],
        character=[TagEntry(tag="char_name", score=0.74)],
        rating=[TagEntry(tag="safe", score=0.59)],
    )

    processor.on_infer_start(context=context)
    out = processor.process(result, context=context)

    assert [entry.tag for entry in out.tags["general"]] == ["1girl"]
    assert [entry.tag for entry in out.tags["artist"]] == ["artist_name"]
    assert [entry.tag for entry in out.tags["character"]] == ["char_name"]
    assert [entry.tag for entry in out.tags["rating"]] == ["safe"]


def test_tag_level_thresholds_applies_threshold_relative_offset(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    processor = TagLevelThresholds(threshold_relative_offset=0.1)
    context = _make_context(csv_path=csv_path)

    result = _SimpleTagResult(
        general=[TagEntry(tag="1girl", score=0.50)],  # 0.45 -> 0.495, so keep
        artist=[TagEntry(tag="artist_name", score=0.60)],  # 0.55 -> 0.605, so drop
        character=[TagEntry(tag="char_name", score=0.83)],  # 0.75 -> 0.825, so keep
        rating=[TagEntry(tag="safe", score=0.65)],  # 0.60 -> 0.66, so drop
    )

    processor.on_infer_start(context=context)
    out = processor.process(result, context=context)

    assert [entry.tag for entry in out.tags["general"]] == ["1girl"]
    assert out.tags["artist"] == []
    assert [entry.tag for entry in out.tags["character"]] == ["char_name"]
    assert out.tags["rating"] == []


def test_tag_level_thresholds_rejects_offset_and_relative_offset_together() -> None:
    try:
        TagLevelThresholds(threshold_offset=-0.01, threshold_relative_offset=0.1)
    except ValueError as exc:
        assert "Use only one of threshold_offset or threshold_relative_offset" in str(exc)
    else:
        raise AssertionError("Expected TagLevelThresholds to reject conflicting threshold adjustments.")


def test_tag_level_thresholds_warns_once_per_context_for_partial_thresholds(tmp_path: Path, caplog) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    processor = TagLevelThresholds()
    context = _make_context(csv_path=csv_path)
    result = _SimpleTagResult(general=[TagEntry(tag="missing_threshold", score=0.99)])

    caplog.set_level(logging.WARNING)

    processor.on_infer_start(context=context)
    processor.process(result, context=context)
    processor.process(result, context=context)

    processor.on_infer_start(context=context)
    processor.process(result, context=context)

    warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    partial_msgs = [msg for msg in warning_messages if "has partial 'best_threshold' data" in msg]

    assert len(partial_msgs) == 1
    assert "4/5 tags have thresholds" in partial_msgs[0]
    assert "1/5 tags (20.0%)" in partial_msgs[0]


def test_tag_level_thresholds_raises_when_threshold_column_is_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv_without_threshold_column(csv_path)

    processor = TagLevelThresholds()
    context = _make_context(csv_path=csv_path)
    result = _SimpleTagResult(general=[TagEntry(tag="1girl", score=0.10)])

    processor.on_infer_start(context=context)

    with pytest.raises(RuntimeError, match=r"is missing the 'best_threshold' column"):
        processor.process(result, context=context)


def test_tag_level_thresholds_uses_fallback_for_missing_thresholds(tmp_path: Path, caplog) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    processor = TagLevelThresholds(threshold_fallback=0.1)
    context = _make_context(csv_path=csv_path)
    result = _SimpleTagResult(
        general=[TagEntry(tag="missing_threshold", score=0.05)],
    )

    caplog.set_level(logging.WARNING)
    processor.on_infer_start(context=context)
    out = processor.process(result, context=context)

    assert out.tags["general"] == []
    messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("will use fallback threshold 0.100" in message for message in messages)


def test_tag_level_thresholds_leaves_missing_thresholds_unfiltered_without_fallback(tmp_path: Path, caplog) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    processor = TagLevelThresholds()
    context = _make_context(csv_path=csv_path)
    result = _SimpleTagResult(
        general=[TagEntry(tag="missing_threshold", score=0.05)],
    )

    caplog.set_level(logging.WARNING)
    processor.on_infer_start(context=context)
    out = processor.process(result, context=context)

    assert [entry.tag for entry in out.tags["general"]] == ["missing_threshold"]
    messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("will remain unfiltered" in message for message in messages)
