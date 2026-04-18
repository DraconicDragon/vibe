from __future__ import annotations

import logging
from pathlib import Path

from vibe.loader import FileMap
from vibe.result_processors import ResultProcessorContext, TagLevelThresholds
from vibe.results import TagEntry, TagResult


class _SimpleTagResult(TagResult):
    def __init__(
        self,
        *,
        general: list[TagEntry] | None = None,
        character: list[TagEntry] | None = None,
        artist: list[TagEntry] | None = None,
        rating: list[TagEntry] | None = None,
    ) -> None:
        super().__init__()
        self.general = general or []
        self.character = character or []
        self.artist = artist or []
        self.rating = rating or []

    def categories(self) -> dict[str, list[TagEntry]]:
        return {
            "general": self.general,
            "character": self.character,
            "artist": self.artist,
            "rating": self.rating,
        }


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

    assert [entry.tag for entry in out.general] == ["missing_threshold"]  # ty:ignore[unresolved-attribute]
    assert [entry.tag for entry in out.artist] == ["artist_name"]  # ty:ignore[unresolved-attribute]
    assert out.character == []  # ty:ignore[unresolved-attribute]
    assert [entry.tag for entry in out.rating] == ["safe"]  # ty:ignore[unresolved-attribute]


def test_tag_level_thresholds_warns_once_per_call_for_partial_thresholds(tmp_path: Path, caplog) -> None:
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
    assert len(partial_msgs) == 2
    assert "4/5 tags have thresholds" in partial_msgs[0]
    assert "1/5 tags (20.0%)" in partial_msgs[0]


def test_tag_level_thresholds_warns_when_threshold_column_is_missing(tmp_path: Path, caplog) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv_without_threshold_column(csv_path)

    processor = TagLevelThresholds()
    context = _make_context(csv_path=csv_path)
    result = _SimpleTagResult(general=[TagEntry(tag="1girl", score=0.10)])

    caplog.set_level(logging.WARNING)
    processor.on_infer_start(context=context)
    out = processor.process(result, context=context)

    assert [entry.tag for entry in out.general] == ["1girl"]  # ty:ignore[unresolved-attribute]
    messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("has no 'best_threshold' column" in message for message in messages)
    assert any("0/4 tags have threshold data" in message for message in messages)
