from __future__ import annotations

from pathlib import Path

import pytest

from autotagger.backends.base import Backend, FileRole, FileSpec
from autotagger.loader import (
    FileMap,
    LoaderError,
    ModelSource,
    resolve_from_hf_repo,
    resolve_from_local_folder,
)


def test_file_map_values_and_to_dict(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    tags = tmp_path / "selected_tags.csv"
    model.write_text("x", encoding="utf-8")
    tags.write_text("name,category\n", encoding="utf-8")

    file_map = FileMap({"model.onnx": model, "selected_tags.csv": tags})

    values = file_map.values()
    assert model in values
    assert tags in values
    assert file_map.to_dict()["model.onnx"] == str(model)


def test_resolve_local_folder_reports_missing_required_file(tmp_path: Path) -> None:
    # Keep one unrelated file so the error shows useful context.
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")

    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=[Backend.ONNX]),
    ]

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_local_folder(tmp_path, specs, Backend.ONNX)

    message = str(excinfo.value)
    assert "model.onnx" in message
    assert "Files present" in message


def test_resolve_hf_repo_skips_missing_optional(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    required = tmp_path / "model.onnx"
    required.write_text("x", encoding="utf-8")

    def fake_download_or_cached(
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
        required: bool = True,
    ) -> Path | None:
        del repo_id, revision, cache_dir, allow_download
        if filename == "model.onnx":
            return required_path
        if filename == "optional.json" and not required:
            return None
        raise AssertionError("Unexpected filename")

    required_path = required
    monkeypatch.setattr("autotagger.loader.download_or_cached", fake_download_or_cached)

    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=[Backend.ONNX]),
        FileSpec("optional.json", role=FileRole.CONFIG, required=False, backends=[Backend.ONNX]),
    ]

    file_map = resolve_from_hf_repo("owner/repo", specs, Backend.ONNX, allow_download=False)

    assert file_map["model.onnx"] == required
    assert file_map.get("optional.json") is None


def test_model_source_unknown_kind_raises() -> None:
    source = ModelSource("unknown", "x")

    with pytest.raises(LoaderError) as excinfo:
        source.resolve([], Backend.ONNX)

    assert "Unknown source kind" in str(excinfo.value)
