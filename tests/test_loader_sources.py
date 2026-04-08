from __future__ import annotations

from pathlib import Path

import pytest

from vibe.backends.base import Backend, FileRole, FileSpec
from vibe.loader import (
    FileMap,
    LoaderError,
    resolve_from_hf_repo,
    resolve_from_local_folder,
    resolve_from_source_string,
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
    monkeypatch.setattr("vibe.loader.download_or_cached", fake_download_or_cached)

    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=[Backend.ONNX]),
        FileSpec("optional.json", role=FileRole.CONFIG, required=False, backends=[Backend.ONNX]),
    ]

    file_map = resolve_from_hf_repo("owner/repo", specs, Backend.ONNX, allow_download=False)

    assert file_map["model.onnx"] == required
    assert file_map.get("optional.json") is None


def test_resolve_hf_repo_supports_file_name_map(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mapped_model = tmp_path / "renamed_model.onnx"
    mapped_tags = tmp_path / "renamed_tags.csv"
    mapped_model.write_text("x", encoding="utf-8")
    mapped_tags.write_text("name,category\n", encoding="utf-8")

    requested_filenames: list[str] = []

    def fake_download_or_cached(
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
        required: bool = True,
    ) -> Path | None:
        del repo_id, revision, cache_dir, allow_download, required
        requested_filenames.append(filename)
        if filename == "renamed_model.onnx":
            return mapped_model
        if filename == "renamed_tags.csv":
            return mapped_tags
        raise AssertionError(f"Unexpected filename: {filename}")

    monkeypatch.setattr("vibe.loader.download_or_cached", fake_download_or_cached)

    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, backends=[Backend.ONNX]),
        FileSpec("selected_tags.csv", role=FileRole.TAG_LIST),
    ]

    file_map = resolve_from_hf_repo(
        "owner/repo",
        specs,
        Backend.ONNX,
        file_name_map={
            "model.onnx": "renamed_model.onnx",
            "selected_tags.csv": "renamed_tags.csv",
        },
        allow_download=False,
    )

    assert requested_filenames == ["renamed_model.onnx", "renamed_tags.csv"]
    assert file_map["model.onnx"] == mapped_model
    assert file_map["selected_tags.csv"] == mapped_tags


def test_source_string_hf_prefix_supports_file_name_map(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, backends=[Backend.ONNX]),
    ]

    captured: dict[str, str] = {}

    def _fake_resolve_from_hf_repo(
        repo_id: str,
        file_specs: list[FileSpec],
        backend: Backend,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
        file_name_map: dict[str, str] | None = None,
    ) -> FileMap:
        del file_specs, backend, revision, cache_dir, allow_download
        captured["repo_id"] = repo_id
        captured["mapped"] = "" if file_name_map is None else file_name_map.get("model.onnx", "")
        return FileMap({})

    monkeypatch.setattr("vibe.loader.resolve_from_hf_repo", _fake_resolve_from_hf_repo)

    resolve_from_source_string(
        "hf:owner/repo",
        specs,
        Backend.ONNX,
        file_name_map={"model.onnx": "renamed_model.onnx"},
        allow_download=False,
    )

    assert captured["repo_id"] == "owner/repo"
    assert captured["mapped"] == "renamed_model.onnx"


def test_source_string_local_prefix_is_strict_and_suggests_hf() -> None:
    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=[Backend.ONNX]),
    ]

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_source_string("local:owner/repo", specs, Backend.ONNX)

    message = str(excinfo.value)
    assert "Requested local source via 'local:'" in message
    assert "looks like a HuggingFace repo ID" in message


def test_source_string_hf_prefix_suggests_local_when_path_exists(tmp_path: Path) -> None:
    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=[Backend.ONNX]),
    ]

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_source_string(f"hf:{tmp_path}", specs, Backend.ONNX, allow_download=False)

    message = str(excinfo.value)
    assert "Requested HF source via 'hf:'" in message
    assert "looks like a local folder" in message


def test_source_string_auto_tries_hf_after_local_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=[Backend.ONNX]),
    ]

    local_dir = tmp_path / "owner" / "repo"
    local_dir.mkdir(parents=True)

    def _always_missing(*args, **kwargs):
        del args, kwargs
        raise LoaderError("hf missing")

    monkeypatch.setattr("vibe.loader.resolve_from_hf_repo", _always_missing)

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_source_string(str(local_dir), specs, Backend.ONNX, allow_download=False)

    message = str(excinfo.value)
    assert "Auto mode tries local folder first" in message
    assert "Local attempt failed" in message
    assert "HF attempt failed" in message


def test_resolve_local_folder_supports_file_name_map(tmp_path: Path) -> None:
    model = tmp_path / "wdeva02.onnx"
    tags = tmp_path / "my_tags.csv"
    model.write_text("x", encoding="utf-8")
    tags.write_text("name,category\n", encoding="utf-8")

    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, backends=[Backend.ONNX]),
        FileSpec("selected_tags.csv", role=FileRole.TAG_LIST),
    ]

    file_map = resolve_from_local_folder(
        tmp_path,
        specs,
        Backend.ONNX,
        file_name_map={
            "model.onnx": "wdeva02.onnx",
            "selected_tags.csv": "my_tags.csv",
        },
    )

    assert file_map["model.onnx"] == model
    assert file_map["selected_tags.csv"] == tags


def test_resolve_local_folder_rejects_unknown_file_name_map_key(tmp_path: Path) -> None:
    specs = [
        FileSpec("model.onnx", role=FileRole.WEIGHTS, backends=[Backend.ONNX]),
    ]

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_local_folder(
            tmp_path,
            specs,
            Backend.ONNX,
            file_name_map={"unknown.bin": "custom.bin"},
        )

    assert "file_name_map contains unknown key 'unknown.bin'" in str(excinfo.value)
