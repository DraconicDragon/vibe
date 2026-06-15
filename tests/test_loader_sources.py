from __future__ import annotations

from pathlib import Path

import pytest

from vibe.backends.base import Backend, FileRole, FileSpec
from vibe.hf_downloader import HFDownloadError
from vibe.loader import (
    FileMap,
    LoaderError,
    resolve_from_hf_repo,
    resolve_from_local_folder,
    resolve_from_source_string,
    resolve_from_sources,
)


def test_resolve_local_folder_reports_missing_required_file(tmp_path: Path) -> None:
    # Keep one unrelated file so the error shows useful context.
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")

    specs = (FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=(Backend.ONNX,)),)

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_local_folder(tmp_path, specs, Backend.ONNX)

    message = str(excinfo.value)
    assert "model.onnx" in message
    assert "Files present" in message


def test_resolve_local_folder_accepts_hf_style_subdir_for_required_files(tmp_path: Path) -> None:
    folder = tmp_path / "anime_aesthetic"
    subdir = folder / "swinv2pv3_v0_448_ls0.2"
    subdir.mkdir(parents=True)

    model = subdir / "model.onnx"
    meta = subdir / "meta.json"
    samples = subdir / "samples.npz"
    model.write_text("model", encoding="utf-8")
    meta.write_text('{"labels": ["a"]}', encoding="utf-8")
    samples.write_bytes(b"npz")

    specs = (
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            required=True,
            backends=(Backend.ONNX,),
            hf_subdir="swinv2pv3_v0_448_ls0.2",
        ),
        FileSpec(
            name="meta.json",
            role=FileRole.CONFIG,
            required=True,
            hf_subdir="swinv2pv3_v0_448_ls0.2",
        ),
        FileSpec(
            name="samples.npz",
            role=FileRole.MAPPING,
            required=True,
            hf_subdir="swinv2pv3_v0_448_ls0.2",
        ),
    )

    file_map = resolve_from_local_folder(folder, specs, Backend.ONNX)

    assert file_map["model.onnx"] == model
    assert file_map["meta.json"] == meta
    assert file_map["samples.npz"] == samples


def test_resolve_local_folder_prefers_root_over_hf_style_subdir(tmp_path: Path) -> None:
    folder = tmp_path / "anime_aesthetic"
    subdir = folder / "swinv2pv3_v0_448_ls0.2"
    subdir.mkdir(parents=True)

    root_model = folder / "model.onnx"
    subdir_model = subdir / "model.onnx"
    root_model.write_text("root", encoding="utf-8")
    subdir_model.write_text("subdir", encoding="utf-8")

    specs = (
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            required=True,
            backends=(Backend.ONNX,),
            hf_subdir="swinv2pv3_v0_448_ls0.2",
        ),
    )

    file_map = resolve_from_local_folder(folder, specs, Backend.ONNX)

    assert file_map["model.onnx"] == root_model


def test_resolve_local_folder_reports_subdir_when_root_missing(tmp_path: Path) -> None:
    folder = tmp_path / "anime_aesthetic"
    subdir = folder / "swinv2pv3_v0_448_ls0.2"
    subdir.mkdir(parents=True)

    subdir_model = subdir / "model.onnx"
    subdir_model.write_text("subdir", encoding="utf-8")

    specs = (
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            required=True,
            backends=(Backend.ONNX,),
            hf_subdir="swinv2pv3_v0_448_ls0.2",
        ),
    )

    file_map = resolve_from_local_folder(folder, specs, Backend.ONNX)

    assert file_map["model.onnx"] == subdir_model


def test_resolve_local_folder_missing_required_mentions_both_candidates(tmp_path: Path) -> None:
    folder = tmp_path / "anime_aesthetic"
    folder.mkdir()

    specs = (
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            required=True,
            backends=(Backend.ONNX,),
            hf_subdir="swinv2pv3_v0_448_ls0.2",
        ),
    )

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_local_folder(folder, specs, Backend.ONNX)

    message = str(excinfo.value)
    assert "model.onnx" in message
    assert "tried 'model.onnx' and HF-style subfolder 'swinv2pv3_v0_448_ls0.2/model.onnx'" in message


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

    specs = (
        FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=(Backend.ONNX,)),
        FileSpec("optional.json", role=FileRole.CONFIG, required=False, backends=(Backend.ONNX,)),
    )

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

    specs = (
        FileSpec("model.onnx", role=FileRole.WEIGHTS, backends=(Backend.ONNX,)),
        FileSpec("selected_tags.csv", role=FileRole.TAG_LIST, backends=(Backend.ONNX,)),
    )

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
    specs = (FileSpec("model.onnx", role=FileRole.WEIGHTS, backends=(Backend.ONNX,)),)

    captured: dict[str, str] = {}

    def _fake_resolve_from_hf_repo(
        repo_id: str,
        file_specs: tuple[FileSpec, ...],
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
    specs = (FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=(Backend.ONNX,)),)

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_source_string("local:owner/repo", specs, Backend.ONNX)

    message = str(excinfo.value)
    assert "Requested local source via 'local:'" in message
    assert "looks like a HuggingFace repo ID" in message


def test_source_string_hf_prefix_suggests_local_when_path_exists(tmp_path: Path) -> None:
    specs = (FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=(Backend.ONNX,)),)

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_source_string(f"hf:{tmp_path}", specs, Backend.ONNX, allow_download=False)

    message = str(excinfo.value)
    assert "Requested HF source via 'hf:'" in message
    assert "looks like a local folder" in message


def test_source_string_auto_existing_local_dir_reports_missing_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    specs = (FileSpec("model.onnx", role=FileRole.WEIGHTS, required=True, backends=(Backend.ONNX,)),)

    local_dir = tmp_path / "owner" / "repo"
    local_dir.mkdir(parents=True)
    del monkeypatch

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_source_string(str(local_dir), specs, Backend.ONNX, allow_download=False)

    message = str(excinfo.value)
    assert "Required file(s) ['model.onnx']" in message
    assert "not found in local folder" in message


def test_source_string_auto_local_dir_backfills_missing_required_from_fallback_hf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "local_model"
    local_dir.mkdir()
    tags = local_dir / "selected_tags.csv"
    tags.write_text("name,category\n", encoding="utf-8")

    resolved_model = tmp_path / "cached_model.safetensors"
    resolved_model.write_text("weights", encoding="utf-8")

    specs = (
        FileSpec("model.safetensors", role=FileRole.WEIGHTS, required=True, backends=(Backend.PYTORCH,)),
        FileSpec("selected_tags.csv", role=FileRole.TAG_LIST, backends=(Backend.PYTORCH,)),
    )

    def _fake_download_with_reason(
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
        required: bool = True,
    ) -> tuple[Path | None, str | None]:
        del revision, cache_dir, allow_download, required
        assert repo_id == "animetimm/caformer_b36.dbv4-full"
        if filename == "model.safetensors":
            return resolved_model, None
        raise AssertionError(f"Unexpected filename: {filename}")

    monkeypatch.setattr("vibe.loader.download_or_cached_with_reason", _fake_download_with_reason)

    file_map = resolve_from_source_string(
        str(local_dir),
        specs,
        Backend.PYTORCH,
        fallback_hf_repo_id="animetimm/caformer_b36.dbv4-full",
        allow_download=True,
    )

    local_materialized = local_dir / "model.safetensors"
    assert local_materialized.is_file()
    assert local_materialized.read_text(encoding="utf-8") == "weights"

    assert file_map["selected_tags.csv"] == tags
    assert file_map["model.safetensors"] == local_materialized


def test_resolve_from_sources_uses_repo_specific_source_map(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = (
        FileSpec("model.safetensors", role=FileRole.WEIGHTS, backends=(Backend.PYTORCH,), repo_id="repo/a"),
        FileSpec("config.json", role=FileRole.CONFIG, repo_id="repo/b"),
    )

    captured: list[tuple[str, list[str]]] = []

    def _fake_resolve_from_source_string(
        source: str,
        file_specs: list[FileSpec],
        backend: Backend,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
        file_name_map: dict[str, str] | None = None,
        fallback_hf_repo_id: str | None = None,
    ) -> FileMap:
        del backend, revision, cache_dir, allow_download, file_name_map, fallback_hf_repo_id
        captured.append((source, [spec.name for spec in file_specs]))
        return FileMap({})

    monkeypatch.setattr("vibe.loader.resolve_from_source_string", _fake_resolve_from_source_string)

    resolve_from_sources(
        "hf:repo/default",
        specs,
        Backend.PYTORCH,
        source_map={
            "repo/a": "local:/path/a",
            "repo/b": "hf:repo/other",
        },
        allow_download=False,
    )

    assert ("local:/path/a", ["model.safetensors"]) in captured
    assert ("hf:repo/other", ["config.json"]) in captured


def test_source_string_auto_local_dir_error_is_single_path_for_missing_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "local_model"
    local_dir.mkdir()
    (local_dir / "selected_tags.csv").write_text("name,category\n", encoding="utf-8")

    specs = (
        FileSpec("config.json", role=FileRole.CONFIG, required=True, backends=(Backend.PYTORCH,)),
        FileSpec("selected_tags.csv", role=FileRole.TAG_LIST, backends=(Backend.PYTORCH,)),
    )

    def _fake_download_with_reason(
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
        required: bool = True,
    ) -> tuple[Path | None, str | None]:
        del revision, cache_dir, allow_download, required
        assert repo_id == "animetimm/caformer_b36.dbv4-full"
        assert filename == "config.json"
        raise HFDownloadError("Auto-download disabled and 'config.json' is not in cache")

    monkeypatch.setattr("vibe.loader.download_or_cached_with_reason", _fake_download_with_reason)

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_source_string(
            str(local_dir),
            specs,
            Backend.PYTORCH,
            fallback_hf_repo_id="animetimm/caformer_b36.dbv4-full",
            allow_download=False,
        )

    message = str(excinfo.value)
    assert "Required file" in message
    assert "was not found in local folder" in message
    assert "could not be resolved from HuggingFace fallback" in message
    assert "Repo id must be in the form" not in message


def test_source_string_auto_local_dir_optional_reason_includes_hf_fallback_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_dir = tmp_path / "local_model"
    local_dir.mkdir()
    tags = local_dir / "selected_tags.csv"
    tags.write_text("name,category\n", encoding="utf-8")

    specs = (
        FileSpec("selected_tags.csv", role=FileRole.TAG_LIST, backends=(Backend.PYTORCH,)),
        FileSpec("config.json", role=FileRole.CONFIG, required=False, backends=(Backend.PYTORCH,)),
    )

    def _fake_download_with_reason(
        repo_id: str,
        filename: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        allow_download: bool | None = None,
        required: bool = True,
    ) -> tuple[Path | None, str | None]:
        del revision, cache_dir, allow_download, required
        assert repo_id == "animetimm/caformer_b36.dbv4-full"
        assert filename == "config.json"
        return (
            None,
            "Failed to access 'config.json' in HuggingFace repo 'animetimm/caformer_b36.dbv4-full' (HTTP 401): unauthorized or forbidden. "
            "Check that your HuggingFace token is configured and that you have access to the repo.",
        )

    monkeypatch.setattr("vibe.loader.download_or_cached_with_reason", _fake_download_with_reason)

    file_map = resolve_from_source_string(
        str(local_dir),
        specs,
        Backend.PYTORCH,
        fallback_hf_repo_id="animetimm/caformer_b36.dbv4-full",
        allow_download=True,
    )

    reason = file_map.optional_missing_reasons().get("config.json")
    assert reason is not None
    assert "not found in local folder" in reason
    assert "HF fallback 'animetimm/caformer_b36.dbv4-full'" in reason
    assert "HTTP 401" in reason


def test_resolve_local_folder_supports_file_name_map(tmp_path: Path) -> None:
    model = tmp_path / "wdeva02.onnx"
    tags = tmp_path / "my_tags.csv"
    model.write_text("x", encoding="utf-8")
    tags.write_text("name,category\n", encoding="utf-8")

    specs = (
        FileSpec("model.onnx", role=FileRole.WEIGHTS, backends=(Backend.ONNX,)),
        FileSpec("selected_tags.csv", role=FileRole.TAG_LIST, backends=(Backend.ONNX,)),
    )

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
    specs = (FileSpec("model.onnx", role=FileRole.WEIGHTS, backends=(Backend.ONNX,)),)

    with pytest.raises(LoaderError) as excinfo:
        resolve_from_local_folder(
            tmp_path,
            specs,
            Backend.ONNX,
            file_name_map={"unknown.bin": "custom.bin"},
        )

    assert "file_name_map contains unknown key 'unknown.bin'" in str(excinfo.value)
