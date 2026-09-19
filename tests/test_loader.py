from __future__ import annotations

from pathlib import Path

import pytest

from vibe.backends.base import ArtifactMap, ArtifactSpec, Backend, FileRole, ModelVariant
from vibe.exceptions import LoaderError
from vibe.loader import (
    _is_explicit_local_syntax,
    _is_local_source,
    parse_hf_source,
    resolve_variant_artifacts,
)


def test_parse_hf_source() -> None:
    repo, sub = parse_hf_source("SmilingWolf/wd-eva02-large-tagger-v3")
    assert repo == "SmilingWolf/wd-eva02-large-tagger-v3"
    assert sub is None

    repo, sub = parse_hf_source("hf:deepghs/anime_aesthetic/swinv2pv3_v0_448_ls0.2")
    assert repo == "deepghs/anime_aesthetic"
    assert sub == "swinv2pv3_v0_448_ls0.2"


def test_is_local_source_detection(tmp_path: Path) -> None:
    assert _is_local_source("local:/opt/models") is True
    assert _is_local_source("./models") is True
    assert _is_local_source("../models") is True
    assert _is_local_source(str(tmp_path)) is True

    assert _is_local_source("hf:SmilingWolf/wd-eva02-large-tagger-v3") is False
    assert _is_explicit_local_syntax("C:\\models") is True


def test_artifact_map_collection_semantics(tmp_path: Path) -> None:
    file_a = tmp_path / "model.onnx"
    file_b = tmp_path / "tags.csv"
    file_a.write_bytes(b"dummy_weights")
    file_b.write_text("tag,category\n", encoding="utf-8")

    spec_a = ArtifactSpec(id="model", name="model.onnx", role=FileRole.WEIGHTS)
    spec_b = ArtifactSpec(id="tags", name="tags.csv", role=FileRole.TAG_LIST)

    file_map = ArtifactMap(
        paths_by_id={"model": file_a, "tags": file_b},
        optional_missing={"config": "optional config not resolved"},
        specs_by_id={"model": spec_a, "tags": spec_b},
    )

    # Keyed access & mapping dunders
    assert file_map.get("model") == file_a
    assert file_map["tags"] == file_b
    assert file_map.get_optional("config") is None
    assert "model" in file_map
    assert len(file_map) == 2
    assert set(file_map.keys()) == {"model", "tags"}

    # Role lookups
    weights = file_map.by_role(FileRole.WEIGHTS)
    assert weights == {"model": file_a}

    # Missing required artifact raises KeyError
    with pytest.raises(KeyError, match="Artifact 'nonexistent' was not resolved"):
        file_map.get("nonexistent")


def test_resolve_local_variant_artifacts(tmp_path: Path) -> None:
    model_file = tmp_path / "model.safetensors"
    tags_file = tmp_path / "selected_tags.csv"
    model_file.write_bytes(b"weights")
    tags_file.write_text("name,category\n", encoding="utf-8")

    variant = ModelVariant(
        backend=Backend.PYTORCH,
        artifacts=(
            ArtifactSpec(id="model_pt", name="model.safetensors", role=FileRole.WEIGHTS),
            ArtifactSpec(id="tag_list", name="selected_tags.csv", role=FileRole.TAG_LIST),
            ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG, required=False),
        ),
    )

    file_map = resolve_variant_artifacts(
        source=f"local:{tmp_path}",
        variant=variant,
        allow_download=False,
    )

    assert file_map.get("model_pt") == model_file
    assert file_map.get("tag_list") == tags_file
    assert file_map.get_optional("config") is None
    assert "config" in file_map.optional_missing


def test_resolve_local_missing_required_file_raises_loader_error(tmp_path: Path) -> None:
    variant = ModelVariant(
        backend=Backend.ONNX,
        artifacts=(ArtifactSpec(id="model_onnx", name="model.onnx", role=FileRole.WEIGHTS, required=True),),
    )

    with pytest.raises(LoaderError, match="Missing in local folder"):
        resolve_variant_artifacts(
            source=f"local:{tmp_path}",
            variant=variant,
            allow_download=False,
        )
