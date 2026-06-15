from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import vibe
from vibe.plugins.generic_timm import (
    GenericTimmMultiScorerPlugin,
    GenericTimmScorerPlugin,
    GenericTimmTaggerPlugin,
)
from vibe.results import MultiScoreResult, OutputType, ScoreResult, TagResult


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            line
            for line in [
                "{",
                '  "architecture": "resnet18",',
                '  "model_args": {"num_classes": 2},',
                '  "id2label": {"0": "cat", "1": "dog"},',
                '  "pretrained_cfg": {',
                '    "input_size": [3, 8, 8],',
                '    "mean": [0.5, 0.5, 0.5],',
                '    "std": [0.5, 0.5, 0.5],',
                '    "interpolation": "bicubic"',
                "  }",
                "}",
            ]
            if line
        ),
        encoding="utf-8",
    )


def test_generic_timm_is_registered_and_requires_source() -> None:
    assert "generic-timm-multi-score" in vibe.list_models()
    assert "generic-timm-score" in vibe.list_models()
    assert "generic-timm-tags" in vibe.list_models()

    info = vibe.describe("generic-timm-multi-score")
    assert info.default_hf_repo is None
    assert info.supported_backends == (vibe.Backend.ONNX, vibe.Backend.PYTORCH)
    assert vibe.describe("generic-timm-score").output_type == OutputType.SCORE
    assert vibe.describe("generic-timm-tags").output_type == OutputType.TAGS


def test_generic_timm_uses_manual_config_preprocess_for_onnx(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    plugin = GenericTimmMultiScorerPlugin()
    plugin.configure(backend=vibe.Backend.ONNX)
    caplog.set_level(logging.WARNING)
    plugin.load_ancillary({"config.json": config_path})
    assert "generic-timm model plugin is experimental" in caplog.text

    arr = plugin.preprocess(Image.new("RGB", (4, 4), (255, 0, 0)))
    assert arr.shape == (1, 3, 8, 8)
    assert arr.dtype == np.float32
    assert np.allclose(arr[0, :, 0, 0], np.array([1.0, -1.0, -1.0], dtype=np.float32))


def test_generic_timm_prefers_native_timm_preprocess_for_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    called = False

    def fake_prepare(self, config):
        nonlocal called
        called = True
        self._runtime_timm_transform = lambda image: np.zeros((3, 8, 8), dtype=np.float32)
        return True

    monkeypatch.setattr(GenericTimmMultiScorerPlugin, "_prepare_native_timm_transform", fake_prepare)
    plugin = GenericTimmMultiScorerPlugin()
    plugin.configure(backend=vibe.Backend.PYTORCH)
    plugin.load_ancillary({"config.json": config_path})

    assert called
    arr = plugin.preprocess(Image.new("RGB", (4, 4), (255, 0, 0)))
    assert arr.shape == (1, 3, 8, 8)
    assert np.all(arr == 0)


def test_generic_timm_postprocess_defaults_to_multi_score_with_config_labels(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    plugin = GenericTimmMultiScorerPlugin()
    plugin.load_ancillary({"config.json": config_path})

    result = plugin.postprocess(np.array([[0.25, 0.75]], dtype=np.float32))
    assert isinstance(result, MultiScoreResult)
    assert result.as_label_score_dict() == {"cat": 0.25, "dog": 0.75}


def test_generic_timm_postprocess_can_return_score(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    plugin = GenericTimmScorerPlugin()
    plugin.load_ancillary({"config.json": config_path})

    result = plugin.postprocess(np.array([[0.42]], dtype=np.float32))
    assert isinstance(result, ScoreResult)
    assert result.output_type == OutputType.SCORE
    assert result.score == pytest.approx(0.42)


def test_generic_timm_postprocess_can_return_tags(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    plugin = GenericTimmTaggerPlugin()
    plugin.load_ancillary({"config.json": config_path})

    result = plugin.postprocess(np.array([[0.25, 0.75]], dtype=np.float32))
    assert isinstance(result, TagResult)
    assert result.output_type == OutputType.TAGS
    assert [entry.tag for entry in result.category("tags")] == ["cat", "dog"]
