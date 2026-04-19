from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import vibe
from tests.common.animetimm_selected_repos import ANIMETIMM_V4_SELECTED_REPOS
from vibe.plugins.wdv4_animetimm import (
    AnimeTimmV4TagResult,
    WDV4CaformerB36FullPlugin,
)


def _write_selected_tags_csv(path: Path) -> None:
    path.write_text(
        "tag_id,name,category,best_threshold\n"
        "1,1girl,0,0.45\n"
        "2,artist_name,1,0.55\n"
        "3,char_name,4,0.75\n"
        "4,safe,9,0.60\n",
        encoding="utf-8",
    )


def _write_config_json(path: Path, *, image_size: int = 512) -> None:
    path.write_text(
        "\n".join(
            [
                "{",
                '  "architecture": "caformer_b36",',
                '  "model_args": {"drop_path_rate": 0.4, "drop_rate": 0.2, "act_layer": "gelu_tanh"},',
                '  "pretrained_cfg": {',
                f'    "test_input_size": [3, {image_size}, {image_size}],',
                '    "mean": [0.485, 0.456, 0.406],',
                '    "std": [0.229, 0.224, 0.225]',
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )


def _write_preprocess_json(path: Path, *, pad_size: int = 512, final_size: int = 384) -> None:
    path.write_text(
        "\n".join(
            [
                "{",
                '  "test": [',
                "    {",
                '      "background_color": "white",',
                '      "interpolation": "bilinear",',
                f'      "size": [{pad_size}, {pad_size}],',
                '      "type": "pad_to_size"',
                "    },",
                "    {",
                '      "interpolation": "bicubic",',
                f'      "size": {final_size},',
                '      "type": "resize"',
                "    },",
                "    {",
                f'      "size": [{final_size}, {final_size}],',
                '      "type": "center_crop"',
                "    },",
                "    {",
                '      "type": "maybe_to_tensor"',
                "    },",
                "    {",
                '      "mean": [0.485, 0.456, 0.406],',
                '      "std": [0.229, 0.224, 0.225],',
                '      "type": "normalize"',
                "    }",
                "  ]",
                "}",
            ]
        ),
        encoding="utf-8",
    )


def _repo_to_model_id(repo_id: str) -> str:
    suffix = repo_id.split("/", 1)[-1]
    return f"wdv4-{suffix.replace('_', '-').replace('.', '-')}"


@pytest.mark.parametrize("repo_id", ANIMETIMM_V4_SELECTED_REPOS)
def test_registry_contains_all_selected_wdv4_models(repo_id: str) -> None:
    models = set(vibe.list_models())

    expected_model_id = _repo_to_model_id(repo_id)
    assert expected_model_id in models

    alias = repo_id.split("/", 1)[-1]
    resolved = vibe.registry.get(alias)
    assert resolved.model_id == expected_model_id


def test_load_ancillary_parses_animetimm_categories_and_thresholds(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    plugin = WDV4CaformerB36FullPlugin()
    plugin.load_ancillary({"selected_tags.csv": csv_path})

    assert plugin._raw_tag_names == ["1girl", "artist_name", "char_name", "safe"]
    assert plugin._general_indices == [0]
    assert plugin._artist_indices == [1]
    assert plugin._character_indices == [2]
    assert plugin._rating_indices == [3]


def test_preprocess_outputs_nchw_rgb_normalized_float32(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    plugin = WDV4CaformerB36FullPlugin()
    plugin.load_ancillary({"selected_tags.csv": csv_path})
    plugin.IMAGE_SIZE = 4

    image = Image.new("RGB", (2, 2), (255, 0, 0))
    arr = plugin.preprocess(image)

    assert arr.shape == (1, 3, 4, 4)
    assert arr.dtype == np.float32

    # RGB red pixel after timm-style normalize:
    # R: (1 - 0.485) / 0.229
    # G: (0 - 0.456) / 0.224
    # B: (0 - 0.406) / 0.225
    expected = np.array([2.2489083, -2.0357144, -1.8044444], dtype=np.float32)
    assert np.allclose(arr[0, :, 0, 0], expected, atol=1e-5)


def test_postprocess_returns_raw_artist_and_category_entries(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    plugin = WDV4CaformerB36FullPlugin()
    plugin.load_ancillary({"selected_tags.csv": csv_path})

    # Already probabilities in [0, 1].
    # 1girl=0.50, artist=0.56, character=0.70, safe=0.80
    raw = np.array([[0.50, 0.56, 0.70, 0.80]], dtype=np.float32)
    result = plugin.postprocess(raw)

    assert isinstance(result, AnimeTimmV4TagResult)

    assert [entry.tag for entry in result.general] == ["1girl"]
    assert [entry.tag for entry in result.artist] == ["artist_name"]
    assert [entry.tag for entry in result.character] == ["char_name"]
    assert [entry.tag for entry in result.rating] == ["safe"]


def test_pytorch_preprocess_uses_config_image_size_when_available(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    config_path = tmp_path / "config.json"
    _write_selected_tags_csv(csv_path)
    _write_config_json(config_path, image_size=6)

    plugin = WDV4CaformerB36FullPlugin()
    plugin.configure(backend=vibe.Backend.PYTORCH)
    plugin.load_ancillary({"selected_tags.csv": csv_path, "config.json": config_path})

    image = Image.new("RGB", (3, 2), (255, 0, 0))
    arr = plugin.preprocess(image)

    assert arr.shape == (1, 3, 6, 6)
    assert plugin._runtime_image_size == 6


def test_pytorch_preprocess_falls_back_when_config_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    plugin = WDV4CaformerB36FullPlugin()
    plugin.configure(backend=vibe.Backend.PYTORCH)
    plugin.IMAGE_SIZE = 5
    plugin.load_ancillary({"selected_tags.csv": csv_path})

    image = Image.new("RGB", (3, 2), (255, 0, 0))
    arr = plugin.preprocess(image)

    assert arr.shape == (1, 3, 5, 5)
    assert plugin._runtime_image_size is None


def test_pytorch_preprocess_uses_preprocess_json_over_config(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    config_path = tmp_path / "config.json"
    preprocess_path = tmp_path / "preprocess.json"
    _write_selected_tags_csv(csv_path)
    _write_config_json(config_path, image_size=6)
    _write_preprocess_json(preprocess_path, pad_size=8, final_size=4)

    plugin = WDV4CaformerB36FullPlugin()
    plugin.configure(backend=vibe.Backend.PYTORCH)
    plugin.load_ancillary(
        {
            "selected_tags.csv": csv_path,
            "config.json": config_path,
            "preprocess.json": preprocess_path,
        }
    )

    image = Image.new("RGB", (6, 2), (255, 0, 0))
    arr = plugin.preprocess(image)

    assert arr.shape == (1, 3, 4, 4)
    assert plugin._runtime_preprocess_steps is not None
