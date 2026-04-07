from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import vibe
from vibe.plugins.wd_tagger import WDEva02Plugin, WDTagResult
from vibe.result_processors import CharacterIPMapping, CleanTags


def _write_selected_tags_csv(path: Path) -> None:
    path.write_text(
        "name,category\nblue_hair,0\ncat_ears,0\nmiku_hatsune,4\nsafe,9\n^_^,4\n",
        encoding="utf-8",
    )


def test_registry_contains_non_caformer_wd_models() -> None:
    models = set(vibe.list_models())
    assert "wd-eva02-large-v3" in models
    assert "wd-swinv2-v3" in models
    assert "wd-convnext-v3" in models
    assert "wd-vit-v3" in models

    # list_models() returns canonical model IDs; aliases should still resolve.
    assert vibe.registry.get("wd-eva02-v3").model_id == "wd-eva02-large-v3"


def test_list_plugin_classes_contains_wd_classes() -> None:
    classes = set(vibe.list_plugin_classes())
    assert "WDEva02Plugin" in classes
    assert "WDSwinV2Plugin" in classes


def test_load_ancillary_parses_categories_and_normalized_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    plugin = WDEva02Plugin()
    plugin.load_ancillary({"selected_tags.csv": csv_path})

    assert plugin._raw_tag_names[0] == "blue_hair"
    assert plugin._raw_tag_names[4] == "^_^"
    assert plugin._general_indices == [0, 1]
    assert plugin._character_indices == [2, 4]
    assert plugin._rating_indices == [3]


def test_preprocess_outputs_nhwc_bgr_float32(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    plugin = WDEva02Plugin()
    plugin.load_ancillary({"selected_tags.csv": csv_path})
    plugin.IMAGE_SIZE = 4

    image = Image.new("RGB", (2, 2), (255, 0, 0))
    arr = plugin.preprocess(image)

    assert arr.shape == (1, 4, 4, 3)
    assert arr.dtype == np.float32
    # RGB red -> BGR means [0, 0, 255]
    assert np.allclose(arr[0, 0, 0], np.array([0.0, 0.0, 255.0], dtype=np.float32))


def test_postprocess_returns_full_category_lists_and_sigmoid(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    plugin = WDEva02Plugin()
    plugin.load_ancillary({"selected_tags.csv": csv_path})

    # Logits to force sigmoid branch.
    # blue_hair(general): 3.0
    # cat_ears(general): -3.0
    # miku_hatsune(character): 2.0
    # safe(rating): 5.0
    # ^_^(character): -2.0
    raw = np.array([[3.0, -3.0, 2.0, 5.0, -2.0]], dtype=np.float32)
    result = plugin.postprocess(raw)

    assert isinstance(result, WDTagResult)

    assert [entry.tag for entry in result.rating] == ["safe"]
    assert [entry.tag for entry in result.general] == ["blue_hair", "cat_ears"]
    assert [entry.tag for entry in result.character] == ["miku_hatsune", "^_^"]

    # Sorted descending by score in each category.
    assert result.general[0].score > result.general[1].score
    assert result.character[0].score > result.character[1].score
    assert result.character_mapping is None


def test_wd_plugin_declares_supported_processors_by_class() -> None:
    supported = WDEva02Plugin.supported_processors
    assert CleanTags in supported
    assert CharacterIPMapping in supported
