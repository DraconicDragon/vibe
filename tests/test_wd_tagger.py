from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import autotagger
from autotagger.plugins.wd_tagger import WDEva02Plugin, wd_tagger_params


def _write_selected_tags_csv(path: Path) -> None:
    path.write_text(
        "name,category\n" "blue_hair,0\n" "cat_ears,0\n" "miku_hatsune,4\n" "safe,9\n" "^_^,4\n",
        encoding="utf-8",
    )


def test_registry_contains_non_caformer_wd_models() -> None:
    models = set(autotagger.list_models())
    assert "wd-eva02-large" in models
    assert "wd-swinv2" in models
    assert "wd-convnext" in models
    assert "wd-vit" in models


def test_list_plugin_classes_contains_wd_classes() -> None:
    classes = set(autotagger.list_plugin_classes())
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


def test_postprocess_applies_thresholds_and_sigmoid(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(csv_path)

    plugin = WDEva02Plugin()
    plugin.load_ancillary({"selected_tags.csv": csv_path})

    # Logits to force sigmoid branch.
    # blue_hair(general): 3.0 -> pass
    # cat_ears(general): -3.0 -> fail
    # miku_hatsune(character): 2.0 -> pass
    # safe(rating): ignored in TagResult.tags
    # ^_^(character): -2.0 -> fail
    raw = np.array([[3.0, -3.0, 2.0, 5.0, -2.0]], dtype=np.float32)
    result = plugin.postprocess(
        raw,
        {
            "general_threshold": 0.5,
            "character_threshold": 0.8,
            "return_all_scores": False,
            "return_character_mapping": False,
        },
    )

    names = result.tag_names()
    assert "blue_hair" in names
    assert "miku_hatsune" in names
    assert "cat_ears" not in names
    assert result.all_scores is None
    assert result.character_mapping is None


def test_wd_typed_params_helper_returns_only_specified_keys() -> None:
    params = wd_tagger_params(
        general_threshold=0.4,
        clean_tags=True,
    )

    assert params == {
        "general_threshold": 0.4,
        "clean_tags": True,
    }


def test_param_schema_unknown_key_suggests_closest_name() -> None:
    plugin = WDEva02Plugin()

    with pytest.raises(ValueError) as excinfo:
        plugin.param_schema.validate({"generl_threshold": 0.3})

    message = str(excinfo.value)
    assert "Unknown parameter(s)" in message
    assert "generl_threshold" in message
    assert "general_threshold" in message
