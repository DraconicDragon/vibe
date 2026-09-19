from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import vibe
from vibe.backends.base import ArtifactMap, ArtifactSpec, Backend, FileRole
from vibe.metadata import OutputKind
from vibe.plugins.dghs_aes import DGHSAesSwinV2Plugin
from vibe.plugins.pixai_v1.pixai_v1_plugin import PixAITaggerPlugin
from vibe.plugins.wd_tagger import WDEva02Plugin
from vibe.plugins.wdv4_animetimm import ATCaformerB36Plugin
from vibe.results import TagResult


def test_all_expected_models_registered() -> None:
    models = set(vibe.list_models())

    # Verify major families are registered
    assert "wd-eva02-large-v3" in models
    assert "at-caformer-b36-dbv4-full" in models
    assert "jtp-3" in models
    assert "pixai-tagger-v1.0" in models
    assert "waifu-scorer-v3" in models
    assert "dghs-aes-swinv2pv3-ls0.2" in models


def test_wd_tagger_load_ancillary_and_postprocess(tmp_path: Path) -> None:
    csv_file = tmp_path / "selected_tags.csv"
    csv_file.write_text(
        "name,category\n1girl,0\nsolo,0\nhatsune_miku,4\nsafe,9\n",
        encoding="utf-8",
    )

    plugin = WDEva02Plugin()
    artifacts = ArtifactMap(
        paths_by_id={"tag_list": csv_file},
        specs_by_id={"tag_list": ArtifactSpec(id="tag_list", name="selected_tags.csv", role=FileRole.TAG_LIST)},
    )
    plugin.load_ancillary(artifacts)

    assert plugin.catalog.names == ("1girl", "solo", "hatsune_miku", "safe")
    assert plugin._num_classes == 4

    # Postprocess logits -> sigmoid -> TagResult
    logits = np.array([[2.0, 1.0, 3.0, 5.0]], dtype=np.float32)
    plugin.bind_execution_state(type("Plan", (), {"backend": Backend.PYTORCH})())
    result = plugin.postprocess(logits)

    assert isinstance(result, TagResult)
    assert result.output_type == OutputKind.TAGS
    assert [e.tag for e in result.category("character")] == ["hatsune_miku"]
    assert [e.tag for e in result.category("rating")] == ["safe"]


def test_animetimm_load_ancillary_parses_thresholds(tmp_path: Path) -> None:
    csv_file = tmp_path / "selected_tags.csv"
    csv_file.write_text(
        "tag_id,name,category,best_threshold\n1,1girl,0,0.45\n2,safe,9,0.60\n",
        encoding="utf-8",
    )

    plugin = ATCaformerB36Plugin()
    artifacts = ArtifactMap(
        paths_by_id={"tag_list": csv_file},
        specs_by_id={"tag_list": ArtifactSpec(id="tag_list", name="selected_tags.csv", role=FileRole.TAG_LIST)},
    )
    plugin.load_ancillary(artifacts)

    assert plugin.catalog.names == ("1girl", "safe")
    assert plugin.thresholds.values == {"1girl": 0.45, "safe": 0.60}


def test_pixai_tagger_load_ancillary_splits(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_data = {
        "tags": ["1girl", "solo", "grea", "safe"],
        "tags_split": [["general", 2], ["character", 1], ["rating", 1]],
    }
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    plugin = PixAITaggerPlugin()
    artifacts = ArtifactMap(
        paths_by_id={"config": config_file},
        specs_by_id={"config": ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG)},
    )
    plugin.load_ancillary(artifacts)

    assert plugin.catalog.names == ("1girl", "solo", "grea", "safe")
    assert [info.name for info in plugin.catalog.by_category["character"]] == ["grea"]
    assert [info.name for info in plugin.catalog.by_category["general"]] == ["1girl", "solo"]


def test_dghs_aesthetic_descriptor_contracts() -> None:
    plugin = DGHSAesSwinV2Plugin()
    desc = plugin.describe()

    assert desc.output.kind == OutputKind.MULTI_SCORE
    assert "CalibrationProvider" in desc.capabilities
    assert "percentile" in desc.output.output_extras
