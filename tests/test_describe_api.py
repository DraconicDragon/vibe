from __future__ import annotations

import vibe
from vibe.backends.base import ModelPluginInfo


def test_describe_returns_typed_info_object() -> None:
    info = vibe.describe("wd-eva02-large-v3")

    assert isinstance(info, ModelPluginInfo)
    assert info.model_id == "wd-eva02-large-v3"
    assert any(spec.name == "selected_tags.csv" for spec in info.required_files)


def test_describe_supports_backward_compatible_dict_access() -> None:
    info = vibe.describe("wd-eva02-large-v3")

    assert info["model_id"] == info.model_id
    assert isinstance(info["required_files"], list)


def test_describe_all_returns_typed_info_list() -> None:
    infos = vibe.describe_all()

    assert infos
    assert all(isinstance(info, ModelPluginInfo) for info in infos)
