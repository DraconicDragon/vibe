from __future__ import annotations

import vibe
from vibe.metadata import ModelDescriptor, OutputKind


def test_describe_returns_typed_model_descriptor() -> None:
    desc = vibe.describe("wd-eva02-large-v3")

    assert isinstance(desc, ModelDescriptor)
    assert desc.identity.model_id == "wd-eva02-large-v3"
    assert desc.family_name == "SmilingWolf WD Taggers"
    assert "tagger" in desc.model_types
    assert "LabelCatalogProvider" in desc.capabilities
    assert desc.output.kind == OutputKind.TAGS

    # Check variants
    assert len(desc.variants) >= 1
    artifact_ids = {art.id for v in desc.variants for art in v.artifacts}
    assert "tag_list" in artifact_ids


def test_describe_all_returns_valid_descriptors() -> None:
    descriptors = vibe.describe_all()

    assert descriptors
    assert all(isinstance(d, ModelDescriptor) for d in descriptors)

    model_ids = {d.identity.model_id for d in descriptors}
    assert "wd-eva02-large-v3" in model_ids
    assert "jtp-3" in model_ids
    assert "waifu-scorer-v3" in model_ids


def test_describe_to_dict_serialization() -> None:
    desc = vibe.describe("wd-eva02-large-v3")
    serialized = desc.to_dict()

    assert isinstance(serialized, dict)
    assert serialized["schema_version"] == "1.0.0"
    assert serialized["identity"]["model_id"] == "wd-eva02-large-v3"
    assert serialized["output"]["kind"] == "tags"
    assert "general" in serialized["output"]["categories"]
    assert isinstance(serialized["variants"], list)

    first_variant = serialized["variants"][0]
    assert "backend" in first_variant
    assert "artifacts" in first_variant
    assert any(a["id"] == "tag_list" for a in first_variant["artifacts"])
