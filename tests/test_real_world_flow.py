from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

import autotagger
from autotagger.plugins.wd_tagger import wd_tagger_params
from autotagger.results import TagResult


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@pytest.mark.skipif(
    os.getenv("AUTOTAGGER_REAL_WORLD_TEST", "0") != "1",
    reason="Set AUTOTAGGER_REAL_WORLD_TEST=1 to run this integration smoke test.",
)
def test_real_world_library_flow_with_typed_params() -> None:
    # This checks the same call shape downstream apps will use.
    model_source = os.getenv("AUTOTAGGER_REAL_WORLD_MODEL_SOURCE", "")
    image_paths = _env_list("AUTOTAGGER_REAL_WORLD_IMAGE_PATHS")

    assert model_source, "Set AUTOTAGGER_REAL_WORLD_MODEL_SOURCE"
    assert image_paths, "Set AUTOTAGGER_REAL_WORLD_IMAGE_PATHS as comma-separated paths"

    images = [Image.open(Path(p)).convert("RGB") for p in image_paths]

    session = autotagger.load(
        "wd-eva02-large",
        source=model_source,
        backend="onnx",
        auto_download=False,
    )

    params = wd_tagger_params(
        general_threshold=0.35,
        character_threshold=0.85,
        return_all_scores=False,
        return_character_mapping=True,
        clean_tags=True,
    )

    result = session.infer(images[0], params=params)
    assert isinstance(result, TagResult)
    assert isinstance(result.to_dict(), dict)

    batch_results = session.infer_many(images, params=params, batch_size=max(1, len(images)), batch_method="auto")
    assert len(batch_results) == len(images)
