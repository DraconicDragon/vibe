from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

import vibe
from vibe.result_processors import CharacterIPMapping, CleanTags
from vibe.results import TagResult


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@pytest.mark.skipif(
    os.getenv("VIBE_REAL_WORLD_TEST", "0") != "1",
    reason="Set VIBE_REAL_WORLD_TEST=1 to run this integration smoke test.",
)
def test_real_world_library_flow_with_processors() -> None:
    # This checks the same call shape downstream apps will use.
    model_source = os.getenv("VIBE_REAL_WORLD_MODEL_SOURCE", "")
    image_paths = _env_list("VIBE_REAL_WORLD_IMAGE_PATHS")

    assert model_source, "Set VIBE_REAL_WORLD_MODEL_SOURCE"
    assert image_paths, "Set VIBE_REAL_WORLD_IMAGE_PATHS as comma-separated paths"

    images = [Image.open(Path(p)).convert("RGB") for p in image_paths]

    session = vibe.load(
        "wd-eva02-large-v3",
        source=model_source,
        backend="onnx",
        auto_download=False,
    )

    result = session.infer(images[0], result_processors=[CharacterIPMapping(), CleanTags()])
    single = result.items[0].result
    assert isinstance(single, TagResult)
    assert isinstance(single.to_dict(), dict)

    batch_results = session.infer(
        images,
        result_processors=[CharacterIPMapping(), CleanTags()],
        batch_size=max(1, len(images)),
        batch_method="auto",
    )
    assert len(batch_results) == len(images)
