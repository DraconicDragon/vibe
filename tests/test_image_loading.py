from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vibe.image_loading import iter_load_images


def test_iter_load_images_in_memory_refs() -> None:
    image_a = object()
    image_b = object()

    chunks = list(iter_load_images([(image_a, "a"), (image_b, "b")], batch_size=1, prefetch=False))

    assert len(chunks) == 2
    assert chunks[0].images[0] is image_a
    assert chunks[0].refs == ["a"]
    assert chunks[1].images[0] is image_b
    assert chunks[1].refs == ["b"]


def test_iter_load_images_path(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(path)

    chunks = list(iter_load_images([str(path)], batch_size=1, prefetch=False))

    assert len(chunks) == 1
    assert isinstance(chunks[0].images[0], Image.Image)


def test_iter_load_images_duplicate_refs() -> None:
    image = object()

    with pytest.raises(ValueError):
        list(iter_load_images([(image, "dup"), (image, "dup")], batch_size=1))
