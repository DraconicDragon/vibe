from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vibe.image_loading import (
    ImageChunk,
    iter_load_images,
    normalize_input_format,
)


def _dummy_image(color: int = 128) -> Image.Image:
    arr = np.full((32, 32, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


# region Input Normalization


def test_normalize_single_image_like() -> None:
    img = _dummy_image()
    norm = normalize_input_format(img)

    assert norm.sized is True
    assert norm.total == 1
    assert norm.refs is None  # Auto-indexed
    assert norm.values == [img]


def test_normalize_single_image_with_custom_ref() -> None:
    img = _dummy_image()
    norm = normalize_input_format(img, refs=["custom_id"])

    assert norm.sized is True
    assert norm.total == 1
    assert norm.refs == ["custom_id"]


def test_normalize_sized_sequence_with_refs() -> None:
    images = [_dummy_image(1), _dummy_image(2)]
    refs = ["a", "b"]
    norm = normalize_input_format(images, refs=refs)

    assert norm.sized is True
    assert norm.total == 2
    assert norm.refs == ["a", "b"]


def test_normalize_sized_sequence_rejects_ref_length_mismatch() -> None:
    images = [_dummy_image(1), _dummy_image(2)]
    with pytest.raises(ValueError, match="Length mismatch"):
        normalize_input_format(images, refs=["only_one"])


def test_normalize_sized_sequence_rejects_duplicate_refs() -> None:
    images = [_dummy_image(1), _dummy_image(2)]
    with pytest.raises(ValueError, match="Explicit refs must be unique, found duplicate: 'a'"):
        normalize_input_format(images, refs=["a", "a"])


def test_normalize_mapping_preserves_keys_as_refs() -> None:
    images = {"ref1": _dummy_image(1), "ref2": _dummy_image(2)}
    norm = normalize_input_format(images)

    assert norm.sized is True
    assert norm.total == 2
    assert norm.refs == ["ref1", "ref2"]


def test_normalize_lazy_generator_is_unsized() -> None:
    def gen():
        yield _dummy_image(1)
        yield _dummy_image(2)

    norm = normalize_input_format(gen())
    assert norm.sized is False
    assert norm.total is None
    assert norm.refs is None


# endregion


# region Streaming & Batching


def test_iter_load_images_from_paths(tmp_path: Path) -> None:
    path1 = tmp_path / "img1.png"
    path2 = tmp_path / "img2.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(path1)
    Image.new("RGB", (16, 16), (0, 255, 0)).save(path2)

    chunks = list(iter_load_images([str(path1), str(path2)], batch_size=2, prefetch=False))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, ImageChunk)
    assert len(chunk.images) == 2
    assert isinstance(chunk.images[0], Image.Image)
    assert chunk.refs == [0, 1]


def test_streaming_generator_slices_batches_correctly() -> None:
    def gen():
        for i in range(5):
            yield _dummy_image(i)

    chunks = list(iter_load_images(gen(), batch_size=2, prefetch=False))

    assert len(chunks) == 3
    assert chunks[0].start_index == 0
    assert chunks[0].refs == [0, 1]
    assert chunks[1].start_index == 2
    assert chunks[1].refs == [2, 3]
    assert chunks[2].start_index == 4
    assert chunks[2].refs == [4]


def test_streaming_generator_cleans_up_on_early_break() -> None:
    generator_closed = False

    def tracked_gen() -> Generator[Image.Image, None, None]:
        nonlocal generator_closed
        try:
            yield _dummy_image(1)
            yield _dummy_image(2)
        finally:
            generator_closed = True

    for chunk in iter_load_images(tracked_gen(), batch_size=1, prefetch=False):
        assert len(chunk.images) == 1
        break

    assert generator_closed is True


def test_memory_bounded_streaming() -> None:
    TOTAL_ITEMS = 500
    BATCH_SIZE = 10

    def large_image_generator():
        for _ in range(TOTAL_ITEMS):
            yield _dummy_image(200)

    gc.collect()
    tracemalloc.start()

    processed_count = 0
    snapshot_before = tracemalloc.take_snapshot()

    for chunk in iter_load_images(large_image_generator(), batch_size=BATCH_SIZE, prefetch=False):
        processed_count += len(chunk.images)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    assert processed_count == TOTAL_ITEMS

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_growth_kb = sum(stat.size_diff for stat in top_stats) / 1024

    # Processing 500 items in small batches must stay tightly bounded (< 1MB growth)
    assert total_growth_kb < 1000, f"Memory growth ({total_growth_kb:.2f} KB) exceeded streaming threshold."


# endregion
