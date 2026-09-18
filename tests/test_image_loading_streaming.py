"""
Tests for streaming image loading, memory bounds, ref alignment, and generator cleanup.
"""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Generator

import numpy as np
import pytest
from PIL import Image

from vibe.image_loading import (
    NormalizedInput,
    iter_load_images,
    normalize_input_format,
)


def _dummy_image(color: int = 128) -> Image.Image:
    """Create a small in-memory test image."""
    arr = np.full((32, 32, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


# region Unit Tests: normalize_input_format


def test_normalize_sized_sequence():
    images = [_dummy_image(1), _dummy_image(2), _dummy_image(3)]
    norm = normalize_input_format(images)
    assert norm.sized is True
    assert norm.total == 3
    assert norm.refs == [0, 1, 2]
    assert len(norm.values) == 3


def test_normalize_sized_sequence_with_refs():
    images = [_dummy_image(1), _dummy_image(2)]
    refs = ["a", "b"]
    norm = normalize_input_format(images, refs=refs)
    assert norm.sized is True
    assert norm.total == 2
    assert norm.refs == ["a", "b"]


def test_normalize_sized_sequence_ref_mismatch():
    images = [_dummy_image(1), _dummy_image(2)]
    with pytest.raises(ValueError, match="Length mismatch"):
        normalize_input_format(images, refs=["a"])


def test_normalize_sized_sequence_duplicate_refs():
    images = [_dummy_image(1), _dummy_image(2)]
    with pytest.raises(ValueError, match="duplicate"):
        normalize_input_format(images, refs=["a", "a"])


def test_normalize_mapping():
    images = {"k1": _dummy_image(1), "k2": _dummy_image(2)}
    norm = normalize_input_format(images)
    assert norm.sized is True
    assert norm.total == 2
    assert norm.refs == ["k1", "k2"]


def test_normalize_lazy_generator():
    def gen():
        yield _dummy_image(1)
        yield _dummy_image(2)

    norm = normalize_input_format(gen())
    assert norm.sized is False
    assert norm.total is None
    assert norm.refs is None
    assert not isinstance(norm.values, list)


def test_normalized_input_post_init_enforces_iterators():
    # If external user constructs NormalizedInput(sized=False) with raw list, post_init converts to iterator
    raw_list = [_dummy_image(1), _dummy_image(2)]
    norm = NormalizedInput(sized=False, values=raw_list, refs=None, total=None)
    assert not isinstance(norm.values, list)
    assert hasattr(norm.values, "__next__")


# endregion Unit Tests: normalize_input_format


# region Streaming & Batching Tests


def test_streaming_generator_default_refs():
    def gen():
        for i in range(7):
            yield _dummy_image(i)

    chunks = list(iter_load_images(gen(), batch_size=3))
    assert len(chunks) == 3

    # Batch 1: indices 0..2
    assert chunks[0].start_index == 0
    assert chunks[0].refs == [0, 1, 2]
    assert len(chunks[0].images) == 3

    # Batch 2: indices 3..5
    assert chunks[1].start_index == 3
    assert chunks[1].refs == [3, 4, 5]
    assert len(chunks[1].images) == 3

    # Batch 3: index 6
    assert chunks[2].start_index == 6
    assert chunks[2].refs == [6]
    assert len(chunks[2].images) == 1


def test_streaming_generator_with_matching_refs():
    def img_gen():
        yield _dummy_image(1)
        yield _dummy_image(2)

    def ref_gen():
        yield "custom_1"
        yield "custom_2"

    chunks = list(iter_load_images(img_gen(), refs=ref_gen(), batch_size=1))
    assert len(chunks) == 2
    assert chunks[0].refs == ["custom_1"]
    assert chunks[1].refs == ["custom_2"]


def test_streaming_generator_refs_too_short():
    def img_gen():
        yield _dummy_image(1)
        yield _dummy_image(2)

    def ref_gen():
        yield "only_one"

    with pytest.raises(ValueError, match="exhausted before images stream ended"):
        list(iter_load_images(img_gen(), refs=ref_gen(), batch_size=1))


def test_streaming_generator_refs_too_long():
    def img_gen():
        yield _dummy_image(1)

    def ref_gen():
        yield "ref1"
        yield "excess_ref"

    with pytest.raises(ValueError, match="contains more items than the images stream"):
        list(iter_load_images(img_gen(), refs=ref_gen(), batch_size=1))


def test_streaming_generator_mid_stream_duplicate_ref():
    def img_gen():
        yield _dummy_image(1)
        yield _dummy_image(2)
        yield _dummy_image(3)

    refs = ["ref_a", "ref_b", "ref_a"]  # Duplicate at index 2

    gen = iter_load_images(img_gen(), refs=refs, batch_size=1)

    # First item succeeds
    chunk1 = next(gen)
    assert chunk1.refs == ["ref_a"]

    # Second item succeeds
    chunk2 = next(gen)
    assert chunk2.refs == ["ref_b"]

    # Third item encounters duplicate ref and raises
    with pytest.raises(ValueError, match="Duplicate ref encountered in stream"):
        next(gen)


def test_streaming_generator_cleanup_on_early_break():
    generator_closed = False

    def tracked_gen() -> Generator[Image.Image, None, None]:
        nonlocal generator_closed
        try:
            yield _dummy_image(1)
            yield _dummy_image(2)
            yield _dummy_image(3)
        finally:
            generator_closed = True

    # Consume only 1 chunk and break early
    for chunk in iter_load_images(tracked_gen(), batch_size=1):
        assert len(chunk.images) == 1
        break

    assert generator_closed is True, "Source generator was not closed on early exit."


def test_streaming_generator_cleanup_on_exception():
    generator_closed = False

    def tracked_gen() -> Generator[Image.Image, None, None]:
        nonlocal generator_closed
        try:
            yield _dummy_image(1)
            yield _dummy_image(2)
        finally:
            generator_closed = True

    with pytest.raises(RuntimeError, match="Simulated consumer crash"):
        for chunk in iter_load_images(tracked_gen(), batch_size=1):
            raise RuntimeError("Simulated consumer crash")

    assert generator_closed is True, "Source generator was not closed after exception."


def test_streaming_prefetch_queue_operation():
    """Verify prefetch mode works with streaming generator inputs."""

    def gen():
        for i in range(10):
            yield _dummy_image(i)

    chunks = list(iter_load_images(gen(), batch_size=2, prefetch=True, prefetch_batch_limit=4))
    assert len(chunks) == 5
    assert sum(len(c.images) for c in chunks) == 10


# endregion Streaming & Batching Tests


# region Memory Bound Smoke Test


def test_memory_bounded_streaming():
    """
    Verify that processing a large stream in small batches maintains bounded memory.
    """
    TOTAL_ITEMS = 3000
    BATCH_SIZE = 16

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

    # Streaming 3000 images with batch_size=16 keeps peak resident memory bounded (< 1.5MB growth)
    assert total_growth_kb < 1500, f"Memory growth ({total_growth_kb:.2f} KB) exceeded streaming threshold."


# endregion Memory Bound Smoke Test
