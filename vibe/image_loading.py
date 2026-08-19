"""Shared image loading helpers for multi-model workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

CancelCheck = Callable[[], None]

try:
    import pillow_jxl  # noqa: F401

    _HAS_PILLOW_JXL = True
    logger.debug("pillow_jxl plugin loaded successfully.")
except ImportError as e:
    logger.debug("pillow_jxl not available: %s", e)
    _HAS_PILLOW_JXL = False

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HAS_PILLOW_HEIF = True
    logger.debug("pillow_heif plugin loaded successfully.")
except ImportError as e:
    logger.debug("pillow_heif not available: %s", e)
    _HAS_PILLOW_HEIF = False

logger.debug("Registered Pillow open formats: %s", sorted(Image.OPEN.keys()))


@dataclass(frozen=True)
class ImageChunk:
    """Loaded image batch with refs aligned to original inputs."""

    start_index: int
    images: list[Any]
    refs: list[Any]


def _is_image_like(x: Any) -> bool:
    """Check if an object looks like a single image (path, array, PIL image, tensor)."""
    if isinstance(x, (str, Path, bytes, bytearray)):
        return True
    if hasattr(x, "shape"):  # numpy arrays, torch tensors
        return True
    return bool(isinstance(x, Image.Image))


def normalize_input_format(
    images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
    *,
    error_cls: type[Exception] = ValueError,
) -> tuple[list[Any | str], list[Any]]:
    # 1. Single image types (paths, byte streams, PIL images, numpy arrays, tensors, non-iterables)
    if (
        isinstance(images, (str, Path, bytes, bytearray))
        or hasattr(images, "shape")
        or not isinstance(images, Iterable)
    ):
        entries = [images]

    # 2. Tuples: Distinguish single (image, ref) 2-tuple from a tuple container of items
    elif isinstance(images, tuple):
        # If it is a 2-tuple where only the first item is image-like and the second is non-image metadata
        # (e.g. integer ID, dict, UUID), treat it as a single (image, ref) pair.
        # Otherwise (both are image-like, or len != 2), treat the tuple as a batch container.
        if len(images) == 2 and _is_image_like(images[0]) and not _is_image_like(images[1]):
            entries = [images]
        else:
            entries = list(images)

    # 3. Other iterables (list, generator, iterator, set, deque, etc.)
    else:
        entries = list(images)

    if not entries:
        return [], []

    tuple_count = sum(1 for item in entries if isinstance(item, tuple))
    if 0 < tuple_count < len(entries):
        raise error_cls(
            "Mixed input formats are not supported. "
            f"Received {tuple_count} tuple item(s) and {len(entries) - tuple_count} bare item(s). "
            "Use either all bare images/paths or all (image_or_path, ref) tuples."
        )

    values: list[Any | str] = []
    refs: list[Any] = []
    if tuple_count == len(entries):
        for i, item in enumerate(entries):
            if not isinstance(item, tuple) or len(item) != 2:
                raise error_cls(f"Tuple input at index {i} must be exactly (image_or_path, ref).")
            value, ref = item
            values.append(value)
            refs.append(ref)
        duplicates = _find_duplicates(refs)
        if duplicates:
            duplicate_str = ", ".join(repr(value) for value in duplicates)
            raise error_cls(f"Explicit refs must be unique. Duplicate refs: {duplicate_str}")
    else:
        values = list(entries)
        refs = list(range(len(values)))

    return values, refs


def should_prefetch_image_loading(*, path_inputs: int) -> bool:
    # Prefetch only helps when there are multiple path-based inputs.
    return path_inputs > 1


def load_image_if_path(value: Any | str, index: int, error_cls: type[Exception] = ValueError) -> Any:
    if not isinstance(value, (str, Path)):
        return value

    from PIL import Image

    path = Path(value)

    try:
        with Image.open(path) as img:
            return img.copy()
    except Exception as exc:
        raise error_cls(f"Failed to load image at index {index} from path '{path}': {exc}") from exc


def iter_load_images(
    images: Any,
    *,
    batch_size: int = 1,
    prefetch_batch_limit: int = 8,
    prefetch: bool | None = None,
    cancel_check: CancelCheck | None = None,
    error_cls: type[Exception] = ValueError,
) -> Iterator[ImageChunk]:
    """Yield image batches loaded from the input list, efficiently flattening the loading pipeline."""
    values, refs = normalize_input_format(images, error_cls=error_cls)
    if not values:
        return

    path_inputs = sum(1 for value in values if isinstance(value, (str, Path)))
    use_prefetch = should_prefetch_image_loading(path_inputs=path_inputs) if prefetch is None else bool(prefetch)

    def _load_batch(start_idx: int, end_idx: int) -> list[Any]:
        batch = []
        for i in range(start_idx, end_idx):
            if cancel_check:
                cancel_check()
            batch.append(load_image_if_path(values[i], index=i, error_cls=error_cls))
        return batch

    total = len(values)

    if not use_prefetch:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield ImageChunk(start_index=start, images=_load_batch(start, end), refs=refs[start:end])
        return

    # Queue-based prefetch
    from collections import deque

    max_prefetch_batches = max(1, prefetch_batch_limit // batch_size)
    futures_queue: deque[tuple[int, int, Future[list[Any]]]] = deque()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vibe-image-loader") as executor:
        current_idx = 0
        while current_idx < total and len(futures_queue) < max_prefetch_batches:
            end_idx = min(current_idx + batch_size, total)
            futures_queue.append((current_idx, end_idx, executor.submit(_load_batch, current_idx, end_idx)))
            current_idx = end_idx

        while futures_queue:
            chunk_start, chunk_end, future = futures_queue.popleft()
            loaded_images = _await_loaded_chunk(future, cancel_check=cancel_check)

            if current_idx < total:
                end_idx = min(current_idx + batch_size, total)
                futures_queue.append((current_idx, end_idx, executor.submit(_load_batch, current_idx, end_idx)))
                current_idx = end_idx

            yield ImageChunk(start_index=chunk_start, images=loaded_images, refs=refs[chunk_start:chunk_end])


def _await_loaded_chunk(
    future: Future[list[Any]],
    *,
    cancel_check: CancelCheck | None = None,
) -> list[Any]:
    while True:
        if cancel_check is not None:
            cancel_check()
        try:
            return future.result(timeout=0.05)
        except FutureTimeoutError:
            continue


def _robust_eq(a: Any, b: Any) -> bool:
    """Safely compare two refs, falling back to identity if `==` doesn't yield a single truth value."""
    if a is b:
        return True
    try:
        eq = a == b
    except Exception:
        # Ref's __eq__ raised arbitrary/unexpected exception - treat as "no match", never crash.
        return False
    try:
        return bool(eq)
    except (ValueError, TypeError):
        # e.g. a multi-element numpy/torch array: bool() on it raises ValueError.
        return False


def _find_duplicates(values: list[Any]) -> list[Any]:
    seen_hashable: set[Any] = set()
    seen_unhashable: list[Any] = []
    duplicates: list[Any] = []
    duplicates_set: set[Any] = set()  # fast membership check for hashable dupes

    for value in values:
        try:
            is_seen = value in seen_hashable
        except TypeError:
            # Unhashable (e.g. numpy array), fall back to robust linear scan
            if any(_robust_eq(value, existing) for existing in seen_unhashable):
                if not any(_robust_eq(value, existing) for existing in duplicates):
                    duplicates.append(value)
            else:
                seen_unhashable.append(value)
            continue

        if is_seen:
            if value not in duplicates_set:
                duplicates.append(value)
                duplicates_set.add(value)
        else:
            seen_hashable.add(value)

    return duplicates
