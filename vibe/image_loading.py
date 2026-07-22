"""Shared image loading helpers for multi-model workflows."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

CancelCheck = Callable[[], None]


try:
    import pillow_jxl  # noqa: F401

    _HAS_PILLOW_JXL = True
except ImportError:
    _HAS_PILLOW_JXL = False

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()

    _HAS_PILLOW_HEIF = True
except ImportError:
    _HAS_PILLOW_HEIF = False


@dataclass(frozen=True)
class ImageChunk:
    """Loaded image batch with refs aligned to original inputs."""

    start_index: int
    images: list[Any]
    refs: list[Any]


def normalize_input_format(
    images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
    *,
    error_cls: type[Exception] = ValueError,
) -> tuple[list[Any | str], list[Any]]:
    entries = images if isinstance(images, list) else [images]
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
    images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
    *,
    batch_size: int = 1,
    prefetch: bool | None = None,
    cancel_check: CancelCheck | None = None,
    error_cls: type[Exception] = ValueError,
) -> Iterator[ImageChunk]:
    """Yield image batches loaded from the input list, efficiently flattening the loading pipeline."""

    values, refs = normalize_input_format(images, error_cls=error_cls)
    if not values:
        return

    path_inputs = sum(1 for value in values if isinstance(value, (str, Path)))
    use_prefetch = (path_inputs > 1) if prefetch is None else bool(prefetch)

    def _load_batch(start_idx: int, end_idx: int) -> list[Any]:
        batch = []
        for i in range(start_idx, end_idx):
            if cancel_check:
                cancel_check()
            batch.append(load_image_if_path(values[i], index=i, error_cls=error_cls))
        return batch

    total = len(values)
    prefetch_read_size = max(batch_size, 8) if use_prefetch else batch_size

    if not use_prefetch:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield ImageChunk(start_index=start, images=_load_batch(start, end), refs=refs[start:end])
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vibe-image-loader") as executor:
        start = 0
        end = min(prefetch_read_size, total)
        future = executor.submit(_load_batch, start, end)

        while start < total:
            if cancel_check:
                cancel_check()

            # Await current chunk
            while True:
                try:
                    loaded_images = future.result(timeout=0.05)
                    break
                except FutureTimeoutError:
                    if cancel_check:
                        cancel_check()

            # Dispatch next chunk while yielding the current one
            next_start = start + batch_size
            next_end = min(next_start + prefetch_read_size, total)

            if next_start < total:
                future = executor.submit(_load_batch, next_start, next_end)

            yield ImageChunk(start_index=start, images=loaded_images, refs=refs[start:end])
            start = next_start


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


def _find_duplicates(values: list[Any]) -> list[Any]:
    seen_hashable: set[Any] = set()
    seen_unhashable: list[Any] = []
    duplicates: list[Any] = []
    duplicates_set: set[Any] = set()  # fast membership check for hashable dupes

    for value in values:
        try:
            is_seen = value in seen_hashable
        except TypeError:
            # Unhashable, fall back to linear scan
            if any(value == existing for existing in seen_unhashable):
                if not any(value == existing for existing in duplicates):
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
