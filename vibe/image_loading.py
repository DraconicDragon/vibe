"""Shared image loading helpers for multi-model workflows."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

logger = logging.getLogger(__name__)

CancelCheck = Callable[[], None]


try:
    import pillow_jxl  # noqa: F401

    _HAS_PILLOW_JXL = True
except ImportError:
    _HAS_PILLOW_JXL = False

try:
    import pillow_heif  # noqa: F401

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


def load_image_if_path(
    value: Any | str,
    index: int,
    cancel_check: CancelCheck | None = None,
    error_cls: type[Exception] = ValueError,
    has_pillow_jxl: bool | None = None,
    has_pillow_heif: bool | None = None,
) -> Any:
    if cancel_check is not None:
        cancel_check()
    if not isinstance(value, (str, Path)):
        return value

    from PIL import Image

    path = Path(value)
    logger.debug("Loading image index=%s path=%s", index, path)
    try:
        with Image.open(path) as img:
            loaded = img.copy()
    except Exception as exc:
        suffix = Path(path).suffix.lower()
        hint = ""
        pillow_jxl_available = _HAS_PILLOW_JXL if has_pillow_jxl is None else has_pillow_jxl
        pillow_heif_available = _HAS_PILLOW_HEIF if has_pillow_heif is None else has_pillow_heif
        if suffix == ".jxl" and not pillow_jxl_available:
            hint = " Install 'pillow-jxl-plugin' to enable JPEG XL support: pip install pillow-jxl-plugin"
        elif suffix in {".heif", ".heic"} and not pillow_heif_available:
            hint = " Install 'pillow-heif' to enable HEIF/HEIC support: pip install pillow-heif"
        raise error_cls(f"Failed to load image at index {index} from path '{path}': {exc}.{hint}") from exc

    if cancel_check is not None:
        cancel_check()
    logger.debug("Loaded image index=%s path=%s", index, path)
    return loaded


def load_images(
    values: Sequence[Any | str],
    *,
    start_index: int = 0,
    load_image_fn: Callable[[Any | str, int], Any] | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[Any]:
    normalized_images: list[Any] = []
    for offset, value in enumerate(values):
        if cancel_check is not None:
            cancel_check()
        loader = load_image_fn or load_image_if_path
        normalized_images.append(loader(value, start_index + offset))
    if cancel_check is not None:
        cancel_check()
    return normalized_images


def iter_loaded_image_chunks(
    values: Sequence[Any | str],
    *,
    chunk_size: int,
    use_prefetch: bool,
    load_image_fn: Callable[[Any | str, int], Any] | None = None,
    cancel_check: CancelCheck | None = None,
) -> Iterator[tuple[int, list[Any]]]:
    if not values:
        return

    if not use_prefetch:
        for start in range(0, len(values), chunk_size):
            if cancel_check is not None:
                cancel_check()
            chunk_values = values[start : start + chunk_size]
            yield (
                start,
                load_images(
                    chunk_values,
                    start_index=start,
                    load_image_fn=load_image_fn,
                    cancel_check=cancel_check,
                ),
            )
        return

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vibe-image-loader")
    fast_shutdown = False
    try:
        start = 0
        future: Future[list[Any]] = executor.submit(
            load_images,
            values[start : start + chunk_size],
            start_index=start,
            load_image_fn=load_image_fn,
            cancel_check=cancel_check,
        )

        while True:
            loaded_images = _await_loaded_chunk(future, cancel_check=cancel_check)
            next_start = start + chunk_size
            next_future: Future[list[Any]] | None = None
            if next_start < len(values):
                next_future = executor.submit(
                    load_images,
                    values[next_start : next_start + chunk_size],
                    start_index=next_start,
                    load_image_fn=load_image_fn,
                    cancel_check=cancel_check,
                )

            yield start, loaded_images

            if next_future is None:
                break
            start = next_start
            future = next_future
    except Exception:
        fast_shutdown = True
        raise
    finally:
        executor.shutdown(wait=not fast_shutdown, cancel_futures=True)


def iter_load_images(
    images: Any | str | list[Any] | list[str] | list[tuple[Any | str, Any]],
    *,
    batch_size: int = 1,
    prefetch: bool | None = None,
    cancel_check: CancelCheck | None = None,
    error_cls: type[Exception] = ValueError,
    has_pillow_jxl: bool | None = None,
    has_pillow_heif: bool | None = None,
) -> Iterator[ImageChunk]:
    """Yield image batches loaded from the input list.

    The input can be bare images/paths or (image_or_path, ref) tuples.
    Each yielded chunk keeps the original refs aligned with the loaded images.
    """
    values, refs = normalize_input_format(images, error_cls=error_cls)
    if not values:
        return

    path_inputs = sum(1 for value in values if isinstance(value, (str, Path)))
    use_prefetch = should_prefetch_image_loading(path_inputs=path_inputs) if prefetch is None else bool(prefetch)

    def _loader(value: Any | str, index: int) -> Any:
        return load_image_if_path(
            value,
            index=index,
            cancel_check=cancel_check,
            error_cls=error_cls,
            has_pillow_jxl=has_pillow_jxl,
            has_pillow_heif=has_pillow_heif,
        )

    for start, chunk_images in iter_loaded_image_chunks(
        values,
        chunk_size=batch_size,
        use_prefetch=use_prefetch,
        load_image_fn=_loader,
        cancel_check=cancel_check,
    ):
        yield ImageChunk(
            start_index=start,
            images=chunk_images,
            refs=refs[start : start + len(chunk_images)],
        )


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
