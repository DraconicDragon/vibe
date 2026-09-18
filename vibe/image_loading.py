"""
Shared image loading helpers for multi-model workflows with true generator streaming.
"""

from __future__ import annotations

import io
import itertools
import logging
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

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


def _safe_equals(a: Any, b: Any) -> bool:
    """Safe equality check handling identity, standard types, and array/tensor types."""
    if a is b:
        return True
    try:
        res = a == b
        if hasattr(res, "all") and callable(res.all):
            return bool(res.all())
        return bool(res)
    except Exception:
        return False


class _ReferenceTracker:
    """
    Tracks seen references with O(1) hashable lookups and a safe fallback
    for unhashable items (including array-like objects).
    """

    __slots__ = ("_error_cls", "_seen_hashable", "_seen_unhashable")

    def __init__(self, error_cls: type[Exception] = ValueError) -> None:
        self._seen_hashable: set[Any] = set()
        self._seen_unhashable: list[Any] = []
        self._error_cls = error_cls

    def add(self, r: Any) -> None:
        """Record a reference key, raising error_cls if it was already observed."""
        try:
            is_dup = r in self._seen_hashable
            hashable = True
        except TypeError:
            is_dup = any(_safe_equals(r, s) for s in self._seen_unhashable)
            hashable = False

        if is_dup:
            raise self._error_cls(f"Explicit refs must be unique, found duplicate: {r!r}")

        if hashable:
            self._seen_hashable.add(r)
        else:
            self._seen_unhashable.append(r)

    def add_all(self, refs: Iterable[Any]) -> None:
        """Record multiple reference keys in order, raising on the first duplicate."""
        for r in refs:
            self.add(r)


@dataclass(frozen=True)
class NormalizedInput:
    """
    Standardized representation of input items before decoding and batching.

    Attributes:
        sized: True if the input container has a known upfront length (Sequence, Mapping).
               False if the input is an unsized lazy stream (Generator, Iterator).
        values: The collection or iterator of image inputs (paths, PIL Images, arrays, bytes).
        refs: The collection or iterator of reference keys, or None for auto-generated integers.
        total: Known total item count for sized inputs, or None for open-ended streams.
    """

    sized: bool
    values: list[Any] | Iterator[Any]
    refs: list[Any] | Iterator[Any] | None
    total: int | None

    def __post_init__(self) -> None:
        if not self.sized:
            # Defensive invariant: ensure values and refs are true iterators when unsized
            if not isinstance(self.values, Iterator):
                object.__setattr__(self, "values", iter(self.values))
            if self.refs is not None and not isinstance(self.refs, Iterator):
                object.__setattr__(self, "refs", iter(self.refs))


@dataclass(frozen=True)
class ImageChunk:
    """Loaded image batch with refs aligned to original inputs."""

    start_index: int
    images: list[Any]
    refs: list[Any]


def _safe_close(target: Any) -> None:
    """Safely close a generator or closeable stream without raising unhandled exceptions."""
    if target is None:
        return

    if isinstance(target, Generator):
        try:
            target.close()
        except Exception as exc:
            logger.debug("Failed closing generator %r: %s", target, exc)
        return

    close_fn: Any = getattr(target, "close", None)
    if close_fn is not None:
        try:
            close_fn()
        except TypeError:
            pass  # Attribute 'close' was not a 0-argument callable
        except Exception as exc:
            logger.debug("Failed closing resource %r: %s", target, exc)


def _is_image_like(x: Any) -> bool:
    """
    Check if an object looks like a single image.

    Recognizes:
      - Paths and raw image bytes.
      - PIL Images.
      - 2D (H, W) or 3D (H, W, C)/(C, H, W) NumPy arrays and PyTorch tensors.

    Returns False for 4D+ batched tensors so they are treated as collections of images.
    """
    if isinstance(x, (str, Path, bytes, bytearray)):
        return True
    if isinstance(x, Image.Image):
        return True

    shape = getattr(x, "shape", None)
    if shape is not None:
        try:
            # 2D (H, W) or 3D (H, W, C)/(C, H, W) are single image arrays/tensors.
            # 4D+ (B, H, W, C)/(B, C, H, W) is a collection/batch of images.
            return len(shape) in (2, 3)
        except TypeError:
            return False

    return False


def normalize_input_format(
    images: Any,
    *,
    refs: Sequence[Any] | Iterable[Any] | None = None,
    error_cls: type[Exception] = ValueError,
) -> NormalizedInput:
    """
    Normalize image inputs into a structured NormalizedInput bundle.

    Dispatches across 5 input formats:
      1. Keyed Mapping: Dictionary of {ref: image_input}. Keys become explicit references.
         Unique keys are guaranteed by Python's mapping contract.
      2. Single Image-Like: Path, bytes, 2D/3D array, tensor, or PIL.Image. Wrapped as 1-item batch.
      3. Sized Non-Mapping Iterable: list, tuple, custom Sequence, or 4D batch array/tensor. Validates
         refs length and uniqueness upfront; total count is known.
      4. Unsized Lazy Stream: Generator, Iterator, DataLoader, or custom Iterable without
         a fixed length. Processed lazily without upfront memory materialization.
      5. Scalar Fallback: Any single object not recognized as a sequence.

    Streaming Semantic Caveats:
      - Duplicate Refs: For unsized streams, duplicate ref detection occurs incrementally as
        items are pulled. A duplicate ref encountered mid-stream raises error_cls at that point;
        items yielded prior to the error are not retracted.
      - Ref Length Mismatch: If explicit `refs` runs out before `images`, error_cls is raised
        immediately. If `refs` contains trailing excess items, the mismatch is detected only if
        the `images` stream is fully consumed (early cancellation terminates without checking).
    """
    # 1. Keyed dictionary input (keys are guaranteed unique by Python dictionary semantics)
    if isinstance(images, Mapping):
        if refs is not None:
            raise error_cls("Cannot pass explicit 'refs' when 'images' is already a dictionary mapping.")
        values = list(images.values())
        assigned_refs = list(images.keys())
        return NormalizedInput(sized=True, values=values, refs=assigned_refs, total=len(values))

    # 2. Single recognized image-like object
    if _is_image_like(images):
        values = [images]
        if refs is not None:
            assigned_refs = list(refs)
            if len(assigned_refs) != 1:
                raise error_cls(f"Expected 1 ref for single image, got {len(assigned_refs)}.")
            return NormalizedInput(sized=True, values=values, refs=assigned_refs, total=1)
        # Keep refs=None for auto-generated monotonic integer index (hot path)
        return NormalizedInput(sized=True, values=values, refs=None, total=1)

    # 3. Sized non-mapping iterable (bounded consumption for open-ended ref iterators)
    if (isinstance(images, Sequence) or hasattr(images, "__len__")) and not isinstance(images, (str, bytes, bytearray)):
        values = list(images)
        if refs is not None:
            if isinstance(refs, (Sequence, Mapping)) or hasattr(refs, "__len__"):
                assigned_refs = list(refs)
            else:
                assigned_refs = list(itertools.islice(refs, len(values) + 1))

            if len(assigned_refs) != len(values):
                raise error_cls(
                    f"Length mismatch: {len(values)} images provided, but {len(assigned_refs)} refs provided."
                )
            tracker = _ReferenceTracker(error_cls=error_cls)
            tracker.add_all(assigned_refs)
            return NormalizedInput(sized=True, values=values, refs=assigned_refs, total=len(values))

        # Keep refs=None for auto-generated monotonic integer indices (hot path)
        return NormalizedInput(sized=True, values=values, refs=None, total=len(values))

    # 4. Unsized lazy stream (Generator, Iterator, DataLoader, DB Cursors)
    if isinstance(images, Iterable) and not isinstance(images, (str, bytes, bytearray)):
        return NormalizedInput(
            sized=False,
            values=iter(images),
            refs=iter(refs) if refs is not None else None,
            total=None,
        )

    # 5. Scalar fallback
    values = [images]
    if refs is not None:
        assigned_refs = list(refs)
        if len(assigned_refs) != 1:
            raise error_cls(f"Expected 1 ref for single object, got {len(assigned_refs)}.")
        return NormalizedInput(sized=True, values=values, refs=assigned_refs, total=1)
    return NormalizedInput(sized=True, values=values, refs=None, total=1)


def _zip_with_refs(
    values_iter: Iterator[Any],
    refs_iter: Iterator[Any] | None,
    *,
    error_cls: type[Exception] = ValueError,
    validate_refs: bool = True,
) -> Generator[tuple[int, Any, Any], None, None]:
    """
    Yields (index, value, ref) triples with lockstep consumption and online duplicate checking.

    Notes on Length Validation:
      - If refs_iter is exhausted before values_iter, raises error_cls immediately.
      - If values_iter is exhausted first, closes refs_iter and raises error_cls. Note that
        if iteration is aborted early (e.g. caller breaks or task cancelled), excess trailing
        refs are not checked.
    """
    idx = 0
    # Hot-path optimization: default monotonic integer indices are guaranteed unique
    if refs_iter is None:
        for val in values_iter:
            yield (idx, val, idx)
            idx += 1
        return

    tracker = _ReferenceTracker(error_cls=error_cls) if validate_refs else None

    for val in values_iter:
        try:
            ref = next(refs_iter)
        except StopIteration:
            raise error_cls("Explicit 'refs' iterator exhausted before images stream ended.") from None

        if tracker is not None:
            tracker.add(ref)

        yield (idx, val, ref)
        idx += 1

    # Check if refs_iter has leftover items after values_iter naturally exhausted
    has_more_refs = False
    try:
        _ = next(refs_iter)
        has_more_refs = True
    except StopIteration:
        pass

    if has_more_refs:
        _safe_close(refs_iter)
        raise error_cls("Explicit 'refs' iterator contains more items than the images stream.")


def _chunk_triples(
    stream: Iterator[tuple[int, Any, Any]],
    batch_size: int,
) -> Generator[tuple[int, list[Any], list[Any]], None, None]:
    """Slice an iterator of (index, value, ref) triples into batches of batch_size."""
    while True:
        batch = list(itertools.islice(stream, batch_size))
        if not batch:
            break
        start_index = batch[0][0]
        batch_values = [item[1] for item in batch]
        batch_refs = [item[2] for item in batch]
        yield (start_index, batch_values, batch_refs)


def should_prefetch_image_loading(*, path_inputs: int) -> bool:
    """Prefetch only helps when there are multiple path-based inputs."""
    return path_inputs > 1


def load_image_if_path(value: Any | str, index: int, error_cls: type[Exception] = ValueError) -> Any:
    """Load an image from a filesystem path or raw bytes, applying EXIF orientation."""
    if isinstance(value, (bytes, bytearray)):
        try:
            with Image.open(io.BytesIO(value)) as img:
                img.load()
                try:
                    transposed = ImageOps.exif_transpose(img)
                    return transposed.copy() if transposed is img else transposed
                except Exception:
                    return img.copy()
        except Exception as exc:
            raise error_cls(f"Failed to decode image bytes at index {index}: {exc}") from exc

    if not isinstance(value, (str, Path)):
        return value

    path = Path(value)

    try:
        with Image.open(path) as img:
            img.load()
            try:
                transposed = ImageOps.exif_transpose(img)
                return transposed.copy() if transposed is img else transposed
            except Exception:
                return img.copy()
    except Exception as exc:
        raise error_cls(f"Failed to load image at index {index} from path '{path}': {exc}") from exc


def iter_load_images(
    images: Any,
    *,
    refs: Sequence[Any] | Iterable[Any] | None = None,
    batch_size: int = 1,
    prefetch_batch_limit: int = 8,
    prefetch: bool | None = None,
    cancel_check: CancelCheck | None = None,
    error_cls: type[Exception] = ValueError,
) -> Generator[ImageChunk, None, None]:
    """Normalize inputs and stream loaded image chunks."""
    norm = normalize_input_format(images, refs=refs, error_cls=error_cls)
    yield from iter_load_normalized(
        norm=norm,
        batch_size=batch_size,
        prefetch_batch_limit=prefetch_batch_limit,
        prefetch=prefetch,
        cancel_check=cancel_check,
        error_cls=error_cls,
    )


def iter_load_normalized(
    norm: NormalizedInput,
    *,
    batch_size: int = 1,
    prefetch_batch_limit: int = 8,
    prefetch: bool | None = None,
    cancel_check: CancelCheck | None = None,
    error_cls: type[Exception] = ValueError,
) -> Generator[ImageChunk, None, None]:
    """
    Stream loaded image chunks from a NormalizedInput bundle with guaranteed generator lifecycle cleanup.
    """
    values_iter = iter(norm.values)
    refs_iter = iter(norm.refs) if norm.refs is not None else None
    triple_stream: Any = None
    chunks: Any = None

    # Only unsized streams require online validation; sized sequences/mappings were pre-verified
    validate_refs = (not norm.sized) and (norm.refs is not None)

    try:
        triple_stream = _zip_with_refs(
            values_iter,
            refs_iter,
            error_cls=error_cls,
            validate_refs=validate_refs,
        )
        chunks = _chunk_triples(triple_stream, batch_size=batch_size)

        # Prefetch heuristic and diagnostic logging
        if prefetch is None:
            if norm.sized and isinstance(norm.values, list):
                path_inputs = sum(1 for value in norm.values if isinstance(value, (str, Path)))
                use_prefetch = should_prefetch_image_loading(path_inputs=path_inputs)
                if path_inputs > 0:
                    logger.info("Loading input images paths=%s total_inputs=%s", path_inputs, norm.total)
                    if use_prefetch:
                        logger.debug("Image prefetch enabled for %s path inputs", path_inputs)
            else:
                use_prefetch = False
        else:
            use_prefetch = bool(prefetch)
            if use_prefetch:
                logger.debug("Image prefetch explicitly enabled")

        def _load_batch(batch_values: list[Any], start_idx: int) -> list[Any]:
            batch = []
            for i, val in enumerate(batch_values):
                if cancel_check:
                    cancel_check()
                batch.append(load_image_if_path(val, index=start_idx + i, error_cls=error_cls))
            return batch

        if not use_prefetch:
            for start_idx, batch_values, batch_refs in chunks:
                if cancel_check:
                    cancel_check()
                loaded_images = _load_batch(batch_values, start_idx)
                yield ImageChunk(start_index=start_idx, images=loaded_images, refs=batch_refs)
            return

        # Queue-based prefetch with unified queue helper
        from collections import deque

        max_prefetch_batches = max(1, prefetch_batch_limit // batch_size)
        futures_queue: deque[tuple[int, list[Any], Future[list[Any]]]] = deque()

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vibe-image-loader")

        def _enqueue_next() -> bool:
            try:
                nxt_start, nxt_vals, nxt_refs = next(chunks)
                nxt_future = executor.submit(_load_batch, nxt_vals, nxt_start)
                futures_queue.append((nxt_start, nxt_refs, nxt_future))
                return True
            except StopIteration:
                return False

        try:
            # Prefill queue up to limit
            while len(futures_queue) < max_prefetch_batches:
                if not _enqueue_next():
                    break

            while futures_queue:
                start_idx, batch_refs, future = futures_queue.popleft()
                loaded_images = _await_loaded_chunk(future, cancel_check=cancel_check)

                # Maintain prefetch buffer
                _enqueue_next()

                yield ImageChunk(
                    start_index=start_idx,
                    images=loaded_images,
                    refs=batch_refs,
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    finally:
        # Close active iterators from outermost stage inward
        _safe_close(chunks)
        _safe_close(triple_stream)
        _safe_close(values_iter)
        if refs_iter is not None:
            _safe_close(refs_iter)


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
