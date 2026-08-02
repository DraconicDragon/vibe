"""Model-agnostic tensor and batch collation utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vibe.exceptions import SessionError

logger = logging.getLogger(__name__)


def _is_structured_container(item: Any) -> bool:
    """Check if an item is a custom batch container (e.g. NaFlex/JTP batch)."""
    return hasattr(item, "patches") and hasattr(item, "sizes")


def _describe_preprocessed_sample(item: Any) -> dict[str, Any]:
    shape = getattr(item, "shape", None)
    if shape is not None:
        return {
            "type": type(item).__name__,
            "shape": tuple(shape) if isinstance(shape, tuple) else shape,
            "dtype": str(getattr(item, "dtype", None)),
        }

    parts: dict[str, Any] = {"type": type(item).__name__}
    for field in ("patches", "sizes"):
        value = getattr(item, field, None)
        if value is not None:
            parts[field] = {
                "shape": tuple(getattr(value, "shape", ())) if getattr(value, "shape", None) is not None else None,
                "dtype": str(getattr(value, "dtype", None)),
            }
    return parts


def stack_batch(chunk: list[Any], model_id: str) -> Any:
    """Collate a list of preprocessed samples into a single batch data structure."""
    if not chunk:
        raise SessionError("Cannot stack an empty batch chunk.")

    first = chunk[0]
    cls = type(first)

    # 1. Generic Structured Batch Containers (e.g., NaFlex / SigLIP2)
    if _is_structured_container(first):
        try:
            import torch

            patches = torch.stack([item.patches for item in chunk], dim=0)
            sizes = torch.stack([item.sizes for item in chunk], dim=0)
            logger.debug(
                "Stacked structured batch model_id=%s batch_size=%d patches_shape=%s sizes_shape=%s",
                model_id,
                len(chunk),
                patches.shape,
                sizes.shape,
            )
            return cls(patches, sizes)
        except Exception as exc:
            logger.error(
                "Failed to stack structured batch for model_id=%s sample_descriptions=%s",
                model_id,
                [_describe_preprocessed_sample(item) for item in chunk],
            )
            raise SessionError(f"Could not collate structured batch for model '{model_id}': {exc}") from exc

    # 2. NumPy Arrays
    if isinstance(first, np.ndarray):
        try:
            stacked = np.concatenate(chunk, axis=0)
            logger.debug("Stacked numpy batch shape=%s dtype=%s", stacked.shape, stacked.dtype)
            return stacked
        except Exception as exc:
            logger.error(
                "Failed to stack numpy batch for model_id=%s sample_shapes=%s",
                model_id,
                [getattr(item, "shape", None) for item in chunk],
            )
            raise SessionError(f"Could not concatenate NumPy samples for model '{model_id}': {exc}") from exc

    # 3. PyTorch Tensors
    try:
        import torch

        if isinstance(first, torch.Tensor):
            stacked = torch.cat(chunk, dim=0)
            logger.debug("Stacked torch batch shape=%s dtype=%s", stacked.shape, stacked.dtype)
            return stacked
    except Exception:
        pass

    logger.error(
        "Unsupported preprocessed batch type for model_id=%s sample_descriptions=%s",
        model_id,
        [_describe_preprocessed_sample(item) for item in chunk],
    )
    raise SessionError(f"Unsupported preprocessed tensor type '{cls.__name__}' for batching.")


def split_batch_output(raw_output: Any, expected: int, model_id: str) -> list[Any]:
    """Split a batched raw model output back into per-sample raw outputs."""
    shape = getattr(raw_output, "shape", None)
    ndim = getattr(raw_output, "ndim", None)

    if ndim == 0:
        return [raw_output for _ in range(expected)]

    if shape is not None and len(shape) > 0 and shape[0] == expected:
        return [raw_output[i : i + 1] for i in range(expected)]

    if expected == 1:
        return [raw_output]

    try:
        arr = np.asarray(raw_output)
    except Exception as exc:
        raise SessionError(
            f"Backend output batch dimension mismatch: expected {expected}, got unknown output type."
        ) from exc

    if arr.ndim == 0:
        return [arr for _ in range(expected)]
    if arr.shape[0] == expected:
        return [arr[i : i + 1] for i in range(expected)]

    logger.error(
        "Backend output batch mismatch model_id=%s expected=%s actual_shape=%s",
        model_id,
        expected,
        arr.shape,
    )
    raise SessionError(f"Backend output batch dimension mismatch: expected {expected}, got {arr.shape}.")
