import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _is_structured_jtp3_batch(item: Any) -> bool:
    try:
        from vibe.plugins.jtp_hydra.jtp_hydra_modelplugin import JTPHydraBatch
    except Exception:
        JTPHydraBatch = None  # type: ignore[assignment]

    if JTPHydraBatch is not None and isinstance(item, JTPHydraBatch):
        return True

    return all(hasattr(item, field) for field in ("patches", "sizes"))


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
    first = chunk[0]
    if _is_structured_jtp3_batch(first):
        try:
            import torch

            from vibe.plugins.jtp_hydra.jtp_hydra_modelplugin import JTPHydraBatch

            patches = torch.stack([item.patches for item in chunk], dim=0)
            sizes = torch.stack([item.sizes for item in chunk], dim=0)
            logger.debug(
                "Stacked JTP-3 / Hydra batch batch_size=%d patches_shape=%s sizes_shape=%s",
                len(chunk),
                patches.shape,
                sizes.shape,
            )
            return JTPHydraBatch(patches, sizes)
        except Exception as exc:
            logger.error(
                "Failed to stack JTP-3 / Hydra batch for model_id=%s sample_descriptions=%s",
                model_id,
                [_describe_preprocessed_sample(item) for item in chunk],
            )
            raise ValueError(
                "Could not build a true JTP-3 / Hydra batch. This usually means preprocessed "
                f"patch tensors have incompatible shapes for stacking. Details: {exc}"
            ) from exc

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
            raise ValueError(
                "Could not build a true batch tensor. This usually means preprocessed "
                f"samples have incompatible shapes for concatenation. Details: {exc}"
            ) from exc

    # Torch-like tensor handling without hard dependency.
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
    raise ValueError("Unsupported preprocessed tensor type for true batching. Use batch_method='sequential'.")


def split_batch_output(raw_output: Any, expected: int, model_id: str) -> list[Any]:
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
        raise ValueError(
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
    raise ValueError(f"Backend output batch dimension mismatch: expected {expected}, got {arr.shape}.")
