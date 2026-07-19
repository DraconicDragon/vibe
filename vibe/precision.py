"""Runtime precision definitions and parsing."""

from __future__ import annotations

from enum import Enum


class Precision(str, Enum):
    """Supported computation and weight precisions."""

    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    # INT8_OV = "int8_ov"


def parse_precision(precision: str | Precision | None) -> Precision:
    """Normalize user precision selector into a canonical Precision enum."""
    if isinstance(precision, Precision):
        return precision

    value = str(precision or "auto").strip().lower()
    if not value:
        return Precision.AUTO

    try:
        return Precision(value)
    except ValueError:
        valid = [p.value for p in Precision]
        raise ValueError(f"Unsupported precision '{precision}'. Choose from: {valid}") from None
