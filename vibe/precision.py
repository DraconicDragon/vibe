"""Runtime precision definitions and parsing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PrecisionPolicy(str, Enum):
    """Explicit policies for computation and weight precision."""

    AUTO = "auto"
    PRESERVE = "preserve"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True)
class PrecisionRequest:
    """A structured request separating how weights and computation should be handled."""

    weight: PrecisionPolicy
    compute: PrecisionPolicy
    fallback_allowed: bool = True


@dataclass(frozen=True)
class ResolvedPrecisionPlan:
    """The actual precision strategy decided by the backend during load."""

    weight_dtype: str
    compute_dtype: str
    autocast_enabled: bool


def parse_precision(precision: str | PrecisionRequest | None) -> PrecisionRequest:
    """Normalize user precision selector into a canonical PrecisionRequest."""
    if isinstance(precision, PrecisionRequest):
        return precision

    value = str(precision or "auto").strip().lower()

    if not value or value == "auto":
        # Auto means: do not aggressively cast weights, but use autocast compute if hardware supports it
        return PrecisionRequest(weight=PrecisionPolicy.PRESERVE, compute=PrecisionPolicy.AUTO)

    try:
        policy = PrecisionPolicy(value)
        # Explicit strings like "fp16" mean: cast weights AND use fp16 compute
        return PrecisionRequest(weight=policy, compute=policy)
    except ValueError:
        valid = [p.value for p in PrecisionPolicy if p not in (PrecisionPolicy.AUTO, PrecisionPolicy.PRESERVE)]
        raise ValueError(f"Unsupported precision '{precision}'. Choose from: auto, {valid}") from None
