from __future__ import annotations

from typing import Final

_VALID_PRECISIONS: Final[set[str]] = {"auto", "fp32", "fp16", "bf16", "int8_ov"}


def normalize_precision_string(precision: str | None) -> str:
    """Normalize user precision selector into a canonical value."""
    value = str("auto" if precision is None else precision).strip().lower()
    if not value:
        return "auto"

    aliases = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
        "ov": "int8_ov",
    }
    value = aliases.get(value, value)

    if value not in _VALID_PRECISIONS:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from: {sorted(_VALID_PRECISIONS)}")
    return value
