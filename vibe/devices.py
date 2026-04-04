from __future__ import annotations

import re
from typing import Literal

BackendName = Literal["onnx", "pytorch"]

_NO_COLON_INDEX_PATTERN = re.compile(r"^(cuda|gpu|rocm|dml)(\d+)$")
_COLON_INDEX_PATTERN = re.compile(r"^(cuda|gpu|rocm|dml):(\d+)$")


def normalize_device_string(device: str, *, backend: BackendName) -> str:
    """Normalize user-facing device strings into backend-specific selectors."""
    value = str(device or "cpu").strip().lower()
    if not value:
        return "cpu"

    no_colon_match = _NO_COLON_INDEX_PATTERN.match(value)
    if no_colon_match:
        family = no_colon_match.group(1)
        index = no_colon_match.group(2)
        raise ValueError(f"Invalid device '{value}'. Use '{family}:{index}' syntax.")

    if value in {"cpu", "auto"}:
        return value

    if value == "gpu":
        return "cuda" if backend == "pytorch" else "gpu"

    if value == "cuda":
        return "cuda" if backend == "pytorch" else "gpu"

    if value == "mps":
        if backend != "pytorch":
            raise ValueError("'mps' is only valid for the pytorch backend.")
        return "mps"

    index_match = _COLON_INDEX_PATTERN.match(value)
    if index_match:
        family = index_match.group(1)
        index = int(index_match.group(2))

        if family == "gpu":
            return f"cuda:{index}" if backend == "pytorch" else f"gpu:{index}"
        if family == "cuda":
            return f"cuda:{index}" if backend == "pytorch" else f"gpu:{index}"
        if family == "rocm":
            if backend == "pytorch":
                raise ValueError("'rocm' selectors are not valid for the pytorch backend device string.")
            return f"rocm:{index}"
        if family == "dml":
            if backend == "pytorch":
                raise ValueError("'dml' selectors are not valid for the pytorch backend device string.")
            return f"dml:{index}"

    if backend == "pytorch":
        raise ValueError(f"Unsupported pytorch device '{value}'.")

    if value in {"rocm", "dml"}:
        return value

    raise ValueError(f"Unsupported onnx device '{value}'.")


def list_available_devices() -> list[str]:
    """Return available device selectors in user-facing format."""
    candidates: list[str] = ["cpu"]

    try:
        import torch

        if torch.cuda.is_available():
            candidates.extend(["cuda", "gpu"])
            count = int(torch.cuda.device_count())
            for i in range(count):
                candidates.append(f"cuda:{i}")
                candidates.append(f"gpu:{i}")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and callable(getattr(mps_backend, "is_available", None)):
            if bool(mps_backend.is_available()):
                candidates.append("mps")
    except Exception:
        pass

    try:
        import onnxruntime as ort

        available = set(str(p) for p in ort.get_available_providers())
        if "ROCMExecutionProvider" in available:
            candidates.append("rocm")
        if "DmlExecutionProvider" in available:
            candidates.append("dml")
    except Exception:
        pass

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
