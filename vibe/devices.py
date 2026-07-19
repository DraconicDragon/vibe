"""Hardware device selection and resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

BackendName = Literal["onnx", "pytorch"]

_DEVICE_PATTERN = re.compile(r"^(?P<family>[a-z]+)(?:[:](?P<index>\d+))?$")


@dataclass(frozen=True)
class DeviceSpec:
    """Structured representation of a compute device request."""

    family: str
    index: int | None = None

    @classmethod
    def parse(cls, device: str | None) -> DeviceSpec:
        value = str(device or "cpu").strip().lower()
        if not value:
            return cls(family="cpu")

        match = _DEVICE_PATTERN.match(value)
        if not match:
            if re.match(r"^[a-z]+\d+$", value):
                raise ValueError(f"Invalid device format '{value}'. Use 'family:index' syntax (e.g. 'cuda:0').")
            return cls(family=value)

        index_str = match.group("index")
        return cls(family=match.group("family"), index=int(index_str) if index_str is not None else None)

    def to_backend_string(self, backend: BackendName) -> str:
        """Resolve the requested device to a backend-specific string."""
        if self.family in {"cpu", "auto"}:
            return self.family

        if backend == "pytorch":
            return self._to_pytorch()
        elif backend == "onnx":
            return self._to_onnx()

        raise ValueError(f"Unknown backend '{backend}'.")

    def _to_pytorch(self) -> str:
        if self.family in {"gpu", "cuda"}:
            return f"cuda:{self.index}" if self.index is not None else "cuda"
        if self.family == "mps":
            return "mps"

        raise ValueError(f"Unsupported PyTorch device family '{self.family}'.")

    def _to_onnx(self) -> str:
        if self.family in {"gpu", "cuda"}:
            base = "cuda" if self.family == "cuda" else "gpu"
            return f"{base}:{self.index}" if self.index is not None else base

        if self.family in {"rocm", "dml"}:
            return f"{self.family}:{self.index}" if self.index is not None else self.family

        raise ValueError(f"Unsupported ONNX device family '{self.family}'.")


def normalize_device_string(device: str | None, *, backend: BackendName) -> str:
    """Convenience wrapper to parse and translate a device string for a specific backend."""
    return DeviceSpec.parse(device).to_backend_string(backend)


def list_available_devices() -> list[str]:
    """Return available device selectors in user-facing format."""
    candidates: set[str] = {"cpu"}

    try:
        import torch

        if torch.cuda.is_available():
            candidates.update({"cuda", "gpu"})
            for i in range(int(torch.cuda.device_count())):
                candidates.update({f"cuda:{i}", f"gpu:{i}"})

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and callable(getattr(mps_backend, "is_available", None)) and mps_backend.is_available():
            candidates.add("mps")
    except ImportError:
        pass

    try:
        import onnxruntime as ort  # ty:ignore[unresolved-import]

        if hasattr(ort, "get_available_providers"):
            available = set(str(p) for p in ort.get_available_providers())
            if "ROCMExecutionProvider" in available:
                candidates.add("rocm")
            if "DmlExecutionProvider" in available:
                candidates.add("dml")
    except ImportError:
        pass

    def _sort_key(dev: str) -> tuple[int, str, int]:
        if dev == "cpu":
            return (0, dev, -1)
        parts = dev.split(":")
        return (1, parts[0], int(parts[1]) if len(parts) > 1 else -1)

    return sorted(candidates, key=_sort_key)
