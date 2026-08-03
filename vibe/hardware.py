"""Hardware capability discovery for UIs and APIs."""

from __future__ import annotations


def list_available_devices() -> list[str]:
    """Return available hardware accelerators in a user-facing format (e.g., 'cpu', 'cuda:0', 'mps')."""
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
            available = {str(p) for p in ort.get_available_providers()}
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
