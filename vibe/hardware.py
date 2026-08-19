"""Hardware capability discovery for UIs and APIs."""

from __future__ import annotations


def list_available_devices() -> list[str]:
    """Return available hardware accelerators in a user-facing format (e.g., 'cpu', 'cuda:0', 'xpu:0', 'mps')."""
    candidates: set[str] = {"cpu"}

    try:
        import torch

        # CUDA and ROCm (AMD HIP) share torch.cuda
        if torch.cuda.is_available():
            candidates.update({"cuda", "gpu"})
            for i in range(int(torch.cuda.device_count())):
                candidates.update({f"cuda:{i}", f"gpu:{i}"})

        # Intel GPU (XPU)
        xpu_backend = getattr(torch, "xpu", None)
        if xpu_backend is not None and xpu_backend.is_available():
            candidates.add("xpu")
            device_count = getattr(xpu_backend, "device_count", lambda: 1)()
            for i in range(int(device_count)):
                candidates.add(f"xpu:{i}")

        # Apple Silicon (MPS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            candidates.add("mps")
    except ImportError:
        pass

    # ONNX Runtime Execution Provider Discovery
    try:
        import onnxruntime as ort  # ty:ignore[unresolved-import, unused-ignore-comment]

        available = set(ort.get_available_providers())
        if "ROCMExecutionProvider" in available or "MIGraphXExecutionProvider" in available:
            candidates.add("rocm")
        if "OpenVINOExecutionProvider" in available:
            candidates.add("openvino")
        if "DmlExecutionProvider" in available:
            candidates.add("dml")
    except (ImportError, AttributeError):
        pass

    def _sort_key(dev: str) -> tuple[int, str, int]:
        if dev == "cpu":
            return (0, dev, -1)
        parts = dev.split(":")
        return (1, parts[0], int(parts[1]) if len(parts) > 1 else -1)

    return sorted(candidates, key=_sort_key)
