"""
Device discovery utility — lists available GPUs and hardware accelerators.
"""

import autotagger


def print_available_devices() -> None:
    """Print all available device selectors."""
    devices = autotagger.list_available_devices()
    print("Available devices:")
    for device in devices:
        print(f"  • {device}")


def print_pytorch_info() -> None:
    """Print PyTorch CUDA/MPS availability (if installed)."""
    try:
        import torch

        print("\nPyTorch info:")
        print(f"  Installed: Yes (v{torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    [{i}] {torch.cuda.get_device_name(i)}")
        print(f"  MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    except ImportError:
        print("\nPyTorch: Not installed")


def print_onnxruntime_info() -> None:
    # NOTE: if it shows azure ep + cpu ep then its cpu only, azure ep is remote
    """Print ONNX Runtime providers (if installed)."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        print("\nONNX Runtime info:")
        print(f"  Installed: Yes (v{ort.__version__})")
        print("  Available providers:")
        for provider in providers:
            print(f"    • {provider}")
    except ImportError:
        print("\nONNX Runtime: Not installed")


if __name__ == "__main__":
    print_available_devices()
    print_pytorch_info()
    print_onnxruntime_info()
