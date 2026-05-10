"""
Device discovery utility — lists available GPUs and hardware accelerators.
Also shows registered models, result processors, and their metadata.
"""

from __future__ import annotations

from collections import defaultdict

import vibe


def print_available_devices() -> None:
    """Print all available device selectors."""
    devices = vibe.list_available_devices()
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


def print_model_families() -> None:
    """Print all registered models grouped by family."""
    print("\nModel Families:")
    print("-" * 80)

    infos = sorted(vibe.describe_all(), key=lambda info: info.model_id)
    grouped: dict[str, list[vibe.ModelPluginInfo]] = defaultdict(list)

    for info in infos:
        grouped[info.family_name].append(info)

    total_models = len(infos)
    print(f"Total models: {total_models}")
    print()

    for family_name in sorted(grouped.keys()):
        family_models = sorted(grouped[family_name], key=lambda info: info.display_name.lower() or info.model_id)
        model_count = len(family_models)

        first_two_ids = [info.model_id for info in family_models[:2]]
        ids_display = ", ".join(first_two_ids)
        if model_count > 2:
            ids_display += f" (... +{model_count - 2} more)"

        print(f"  • {family_name}")
        print(f"      Models: {model_count}")
        print(f"      IDs: {ids_display}")
        print()


def print_supported_result_processors() -> None:
    """Print result processors supported by registered models."""
    print("\nSupported Result Processors (from models):")
    print("-" * 80)

    all_processors: set[str] = set()
    processor_family_map: dict[str, set[str]] = defaultdict(set)

    for info in vibe.describe_all():
        for processor in info.supported_processors:
            all_processors.add(processor)
            processor_family_map[processor].add(info.family_name)

    if not all_processors:
        print("  (No result processors registered)")
        return

    print(f"Total unique processors: {len(all_processors)}\n")
    for processor in sorted(all_processors):
        families = sorted(processor_family_map[processor])
        print(f"  • {processor}")
        print(f"      Used by families: {', '.join(families) or '—'}")


def print_available_result_processors() -> None:
    """Print all available result processor classes with their metadata."""
    processors = vibe.list_processors()
    print("\nAvailable Result Processors (library):")
    print("-" * 80)
    print(f"Total processors: {len(processors)}\n")

    for info in processors:
        print(f"  • {info.processor_id}")
        print(f"      Display name : {info.display_name}")
        print(f"      Description  : {info.description}")
        if info.params:
            print("      Parameters:")
            for p in info.params:
                d = p.to_dict()
                required_marker = " (required)" if d["required"] else ""
                default_str = "" if d["required"] else f", default: {d['default']!r}"
                print(f"        - {d['name']}: {d['type'] or 'any'}{required_marker}{default_str}")
                if d["description"]:
                    print(f"            {d['description']}")
        else:
            print("      Parameters   : none")
        print()


if __name__ == "__main__":
    print_available_devices()
    print_pytorch_info()
    print_onnxruntime_info()
    print_model_families()
    print_supported_result_processors()
    print_available_result_processors()
