"""
Verifies that importing vibe, discovering plugins, and reading metadata
never eagerly loads heavy ML frameworks into memory.
"""

from __future__ import annotations

import subprocess
import sys


def test_no_heavy_imports_on_discovery() -> None:
    """
    Spawns a fresh, unpolluted Python process and verifies that importing vibe
    and describing all models does NOT import PyTorch, ONNX Runtime, Timm, or Transformers.
    """
    test_script = """
import sys
import vibe

# 1. Discover all plugins and serialize all descriptors
models = vibe.list_models()
descriptors = vibe.describe_all()

# 2. Check sys.modules for heavy frameworks
HEAVY_LIBRARIES = {
    "torch",
    "torchvision",
    "onnxruntime",
    "timm",
    "transformers",
    "safetensors",
}

leaked = [lib for lib in HEAVY_LIBRARIES if lib in sys.modules]
if leaked:
    print(f"LEAKED:{','.join(leaked)}")
    sys.exit(1)

print("CLEAN")
sys.exit(0)
"""

    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        error_output = result.stdout.strip() or result.stderr.strip()
        raise AssertionError(
            f"Eager imports detected! The following heavy libraries were loaded during metadata discovery:\n"
            f"  {error_output}\n\n"
            f"Check plugin module-level imports and move heavy frameworks inside "
            f"build_runtime() or preprocess()."
        )
