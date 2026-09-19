from __future__ import annotations

from vibe.backends.base import ExecutionPreference, HardwareIntent
from vibe.hardware import list_available_devices


def test_execution_preference_parse_auto() -> None:
    assert ExecutionPreference.parse(None).intent == HardwareIntent.AUTO
    assert ExecutionPreference.parse("").intent == HardwareIntent.AUTO
    assert ExecutionPreference.parse("auto").intent == HardwareIntent.AUTO


def test_execution_preference_parse_cpu() -> None:
    pref = ExecutionPreference.parse("cpu")
    assert pref.intent == HardwareIntent.CPU
    assert pref.ordinal is None

    pref_ord = ExecutionPreference.parse("cpu:0")
    assert pref_ord.intent == HardwareIntent.CPU
    assert pref_ord.ordinal == 0


def test_execution_preference_parse_accelerator_hints() -> None:
    # Generic "gpu" maps to ACCELERATOR with hint=None
    gpu = ExecutionPreference.parse("gpu")
    assert gpu.intent == HardwareIntent.ACCELERATOR
    assert gpu.hint is None

    gpu_1 = ExecutionPreference.parse("gpu:1")
    assert gpu_1.intent == HardwareIntent.ACCELERATOR
    assert gpu_1.ordinal == 1

    # Specific hardware hints ("cuda", "mps", "rocm", "xpu") are preserved
    cuda_0 = ExecutionPreference.parse("cuda:0")
    assert cuda_0.intent == HardwareIntent.ACCELERATOR
    assert cuda_0.hint == "cuda"
    assert cuda_0.ordinal == 0

    mps = ExecutionPreference.parse("mps")
    assert mps.intent == HardwareIntent.ACCELERATOR
    assert mps.hint == "mps"


def test_execution_preference_serialization() -> None:
    pref = ExecutionPreference.parse("cuda:2")
    data = pref.to_dict()
    assert data == {"intent": "accelerator", "ordinal": 2, "hint": "cuda"}


def test_list_available_devices_includes_cpu() -> None:
    devices = list_available_devices()
    assert isinstance(devices, list)
    assert "cpu" in devices
