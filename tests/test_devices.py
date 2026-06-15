from __future__ import annotations

import pytest

from vibe.devices import list_available_devices, normalize_device_string


def test_normalize_device_string_pytorch_accepts_cuda_syntax_and_gpu_alias() -> None:
    assert normalize_device_string("cuda", backend="pytorch") == "cuda"
    assert normalize_device_string("cuda:0", backend="pytorch") == "cuda:0"
    assert normalize_device_string("gpu", backend="pytorch") == "cuda"
    assert normalize_device_string("gpu:2", backend="pytorch") == "cuda:2"


def test_normalize_device_string_onnx_accepts_gpu_syntax_and_cuda_alias() -> None:
    assert normalize_device_string("gpu", backend="onnx") == "gpu"
    assert normalize_device_string("gpu:0", backend="onnx") == "gpu:0"
    assert normalize_device_string("cuda", backend="onnx") == "gpu"
    assert normalize_device_string("cuda:1", backend="onnx") == "gpu:1"


def test_normalize_device_string_rejects_no_colon_index_syntax() -> None:
    with pytest.raises(ValueError, match="Use 'cuda:0' syntax"):
        normalize_device_string("cuda0", backend="pytorch")

    with pytest.raises(ValueError, match="Use 'gpu:1' syntax"):
        normalize_device_string("gpu1", backend="onnx")
