from __future__ import annotations

from pathlib import Path

from autotagger.backends.runtime import onnx as onnx_runtime


class _FakeOrt:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_available_providers(self) -> list[str]:
        return list(self._providers)


def test_configure_linux_cuda_library_path_adds_nvidia_lib_dirs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onnx_runtime.sys, "platform", "linux")

    nvidia_root = tmp_path / "nvidia"
    cublas_lib = nvidia_root / "cublas" / "lib"
    cudnn_lib = nvidia_root / "cudnn" / "lib"
    cublas_lib.mkdir(parents=True)
    cudnn_lib.mkdir(parents=True)

    monkeypatch.setattr(onnx_runtime, "_iter_candidate_nvidia_lib_dirs", lambda: [nvidia_root])
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")

    added = onnx_runtime._configure_linux_cuda_library_path()

    assert str(cublas_lib) in added
    assert str(cudnn_lib) in added

    parts = [p for p in onnx_runtime.os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    assert str(cublas_lib) in parts
    assert str(cudnn_lib) in parts
    assert "/existing/lib" in parts


def test_configure_linux_cuda_library_path_keeps_existing_entries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(onnx_runtime.sys, "platform", "linux")

    nvidia_root = tmp_path / "nvidia"
    existing_lib = nvidia_root / "curand" / "lib"
    existing_lib.mkdir(parents=True)

    monkeypatch.setattr(onnx_runtime, "_iter_candidate_nvidia_lib_dirs", lambda: [nvidia_root])
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{existing_lib}:/existing/lib")

    added = onnx_runtime._configure_linux_cuda_library_path()

    assert added == []
    assert onnx_runtime.os.environ.get("LD_LIBRARY_PATH") == f"{existing_lib}:/existing/lib"


def test_resolve_onnx_provider_chain_prefers_cuda_then_cpu_for_gpu_device() -> None:
    providers, options = onnx_runtime.resolve_onnx_provider_chain(
        device="gpu",
        requested_providers=None,
        ort_module=_FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert options == [{"device_id": "0"}, {}]


def test_resolve_onnx_provider_chain_uses_rocm_when_cuda_missing() -> None:
    providers, options = onnx_runtime.resolve_onnx_provider_chain(
        device="gpu:1",
        requested_providers=None,
        ort_module=_FakeOrt(["ROCMExecutionProvider", "CPUExecutionProvider"]),
    )

    assert providers == ["ROCMExecutionProvider", "CPUExecutionProvider"]
    assert options == [{"device_id": "1"}, {}]


def test_resolve_onnx_provider_chain_cpu_device_forces_cpu_only() -> None:
    providers, options = onnx_runtime.resolve_onnx_provider_chain(
        device="cpu",
        requested_providers=None,
        ort_module=_FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )

    assert providers == ["CPUExecutionProvider"]
    assert options is None


def test_resolve_onnx_provider_chain_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AUTOTAGGER_ONNX_PROVIDERS", "CPUExecutionProvider,CUDAExecutionProvider")

    providers, options = onnx_runtime.resolve_onnx_provider_chain(
        device="gpu",
        requested_providers=None,
        ort_module=_FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )

    assert providers == ["CPUExecutionProvider", "CUDAExecutionProvider"]
    assert options == [{}, {"device_id": "0"}]
