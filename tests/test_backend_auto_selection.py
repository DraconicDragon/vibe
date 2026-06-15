from __future__ import annotations

from pathlib import Path

import pytest

import vibe
from vibe.backends.base import Backend
from vibe.loader import FileMap
from vibe.plugins.wd_tagger import WDEva02Plugin
from vibe.session import SessionError
from vibe.session_factory import _auto_select_backend, build_session


def test_auto_select_backend_prefers_pytorch_when_only_torch_has_accel(monkeypatch) -> None:
    monkeypatch.setattr("vibe.session_factory._onnx_runtime_capabilities", lambda: (True, False))
    monkeypatch.setattr("vibe.session_factory._pytorch_runtime_capabilities", lambda: (True, True))

    selected = _auto_select_backend(WDEva02Plugin, requested_device="auto")

    assert selected == Backend.PYTORCH


def test_auto_select_backend_prefers_onnx_when_only_onnx_has_accel(monkeypatch) -> None:
    monkeypatch.setattr("vibe.session_factory._onnx_runtime_capabilities", lambda: (True, True))
    monkeypatch.setattr("vibe.session_factory._pytorch_runtime_capabilities", lambda: (True, False))

    selected = _auto_select_backend(WDEva02Plugin, requested_device="auto")

    assert selected == Backend.ONNX


def test_auto_select_backend_prefers_onnx_when_both_equivalent(monkeypatch) -> None:
    monkeypatch.setattr("vibe.session_factory._onnx_runtime_capabilities", lambda: (True, False))
    monkeypatch.setattr("vibe.session_factory._pytorch_runtime_capabilities", lambda: (True, False))

    selected = _auto_select_backend(WDEva02Plugin, requested_device="auto")

    assert selected == Backend.ONNX


class _DummyONNXBackend:
    load_calls = 0

    def load(self, weights_path: Path, providers=None, device: str = "cpu", precision: str = "auto") -> None:
        del weights_path, providers, device, precision
        type(self).load_calls += 1

    def run(self, array):
        return array

    def close(self) -> None:
        return None


class _DummyPyTorchBackend:
    load_calls = 0

    def load(self, weights_path: Path, device: str = "cpu", precision: str = "auto") -> None:
        del weights_path, device, precision
        type(self).load_calls += 1

    def run(self, array):
        return array

    def close(self) -> None:
        return None


def _local_file_map(tmp_path: Path) -> FileMap:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"fake")
    return FileMap({"model.safetensors": weights})


def test_build_session_auto_backend_falls_back_when_primary_files_missing(monkeypatch, tmp_path: Path) -> None:
    plugin_cls = vibe.model_registry.get("wd-eva02-large-v3")

    monkeypatch.setattr("vibe.backends.runtime.onnx.ONNXBackend", _DummyONNXBackend)
    monkeypatch.setattr("vibe.backends.runtime.pytorch.PyTorchBackend", _DummyPyTorchBackend)
    monkeypatch.setattr("vibe.session_factory._auto_select_backend", lambda *args, **kwargs: Backend.ONNX)
    monkeypatch.setattr(plugin_cls, "load_ancillary", lambda self, file_map: None)

    _DummyONNXBackend.load_calls = 0
    _DummyPyTorchBackend.load_calls = 0

    def _fake_resolve(source, file_specs, backend, **kwargs):
        del source, file_specs, kwargs
        if backend == Backend.ONNX:
            raise RuntimeError("Required file 'model.onnx' not found")
        return _local_file_map(tmp_path)

    monkeypatch.setattr("vibe.session_factory.resolve_from_sources", _fake_resolve)

    session = build_session(
        plugin_cls=plugin_cls,
        source=f"local:{tmp_path}",
        backend=None,
        auto_download=False,
    )
    try:
        assert session.backend == Backend.PYTORCH
        assert _DummyONNXBackend.load_calls == 0
        assert _DummyPyTorchBackend.load_calls == 1
    finally:
        session.close()


def test_build_session_explicit_backend_does_not_fallback_on_missing_files(monkeypatch, tmp_path: Path) -> None:
    plugin_cls = vibe.model_registry.get("wd-eva02-large-v3")

    monkeypatch.setattr("vibe.backends.runtime.onnx.ONNXBackend", _DummyONNXBackend)
    monkeypatch.setattr("vibe.backends.runtime.pytorch.PyTorchBackend", _DummyPyTorchBackend)
    monkeypatch.setattr(plugin_cls, "load_ancillary", lambda self, file_map: None)

    _DummyONNXBackend.load_calls = 0
    _DummyPyTorchBackend.load_calls = 0

    def _fake_resolve(source, file_specs, backend, **kwargs):
        del source, file_specs, backend, kwargs
        raise RuntimeError("Required file 'model.onnx' not found")

    monkeypatch.setattr("vibe.session_factory.resolve_from_sources", _fake_resolve)

    with pytest.raises(SessionError, match="model.onnx"):
        build_session(
            plugin_cls=plugin_cls,
            source=f"local:{tmp_path}",
            backend=Backend.ONNX,
            auto_download=False,
        )

    assert _DummyONNXBackend.load_calls == 0
    assert _DummyPyTorchBackend.load_calls == 0


def test_build_session_auto_local_checks_all_backends_before_hf_fallback(monkeypatch, tmp_path: Path) -> None:
    plugin_cls = vibe.model_registry.get("wd-eva02-large-v3")

    monkeypatch.setattr("vibe.backends.runtime.onnx.ONNXBackend", _DummyONNXBackend)
    monkeypatch.setattr("vibe.backends.runtime.pytorch.PyTorchBackend", _DummyPyTorchBackend)
    monkeypatch.setattr("vibe.session_factory._auto_select_backend", lambda *args, **kwargs: Backend.ONNX)
    monkeypatch.setattr(plugin_cls, "load_ancillary", lambda self, file_map: None)

    _DummyONNXBackend.load_calls = 0
    _DummyPyTorchBackend.load_calls = 0

    observed_attempts: list[tuple[str, bool, bool]] = []
    local_dir = tmp_path / "model-dir"
    local_dir.mkdir(parents=True, exist_ok=True)

    def _fake_resolve(source, file_specs, backend, **kwargs):
        del file_specs
        observed_attempts.append(
            (
                backend.value,
                bool(kwargs.get("allow_download")),
                kwargs.get("fallback_hf_repo_id") is not None,
            )
        )
        # succeed only on second-phase PyTorch attempt (after local-only checks)
        if (
            source == str(local_dir)
            and backend == Backend.PYTORCH
            and bool(kwargs.get("allow_download"))
            and kwargs.get("fallback_hf_repo_id") is not None
        ):
            return _local_file_map(tmp_path)
        raise RuntimeError("missing files")

    monkeypatch.setattr("vibe.session_factory.resolve_from_sources", _fake_resolve)

    session = build_session(
        plugin_cls=plugin_cls,
        source=str(local_dir),
        backend=None,
        auto_download=True,
    )
    try:
        assert session.backend == Backend.PYTORCH
    finally:
        session.close()

    assert observed_attempts[:2] == [
        ("onnx", False, False),
        ("pytorch", False, False),
    ]
    assert observed_attempts[2:] == [
        ("onnx", True, True),
        ("pytorch", True, True),
    ]


def test_build_session_auto_logs_concise_fallback_and_selection(monkeypatch, tmp_path: Path, caplog) -> None:
    plugin_cls = vibe.model_registry.get("wd-eva02-large-v3")

    monkeypatch.setattr("vibe.backends.runtime.onnx.ONNXBackend", _DummyONNXBackend)
    monkeypatch.setattr("vibe.backends.runtime.pytorch.PyTorchBackend", _DummyPyTorchBackend)
    monkeypatch.setattr("vibe.session_factory._auto_select_backend", lambda *args, **kwargs: Backend.ONNX)
    monkeypatch.setattr(plugin_cls, "load_ancillary", lambda self, file_map: None)

    local_dir = tmp_path / "model-dir"
    local_dir.mkdir(parents=True, exist_ok=True)

    def _fake_resolve(source, file_specs, backend, **kwargs):
        del source, file_specs, kwargs
        if backend == Backend.ONNX:
            raise RuntimeError("Required file 'model.onnx' not found")
        return _local_file_map(tmp_path)

    monkeypatch.setattr("vibe.session_factory.resolve_from_sources", _fake_resolve)

    caplog.set_level("INFO")
    session = build_session(
        plugin_cls=plugin_cls,
        source=str(local_dir),
        backend=None,
        auto_download=False,
    )
    try:
        assert session.backend == Backend.PYTORCH
    finally:
        session.close()

    messages = "\n".join(record.message for record in caplog.records)
    assert "unavailable locally" in messages
    assert "trying pytorch next" in messages.lower()
    assert "Required file 'model.onnx'" not in messages
