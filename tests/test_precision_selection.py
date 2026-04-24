from __future__ import annotations

from pathlib import Path

import pytest

from vibe.backends.base import Backend
from vibe.loader import FileMap
from vibe.precision import normalize_precision_string
from vibe.session import SessionError
from vibe.session_factory import _make_backend_pool_key, build_session


def test_normalize_precision_aliases() -> None:
    assert normalize_precision_string(None) == "auto"
    assert normalize_precision_string("float32") == "fp32"
    assert normalize_precision_string("float16") == "fp16"
    assert normalize_precision_string("bfloat16") == "bf16"
    assert normalize_precision_string("ov") == "int8_ov"


def test_normalize_precision_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        normalize_precision_string("tf32")


def test_backend_pool_key_includes_precision(tmp_path: Path) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake")

    auto_key = _make_backend_pool_key(
        backend=Backend.ONNX,
        weights_path=weights,
        device="cpu",
        providers=None,
        precision="auto",
    )
    fp16_key = _make_backend_pool_key(
        backend=Backend.ONNX,
        weights_path=weights,
        device="cpu",
        providers=None,
        precision="fp16",
    )

    assert auto_key != fp16_key


class _DummyPyTorchBackend:
    load_kwargs = {}

    def load(self, weights_path: Path, device: str = "cpu", precision: str = "auto") -> None:
        del weights_path
        type(self).load_kwargs = {"device": device, "precision": precision}

    def close(self) -> None:
        pass


class _DummyONNXBackend:
    load_kwargs = {}

    def load(self, weights_path: Path, providers=None, device: str = "cpu", precision: str = "auto") -> None:
        del weights_path, providers, device
        type(self).load_kwargs = {"precision": precision}

    def close(self) -> None:
        pass


def test_build_session_rejects_invalid_precision(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    (tmp_path / "selected_tags.csv").write_text(
        "name,category\nblue_hair,0\ncat_ears,0\nmiku_hatsune,4\nsafe,9\n",
        encoding="utf-8",
    )

    plugin_cls = __import__("vibe").model_registry.get("wd-eva02-v3")

    with pytest.raises(SessionError, match="Unsupported precision"):
        build_session(
            plugin_cls=plugin_cls,
            source=f"local:{tmp_path}",
            backend="onnx",
            precision="bogus",
            auto_download=False,
        )


def test_build_session_pytorch_int8_ov_falls_back_to_auto(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("vibe.backends.runtime.pytorch.PyTorchBackend", _DummyPyTorchBackend)

    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"fake")
    (tmp_path / "selected_tags.csv").write_text(
        "name,category\nblue_hair,0\ncat_ears,0\nmiku_hatsune,4\nsafe,9\n",
        encoding="utf-8",
    )

    plugin_cls = __import__("vibe").model_registry.get("wd-eva02-v3")

    session = build_session(
        plugin_cls=plugin_cls,
        source=f"local:{tmp_path}",
        backend="pytorch",
        precision="int8_ov",
        auto_download=False,
    )
    try:
        assert _DummyPyTorchBackend.load_kwargs["precision"] == "auto"
    finally:
        session.close()


def test_build_session_onnx_precision_warning_only_after_success(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr("vibe.backends.runtime.onnx.ONNXBackend", _DummyONNXBackend)
    monkeypatch.setattr("vibe.backends.runtime.pytorch.PyTorchBackend", _DummyPyTorchBackend)
    monkeypatch.setattr("vibe.session_factory._auto_select_backend", lambda *args, **kwargs: Backend.ONNX)

    plugin_cls = __import__("vibe").model_registry.get("wdv4-convnextv2-huge-dbv4-full")
    monkeypatch.setattr(plugin_cls, "load_ancillary", lambda self, file_map: None)

    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake")
    tags = tmp_path / "selected_tags.csv"
    tags.write_text("name,category\nblue_hair,0\n", encoding="utf-8")

    def _fake_resolve(source, file_specs, backend, **kwargs):
        del source, file_specs, backend, kwargs
        return FileMap({"model.onnx": weights, "selected_tags.csv": tags})

    monkeypatch.setattr("vibe.session_factory.resolve_from_source_string", _fake_resolve)

    caplog.set_level("INFO")
    session = build_session(
        plugin_cls=plugin_cls,
        source=f"local:{tmp_path}",
        backend=None,
        precision="bf16",
        auto_download=False,
    )
    try:
        assert session.backend == Backend.ONNX
        assert _DummyONNXBackend.load_kwargs["precision"] == "bf16"
    finally:
        session.close()

    messages = "\n".join(record.message for record in caplog.records)
    assert "Precision 'bf16' requested while running ONNX backend" in messages
