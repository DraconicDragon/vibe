from __future__ import annotations

from pathlib import Path

import pytest

from vibe.backends.base import Backend
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


def test_build_session_rejects_invalid_precision(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    (tmp_path / "selected_tags.csv").write_text(
        "name,category\nblue_hair,0\ncat_ears,0\nmiku_hatsune,4\nsafe,9\n",
        encoding="utf-8",
    )

    plugin_cls = __import__("vibe").registry.get("wd-eva02-v3")

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

    plugin_cls = __import__("vibe").registry.get("wd-eva02-v3")

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
