from __future__ import annotations

from vibe.backends.base import Backend
from vibe.plugins.wd_tagger import WDEva02Plugin
from vibe.session_factory import _auto_select_backend


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
