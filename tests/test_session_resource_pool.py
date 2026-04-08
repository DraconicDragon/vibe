from __future__ import annotations

from pathlib import Path

import pytest

import vibe
from vibe.session import SessionError
from vibe.session_factory import build_session


class _DummyONNXBackend:
    load_calls = 0
    close_calls = 0

    def __init__(self) -> None:
        self.providers: list[str] = []

    def load(
        self,
        weights_path: Path,
        providers: list[str] | None = None,
        device: str = "cpu",
        precision: str = "auto",
    ) -> None:
        del weights_path, device, precision
        self.providers = providers or ["CPUExecutionProvider"]
        type(self).load_calls += 1

    def run(self, array):
        return array

    def close(self) -> None:
        type(self).close_calls += 1


def _write_selected_tags_csv(path: Path) -> None:
    path.write_text(
        "name,category\nblue_hair,0\ncat_ears,0\nmiku_hatsune,4\nsafe,9\n",
        encoding="utf-8",
    )


def test_build_session_reuses_pooled_backend_until_last_close(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vibe.backends.runtime.onnx.ONNXBackend", _DummyONNXBackend)

    _DummyONNXBackend.load_calls = 0
    _DummyONNXBackend.close_calls = 0

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    _write_selected_tags_csv(tmp_path / "selected_tags.csv")

    plugin_cls = vibe.registry.get("wd-eva02-v3")
    source = f"local:{tmp_path}"

    session1 = build_session(
        plugin_cls=plugin_cls,
        source=source,
        backend="onnx",
        auto_download=False,
    )
    session2 = build_session(
        plugin_cls=plugin_cls,
        source=source,
        backend="onnx",
        auto_download=False,
    )

    assert _DummyONNXBackend.load_calls == 1

    session1.close()
    assert _DummyONNXBackend.close_calls == 0

    session2.close()
    assert _DummyONNXBackend.close_calls == 1


def test_build_session_releases_pooled_backend_on_ancillary_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vibe.backends.runtime.onnx.ONNXBackend", _DummyONNXBackend)

    _DummyONNXBackend.load_calls = 0
    _DummyONNXBackend.close_calls = 0

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    _write_selected_tags_csv(tmp_path / "selected_tags.csv")

    plugin_cls = vibe.registry.get("wd-eva02-v3")
    source = f"local:{tmp_path}"

    def _failing_load_ancillary(self, file_map):
        del self, file_map
        raise RuntimeError("ancillary boom")

    monkeypatch.setattr(plugin_cls, "load_ancillary", _failing_load_ancillary)

    with pytest.raises(SessionError, match="failed to load ancillary files"):
        build_session(
            plugin_cls=plugin_cls,
            source=source,
            backend="onnx",
            auto_download=False,
        )

    assert _DummyONNXBackend.load_calls == 1
    assert _DummyONNXBackend.close_calls == 1
