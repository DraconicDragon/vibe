from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import vibe
from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    ExecutionPlan,
    ExecutionPreference,
    FileRole,
    HardwareIntent,
    RuntimeExecutor,
)
from vibe.exceptions import InferenceCancelled, SessionCapabilityError
from vibe.precision import PrecisionPolicy, PrecisionRequest
from vibe.results import TagResult
from vibe.session import ModelSession
from vibe.session_factory import _acquire_runtime


class _MockSessionBackend(RuntimeExecutor):
    def __init__(self) -> None:
        self.run_calls: list[int] = []
        self.is_closed = False

    def run(self, inputs: Any) -> Any:
        # Record input batch size
        count = len(inputs) if hasattr(inputs, "__len__") else 1
        self.run_calls.append(count)
        return inputs

    def close(self) -> None:
        self.is_closed = True

    def supports_true_batching(self) -> bool:
        return True

    def execution_info(self) -> dict[str, Any]:
        return {"device": "cpu", "provider": "mock"}


def _build_test_session(tmp_path: Path) -> tuple[ModelSession, _MockSessionBackend]:
    plugin_cls = vibe.model_registry.get("test-dummy-tagger")
    plugin = plugin_cls()
    dummy_file = tmp_path / "model.onnx"
    dummy_file.write_bytes(b"dummy")

    artifacts = ArtifactMap(
        paths_by_id={"dummy_weights": dummy_file},
        specs_by_id={"dummy_weights": ArtifactSpec(id="dummy_weights", name="model.onnx", role=FileRole.WEIGHTS)},
    )
    plugin.load_ancillary(artifacts)

    backend = _MockSessionBackend()
    plan = ExecutionPlan(
        backend=Backend.ONNX,
        preference=ExecutionPreference(intent=HardwareIntent.CPU),
        precision=PrecisionRequest(weight=PrecisionPolicy.PRESERVE, compute=PrecisionPolicy.AUTO),
    )
    plugin.bind_execution_state(plan)

    session = ModelSession(
        plugin=plugin,
        backend_instance=backend,
        plan=plan,
        file_map=artifacts,
        source=f"local:{tmp_path}",
        memory_tracking=True,
    )
    return session, backend


def test_session_infer_single_and_batch(tmp_path: Path) -> None:
    session, backend = _build_test_session(tmp_path)

    img1 = Image.new("RGB", (16, 16), (255, 0, 0))
    img2 = Image.new("RGB", (16, 16), (0, 255, 0))

    # Single inference
    res_single = session.infer(img1)
    assert res_single.total_inputs == 1
    assert isinstance(res_single.first(), TagResult)

    # Batch inference with custom dict mapping (preserves keys as refs)
    res_batch = session.infer({"a": img1, "b": img2}, batch_size=2)
    assert res_batch.total_inputs == 2
    assert [item.input_ref for item in res_batch.items] == ["a", "b"]
    assert len(backend.run_calls) >= 2


def test_session_infer_async_streaming(tmp_path: Path) -> None:
    session, _ = _build_test_session(tmp_path)
    images = [Image.new("RGB", (16, 16), (i, i, i)) for i in range(3)]

    async def _run() -> list[list[int]]:
        chunks = []
        async for chunk in session.infer_async(images, batch_size=2):
            chunks.append([item.index for item in chunk.items])
        return chunks

    result_chunks = asyncio.run(_run())
    assert result_chunks == [[0, 1], [2]]


def test_session_infer_cancellation(tmp_path: Path) -> None:
    session, backend = _build_test_session(tmp_path)
    images = [Image.new("RGB", (16, 16), (i, i, i)) for i in range(5)]

    # Add a tiny delay so the background worker is actively in-flight during cancellation
    original_run = backend.run

    def _slow_run(inputs: Any) -> Any:
        import time

        time.sleep(0.02)
        return original_run(inputs)

    backend.run = _slow_run  # type: ignore[method-assign]

    async def _cancel_run() -> None:
        async for chunk in session.infer_async(images, batch_size=1):
            if chunk.items[0].index == 1:
                session.cancel_current_inference()

    with pytest.raises(InferenceCancelled, match="cancelled"):
        asyncio.run(_cancel_run())


def test_session_capability_view_access(tmp_path: Path) -> None:
    session, _ = _build_test_session(tmp_path)

    assert session.is_tagger is True
    assert session.is_scorer is False

    # 1. Tagger view is available and typed
    tagger_view = session.tagger
    assert tagger_view.catalog.names == ("1girl", "solo", "hatsune_miku")
    assert tagger_view.thresholds.values["1girl"] == 0.4
    assert tagger_view.recommendation.global_threshold == 0.35

    # 2. as_tagger() returns the view; as_scorer() safely returns None
    assert session.as_tagger() is not None
    assert session.as_scorer() is None

    # 3. Accessing .scorer property on a tagger raises SessionCapabilityError
    with pytest.raises(SessionCapabilityError, match="is not a scorer"):
        _ = session.scorer

    # 4. inspect_data() exports clean serializable dictionary
    manifest = session.inspect_data()
    assert "catalog" in manifest
    assert manifest["catalog"]["labels"][0]["name"] == "1girl"


def test_session_memory_stats(tmp_path: Path) -> None:
    session, _ = _build_test_session(tmp_path)
    img = Image.new("RGB", (16, 16), (0, 0, 0))

    session.infer(img)
    stats = session.memory_stats()
    assert stats["inference_calls"] == 1

    session.reset_memory_stats()
    assert session.memory_stats()["inference_calls"] == 0


def test_runtime_pooling_refcount() -> None:
    call_count = 0
    close_count = 0

    class DummyResource:
        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    def builder() -> DummyResource:
        nonlocal call_count
        call_count += 1
        return DummyResource()

    key = ("mock_pool_key", 1)

    # 1. First acquisition builds instance
    res1, release1 = _acquire_runtime(key=key, model_id="dummy", build=builder)
    assert call_count == 1

    # 2. Second acquisition reuses instance
    res2, release2 = _acquire_runtime(key=key, model_id="dummy", build=builder)
    assert call_count == 1
    assert res1 is res2

    # 3. First release decrements refcount, does not close
    release1()
    assert close_count == 0

    # 4. Final release closes resource
    release2()
    assert close_count == 1
