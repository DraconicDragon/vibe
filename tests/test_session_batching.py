from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import vibe
from vibe.backends.base import Backend
from vibe.loader import FileMap
from vibe.result_processors import CharacterIPMapping, CleanTags, ResultProcessor
from vibe.results import TagResult
from vibe.session import InferenceCancelled, ModelSession

# Optional dev-configurable real-world smoke test variables.
RUN_REAL_WORLD_SMOKE_TEST = os.getenv("VIBE_REAL_WORLD_TEST", "0") == "1"
REAL_WORLD_MODEL_SOURCE = os.getenv(
    "VIBE_REAL_WORLD_MODEL_SOURCE",
    "/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3",
)
REAL_WORLD_IMAGE_PATHS: list[str] = [
    p.strip()
    for p in os.getenv(
        "VIBE_REAL_WORLD_IMAGE_PATHS",
        "/home/drac/Desktop/hakurei.jpg,/home/drac/Desktop/gogalking.jpg,/home/drac/Desktop/fern_and_frieren_sleep_help.webp",
    ).split(",")
    if p.strip()
]


class _DummyBackend:
    def __init__(self, providers: list[str] | None = None):
        self.providers = providers or ["CPUExecutionProvider"]
        self.calls: list[int] = []

    def run(self, array: np.ndarray) -> np.ndarray:
        self.calls.append(int(array.shape[0]))
        # [blue_hair, cat_ears, miku_hatsune, safe, ^_^]
        if array.shape[0] == 1:
            return np.array([[0.1, 0.2, 0.9, 0.3, 0.05]], dtype=np.float32)
        return np.array(
            [
                [0.1, 0.2, 0.9, 0.3, 0.05],
                [0.7, 0.2, 0.1, 0.3, 0.05],
            ],
            dtype=np.float32,
        )


def _write_selected_tags_csv(path: Path) -> None:
    path.write_text(
        "name,category\nblue_hair,0\ncat_ears,0\nmiku_hatsune,4\nsafe,9\n^_^,4\n",
        encoding="utf-8",
    )


def _build_session(
    tmp_path: Path,
    providers: list[str] | None = None,
) -> tuple[ModelSession, _DummyBackend]:
    plugin = vibe.registry.get("wd-eva02-large")()
    tags = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(tags)
    plugin.configure(auto_download=False)
    plugin.load_ancillary({"selected_tags.csv": tags})

    backend = _DummyBackend(providers=providers)
    session = ModelSession(
        plugin=plugin,
        backend_instance=backend,
        backend=Backend.ONNX,
        file_map=FileMap({"selected_tags.csv": tags}),
        source=f"local:{tmp_path}",
        auto_download=False,
    )
    return session, backend


def _test_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (16, 16), (255, 0, 0)),
        Image.new("RGB", (16, 16), (0, 255, 0)),
    ]


def _write_test_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (16, 16), color).save(path)


def test_infer_accepts_single_path_and_auto_ref_index(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    img_path = tmp_path / "one.png"
    _write_test_image(img_path, (255, 0, 0))

    output = session.infer(str(img_path))

    assert output.total_inputs == 1
    assert len(output.items) == 1
    assert output.items[0].index == 0
    assert output.items[0].input_ref == 0
    assert isinstance(output.first(), TagResult)


def test_infer_accepts_list_of_paths_with_positional_refs(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    _write_test_image(first, (255, 0, 0))
    _write_test_image(second, (0, 255, 0))

    output = session.infer([str(first), str(second)])

    assert output.total_inputs == 2
    assert [item.index for item in output.items] == [0, 1]
    assert [item.input_ref for item in output.items] == [0, 1]


def test_infer_accepts_explicit_tuple_refs(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    _write_test_image(first, (255, 0, 0))
    _write_test_image(second, (0, 255, 0))

    output = session.infer([(str(first), "img-a"), (str(second), "img-b")])

    assert output.total_inputs == 2
    assert [item.index for item in output.items] == [0, 1]
    assert [item.input_ref for item in output.items] == ["img-a", "img-b"]


def test_infer_rejects_mixed_tuple_and_non_tuple_inputs(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)

    with pytest.raises(Exception, match="Mixed input formats"):
        session.infer([_test_images()[0], (_test_images()[1], "img-b")])


def test_infer_rejects_duplicate_explicit_refs(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)

    with pytest.raises(Exception, match="Duplicate refs"):
        session.infer([(_test_images()[0], "dup"), (_test_images()[1], "dup")])


def test_infer_allows_duplicate_path_inputs(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    img_path = tmp_path / "dup.png"
    _write_test_image(img_path, (255, 0, 0))

    output = session.infer([str(img_path), str(img_path)])

    assert output.total_inputs == 2


def test_infer_many_auto_uses_sequential_on_cpu(tmp_path: Path) -> None:
    session, backend = _build_session(tmp_path, providers=["CPUExecutionProvider"])
    results = session.infer(_test_images(), batch_size=2, batch_method="auto")

    assert len(results) == 2
    assert all(isinstance(item.result, TagResult) for item in results)
    assert backend.calls == [1, 1]


def test_infer_many_auto_uses_true_batch_on_gpu_provider(tmp_path: Path) -> None:
    session, backend = _build_session(
        tmp_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    results = session.infer(_test_images(), batch_size=2, batch_method="auto")

    assert len(results) == 2
    assert backend.calls == [2]


def test_infer_many_auto_uses_true_batch_on_any_non_cpu_provider(tmp_path: Path) -> None:
    session, backend = _build_session(
        tmp_path,
        providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )
    results = session.infer(_test_images(), batch_size=2, batch_method="auto")

    assert len(results) == 2
    assert backend.calls == [2]


def test_infer_many_override_forces_sequential(tmp_path: Path) -> None:
    session, backend = _build_session(
        tmp_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    results = session.infer(_test_images(), batch_size=2, batch_method="sequential")

    assert len(results) == 2
    assert backend.calls == [1, 1]


def test_infer_batches_auto_streams_true_batch_chunks(tmp_path: Path) -> None:
    session, backend = _build_session(
        tmp_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    chunks = list(session.infer_batches(_test_images(), batch_size=2, batch_method="auto"))

    assert len(chunks) == 1
    assert chunks[0].total_inputs == 2
    assert [item.index for item in chunks[0].items] == [0, 1]
    assert backend.calls == [2]


def test_infer_batches_auto_streams_sequential_items_on_cpu(tmp_path: Path) -> None:
    session, backend = _build_session(tmp_path, providers=["CPUExecutionProvider"])

    chunks = list(session.infer_batches(_test_images(), batch_size=2, batch_method="auto"))

    assert len(chunks) == 2
    assert all(len(chunk.items) == 1 for chunk in chunks)
    assert [chunk.items[0].index for chunk in chunks] == [0, 1]
    assert backend.calls == [1, 1]


def test_infer_async_streams_chunks(tmp_path: Path) -> None:
    session, backend = _build_session(
        tmp_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    async def _collect() -> list[list[int]]:
        collected: list[list[int]] = []
        async for chunk in session.infer_async(_test_images(), batch_size=2, batch_method="auto"):
            collected.append([item.index for item in chunk.items])
        return collected

    chunk_indices = asyncio.run(_collect())

    assert chunk_indices == [[0, 1]]
    assert backend.calls == [2]


def test_infer_async_accepts_single_input_image(tmp_path: Path) -> None:
    session, backend = _build_session(tmp_path)

    async def _collect() -> list[list[int]]:
        collected: list[list[int]] = []
        async for chunk in session.infer_async(_test_images()[0], batch_size=2, batch_method="auto"):
            collected.append([item.index for item in chunk.items])
        return collected

    chunk_indices = asyncio.run(_collect())

    assert chunk_indices == [[0]]
    assert backend.calls == [1]


def test_cancel_current_inference_returns_false_when_idle(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    assert session.cancel_current_inference() is False


def test_infer_async_can_be_cancelled_cooperatively(tmp_path: Path) -> None:
    session, backend = _build_session(tmp_path, providers=["CPUExecutionProvider"])
    images = [
        Image.new("RGB", (16, 16), (255, 0, 0)),
        Image.new("RGB", (16, 16), (0, 255, 0)),
        Image.new("RGB", (16, 16), (0, 0, 255)),
    ]

    async def _consume_and_cancel() -> None:
        got_first = False
        async for chunk in session.infer_async(images, batch_size=2, batch_method="sequential"):
            if not got_first:
                got_first = True
                assert len(chunk.items) == 1
                assert session.cancel_current_inference() is True

    with pytest.raises(InferenceCancelled, match="cancelled"):
        asyncio.run(_consume_and_cancel())

    assert 1 <= len(backend.calls) < len(images)
    assert all(call == 1 for call in backend.calls)
    assert session.is_inference_running() is False
    assert session.is_cancellation_requested() is False


def test_infer_sync_can_return_partial_results_on_cancel(tmp_path: Path) -> None:
    session, backend = _build_session(tmp_path, providers=["CPUExecutionProvider"])
    images = [
        Image.new("RGB", (16, 16), (255, 0, 0)),
        Image.new("RGB", (16, 16), (0, 255, 0)),
        Image.new("RGB", (16, 16), (0, 0, 255)),
    ]

    original_run = backend.run

    def _slow_run(array: np.ndarray) -> np.ndarray:
        time.sleep(0.03)
        return original_run(array)

    backend.run = _slow_run  # type: ignore[method-assign]

    def _cancel_when_running() -> None:
        while not backend.calls:
            time.sleep(0.005)
        session.cancel_current_inference()

    canceller = threading.Thread(target=_cancel_when_running, daemon=True)
    canceller.start()
    result = session.infer(
        images,
        batch_size=2,
        batch_method="sequential",
        on_cancel="return_partial",
    )
    canceller.join(timeout=0.2)

    assert 1 <= len(result.items) < len(images)
    assert session.is_inference_running() is False
    assert session.is_cancellation_requested() is False


def test_infer_many_sequential_runs_for_all_inputs(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path, providers=["CPUExecutionProvider"])
    results = session.infer(
        _test_images(),
        batch_size=2,
        batch_method="sequential",
    )

    assert len(results) == 2


def test_true_batch_preprocesses_per_chunk_not_whole_input(tmp_path: Path) -> None:
    # This ensures first chunk can run before a later image fails preprocess.
    session, backend = _build_session(
        tmp_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    original_preprocess = session.plugin.preprocess
    calls = 0

    def _flaky_preprocess(image: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("boom")
        return original_preprocess(image)

    session.plugin.preprocess = _flaky_preprocess  # type: ignore[method-assign]

    images = _test_images() + [Image.new("RGB", (16, 16), (0, 0, 255))]
    with pytest.raises(Exception):
        session.infer(images, batch_size=2, batch_method="true")

    assert backend.calls == [2]


def test_result_pipeline_applies_character_mapping_from_csv(tmp_path: Path) -> None:
    mapping_path = tmp_path / "char_ip_map.csv"
    mapping_path.write_text(
        'name,ips\nmiku_hatsune,"[""vocaloid"", ""crypton""]"\n',
        encoding="utf-8",
    )

    session, _ = _build_session(tmp_path)
    result = (
        session.infer(
            _test_images()[0],
            processors=[CharacterIPMapping()],
        )
        .items[0]
        .result
    )

    assert isinstance(result, TagResult)
    assert result.character_mapping == {"miku_hatsune": ["vocaloid", "crypton"]}


def test_result_pipeline_applies_character_mapping_from_manual_json_path(tmp_path: Path) -> None:
    manual_mapping = tmp_path / "manual_mapping.json"
    manual_mapping.write_text(
        '{"mapping": {"miku_hatsune": ["vocaloid", "crypton"]}}',
        encoding="utf-8",
    )

    session, _ = _build_session(tmp_path)
    result = (
        session.infer(
            _test_images()[0],
            processors=[CharacterIPMapping(mapping_file=manual_mapping)],
        )
        .items[0]
        .result
    )

    assert isinstance(result, TagResult)
    assert result.character_mapping == {"miku_hatsune": ["vocaloid", "crypton"]}


def test_result_pipeline_optional_cleaning_preserves_kaomojis(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    result = (
        session.infer(
            _test_images()[0],
            processors=[CleanTags()],
        )
        .items[0]
        .result
    )

    assert isinstance(result, TagResult)
    names = result.tag_names()
    assert "blue hair" in names
    assert "miku hatsune" in names
    assert "^_^" in names
    assert any(entry.tag == "cat ears" for entry in result.category("general"))


def test_as_score_dict_is_sorted_descending_by_default(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    result = session.infer(_test_images()[0]).items[0].result

    scores = list(result.as_score_dict().values())
    assert scores == sorted(scores, reverse=True)


def test_result_pipeline_mapping_and_cleaning_can_run_together(tmp_path: Path) -> None:
    manual_mapping = tmp_path / "manual_mapping.json"
    manual_mapping.write_text(
        '{"mapping": {"miku_hatsune": ["vocaloid_ip", "crypton"]}}',
        encoding="utf-8",
    )

    session, _ = _build_session(tmp_path)
    result = (
        session.infer(
            _test_images()[0],
            processors=[CharacterIPMapping(mapping_file=manual_mapping), CleanTags()],
        )
        .items[0]
        .result
    )

    assert isinstance(result, TagResult)
    assert result.character_mapping == {"miku hatsune": ["vocaloid ip", "crypton"]}


def test_unsupported_processor_logs_warning_but_still_applies(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class _UnsupportedMarker(ResultProcessor):
        pass

    class _NoOpProcessor(_UnsupportedMarker):
        def process(self, result, *, context):
            del context
            return result

    session, _ = _build_session(tmp_path)

    with caplog.at_level("WARNING"):
        result = session.infer(_test_images()[0], processors=[_NoOpProcessor()]).items[0].result

    assert isinstance(result, TagResult)
    assert "not declared as supported" in caplog.text


def test_session_close_prevents_new_inference(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    session.close()

    with pytest.raises(Exception):
        session.infer(_test_images()[0])


def test_session_memory_stats_tracking_controls(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)

    # Default tracking should record one call.
    session.infer(_test_images()[0])
    stats = session.memory_stats()
    assert stats["inference_calls"] == 1

    # Disabling tracking should freeze the call counter.
    session.set_memory_tracking(False)
    session.infer(_test_images()[0])
    stats_after_disable = session.memory_stats()
    assert stats_after_disable["inference_calls"] == 1

    # Reset should clear aggregated stats.
    session.reset_memory_stats()
    reset_stats = session.memory_stats()
    assert reset_stats["inference_calls"] == 0


@pytest.mark.skipif(
    not RUN_REAL_WORLD_SMOKE_TEST,
    reason="Set VIBE_REAL_WORLD_TEST=1 and configure VIBE_REAL_WORLD_* variables to run.",
)
def test_real_world_smoke_batch_modes() -> None:
    assert REAL_WORLD_MODEL_SOURCE, "REAL_WORLD_MODEL_SOURCE must be set"
    assert REAL_WORLD_IMAGE_PATHS, "REAL_WORLD_IMAGE_PATHS must contain at least one image path"

    images = [Image.open(path).convert("RGB") for path in REAL_WORLD_IMAGE_PATHS]

    session = vibe.load(
        "wd-eva02-large",
        source=REAL_WORLD_MODEL_SOURCE,
        backend="onnx",
        auto_download=False,
    )

    # One-by-one method
    sequential_results = session.infer(images, batch_size=1, batch_method="sequential")
    assert len(sequential_results) == len(images)

    # True batching method
    true_batch_results = session.infer(
        images,
        batch_size=max(2, len(images)),
        batch_method="true",
    )
    assert len(true_batch_results) == len(images)
