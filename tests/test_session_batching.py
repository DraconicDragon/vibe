from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import autotagger
from autotagger.backends.base import Backend
from autotagger.loader import FileMap, ModelSource
from autotagger.results import TagResult
from autotagger.session import ModelSession

# Optional dev-configurable real-world smoke test variables.
RUN_REAL_WORLD_SMOKE_TEST = os.getenv("AUTOTAGGER_REAL_WORLD_TEST", "0") == "1"
REAL_WORLD_MODEL_SOURCE = os.getenv(
    "AUTOTAGGER_REAL_WORLD_MODEL_SOURCE",
    "/mnt/T7/Projects/GitHub/vibe/models/wd-eva02-large-tagger-v3",
)
REAL_WORLD_IMAGE_PATHS: list[str] = [
    p.strip()
    for p in os.getenv(
        "AUTOTAGGER_REAL_WORLD_IMAGE_PATHS",
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
        "name,category\n" "blue_hair,0\n" "cat_ears,0\n" "miku_hatsune,4\n" "safe,9\n" "^_^,4\n",
        encoding="utf-8",
    )


def _build_session(
    tmp_path: Path,
    providers: list[str] | None = None,
    *,
    character_mapping_path: str | None = None,
) -> tuple[ModelSession, _DummyBackend]:
    plugin = autotagger.registry.get("wd-eva02-large")()
    tags = tmp_path / "selected_tags.csv"
    _write_selected_tags_csv(tags)
    plugin.configure(auto_download=False, character_mapping_path=character_mapping_path)
    plugin.load_ancillary({"selected_tags.csv": tags})

    backend = _DummyBackend(providers=providers)
    session = ModelSession(
        plugin=plugin,
        backend_instance=backend,
        backend=Backend.ONNX,
        file_map=FileMap({"selected_tags.csv": tags}),
        source=ModelSource.local(tmp_path),
        auto_download=False,
        character_mapping_path=character_mapping_path,
    )
    return session, backend


def _test_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (16, 16), (255, 0, 0)),
        Image.new("RGB", (16, 16), (0, 255, 0)),
    ]


def test_infer_many_auto_uses_sequential_on_cpu(tmp_path: Path) -> None:
    session, backend = _build_session(tmp_path, providers=["CPUExecutionProvider"])
    results = session.infer_many(_test_images(), batch_size=2, batch_method="auto")

    assert len(results) == 2
    assert all(isinstance(r, TagResult) for r in results)
    assert backend.calls == [1, 1]


def test_infer_many_auto_uses_true_batch_on_gpu_provider(tmp_path: Path) -> None:
    session, backend = _build_session(
        tmp_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    results = session.infer_many(_test_images(), batch_size=2, batch_method="auto")

    assert len(results) == 2
    assert backend.calls == [2]


def test_infer_many_auto_uses_true_batch_on_any_non_cpu_provider(tmp_path: Path) -> None:
    session, backend = _build_session(
        tmp_path,
        providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )
    results = session.infer_many(_test_images(), batch_size=2, batch_method="auto")

    assert len(results) == 2
    assert backend.calls == [2]


def test_infer_many_override_forces_sequential(tmp_path: Path) -> None:
    session, backend = _build_session(
        tmp_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    results = session.infer_many(_test_images(), batch_size=2, batch_method="sequential")

    assert len(results) == 2
    assert backend.calls == [1, 1]


def test_infer_many_sequential_validates_params_once(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path, providers=["CPUExecutionProvider"])

    validate_calls = 0
    original_validate = session.plugin.param_schema.validate

    def _counting_validate(user_params: dict[str, object]) -> dict[str, object]:
        nonlocal validate_calls
        validate_calls += 1
        return original_validate(user_params)

    session.plugin.param_schema.validate = _counting_validate  # type: ignore[method-assign]

    results = session.infer_many(
        _test_images(),
        batch_size=2,
        batch_method="sequential",
        params={
            "general_threshold": 0.5,
            "character_threshold": 0.8,
            "return_all_scores": False,
            "return_character_mapping": False,
            "clean_tags": False,
        },
    )

    assert len(results) == 2
    assert validate_calls == 1


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
        session.infer_many(images, batch_size=2, batch_method="true")

    assert backend.calls == [2]


def test_result_pipeline_applies_character_mapping_from_csv(tmp_path: Path) -> None:
    mapping_path = tmp_path / "char_ip_map.csv"
    mapping_path.write_text(
        "name,ips\n" 'miku_hatsune,"[""vocaloid"", ""crypton""]"\n',
        encoding="utf-8",
    )

    session, _ = _build_session(tmp_path)
    result = session.infer(
        _test_images()[0],
        params={
            "general_threshold": 0.5,
            "character_threshold": 0.8,
            "return_all_scores": False,
            "return_character_mapping": True,
        },
    )

    assert isinstance(result, TagResult)
    assert result.character_mapping == {"miku_hatsune": ["vocaloid", "crypton"]}


def test_result_pipeline_applies_character_mapping_from_manual_json_path(tmp_path: Path) -> None:
    manual_mapping = tmp_path / "manual_mapping.json"
    manual_mapping.write_text(
        '{"mapping": {"miku_hatsune": ["vocaloid", "crypton"]}}',
        encoding="utf-8",
    )

    session, _ = _build_session(
        tmp_path,
        character_mapping_path=str(manual_mapping),
    )
    result = session.infer(
        _test_images()[0],
        params={
            "general_threshold": 0.5,
            "character_threshold": 0.8,
            "return_all_scores": False,
            "return_character_mapping": True,
        },
    )

    assert isinstance(result, TagResult)
    assert result.character_mapping == {"miku_hatsune": ["vocaloid", "crypton"]}


def test_result_pipeline_optional_cleaning_preserves_kaomojis(tmp_path: Path) -> None:
    session, _ = _build_session(tmp_path)
    result = session.infer(
        _test_images()[0],
        params={
            "general_threshold": 0.0,
            "character_threshold": 0.0,
            "return_all_scores": True,
            "return_character_mapping": False,
            "clean_tags": True,
        },
    )

    assert isinstance(result, TagResult)
    names = result.tag_names()
    assert "blue hair" in names
    assert "miku hatsune" in names
    assert "^_^" in names
    assert result.all_scores is not None
    assert any(entry.tag == "cat ears" for entry in result.all_scores)


def test_result_pipeline_mapping_and_cleaning_can_run_together(tmp_path: Path) -> None:
    manual_mapping = tmp_path / "manual_mapping.json"
    manual_mapping.write_text(
        '{"mapping": {"miku_hatsune": ["vocaloid_ip", "crypton"]}}',
        encoding="utf-8",
    )

    session, _ = _build_session(
        tmp_path,
        character_mapping_path=str(manual_mapping),
    )
    result = session.infer(
        _test_images()[0],
        params={
            "general_threshold": 0.5,
            "character_threshold": 0.8,
            "return_all_scores": False,
            "return_character_mapping": True,
            "clean_tags": True,
        },
    )

    assert isinstance(result, TagResult)
    assert result.character_mapping == {"miku hatsune": ["vocaloid ip", "crypton"]}


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
    reason="Set AUTOTAGGER_REAL_WORLD_TEST=1 and configure AUTOTAGGER_REAL_WORLD_* variables to run.",
)
def test_real_world_smoke_batch_modes() -> None:
    assert REAL_WORLD_MODEL_SOURCE, "REAL_WORLD_MODEL_SOURCE must be set"
    assert REAL_WORLD_IMAGE_PATHS, "REAL_WORLD_IMAGE_PATHS must contain at least one image path"

    images = [Image.open(path).convert("RGB") for path in REAL_WORLD_IMAGE_PATHS]

    session = autotagger.load(
        "wd-eva02-large",
        source=REAL_WORLD_MODEL_SOURCE,
        backend="onnx",
        auto_download=False,
    )

    # One-by-one method
    sequential_results = session.infer_many(images, batch_size=1, batch_method="sequential")
    assert len(sequential_results) == len(images)

    # True batching method
    true_batch_results = session.infer_many(
        images,
        batch_size=max(2, len(images)),
        batch_method="true",
    )
    assert len(true_batch_results) == len(images)
