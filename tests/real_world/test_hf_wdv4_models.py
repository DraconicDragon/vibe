from __future__ import annotations

import os
from importlib.util import find_spec

import numpy as np
import pytest
from PIL import Image

import vibe
from tests.common.animetimm_selected_repos import ANIMETIMM_V4_SELECTED_REPOS
from vibe.result_processors import TagLevelThresholds
from vibe.results import TagResult

RUN_REAL_WORLD_HF = os.getenv("VIBE_REAL_WORLD_WDV4_HF", "0") == "1"
RUN_REAL_WORLD_THRESHOLDS = os.getenv("VIBE_REAL_WORLD_WDV4_TAG_LEVEL_THRESHOLDS", "0") == "1"


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _repo_to_model_id(repo_id: str) -> str:
    suffix = repo_id.split("/", 1)[-1]
    return f"wdv4-{suffix.replace('_', '-').replace('.', '-')}"


def _backend_available(backend: str) -> bool:
    if backend == "onnx":
        return find_spec("onnxruntime") is not None
    if backend == "pytorch":
        return find_spec("torch") is not None
    return False


def _pytorch_timm_available() -> bool:
    return find_spec("torch") is not None and find_spec("timm") is not None


def _selected_repos() -> list[str]:
    explicit = _env_list("VIBE_REAL_WORLD_WDV4_MODELS")
    repos = explicit or list(ANIMETIMM_V4_SELECTED_REPOS)

    max_models_raw = os.getenv("VIBE_REAL_WORLD_WDV4_MAX_MODELS", "").strip()
    if not max_models_raw:
        return repos

    try:
        max_models = int(max_models_raw)
    except ValueError:
        return repos

    if max_models <= 0:
        return repos
    return repos[:max_models]


@pytest.mark.skipif(
    not RUN_REAL_WORLD_HF,
    reason=("Set VIBE_REAL_WORLD_WDV4_HF=1 to run HF real-world smoke tests for AnimeTimm v4 models."),
)
def test_hf_real_world_wdv4_model_inference_smoke() -> None:
    backend = os.getenv("VIBE_REAL_WORLD_WDV4_BACKEND", "onnx").strip().lower() or "onnx"
    if backend not in {"onnx", "pytorch"}:
        pytest.skip("VIBE_REAL_WORLD_WDV4_BACKEND must be 'onnx' or 'pytorch'.")
    if not _backend_available(backend):
        pytest.skip(f"Backend runtime '{backend}' is not installed in this environment.")

    repos = _selected_repos()
    assert repos, "No model repos selected for real-world test run."

    image = Image.new("RGB", (768, 512), (160, 120, 210))

    for repo_id in repos:
        model_id = _repo_to_model_id(repo_id)

        with vibe.load(
            model_id,
            source=f"hf:{repo_id}",
            backend=backend,
            auto_download=True,
        ) as session:
            unfiltered = session.infer(image).first()

            if RUN_REAL_WORLD_THRESHOLDS:
                filtered = session.infer(image, result_processors=[TagLevelThresholds()]).first()
            else:
                filtered = unfiltered

        assert isinstance(unfiltered, TagResult), f"Model '{repo_id}' returned unexpected result type."
        categories = unfiltered.categories()
        assert categories, f"Model '{repo_id}' returned empty category mapping."
        assert any(len(entries) > 0 for entries in categories.values()), (
            f"Model '{repo_id}' returned category mapping with no entries."
        )

        if RUN_REAL_WORLD_THRESHOLDS:
            unfiltered_count = sum(len(entries) for entries in unfiltered.categories().values())
            filtered_count = sum(len(entries) for entries in filtered.categories().values())
            assert filtered_count <= unfiltered_count, (
                f"Tag-level-thresholds increased output count for model '{repo_id}', which should not happen."
            )


@pytest.mark.skipif(
    not RUN_REAL_WORLD_HF,
    reason=("Set VIBE_REAL_WORLD_WDV4_HF=1 to run HF real-world config/fallback parity tests."),
)
def test_hf_real_world_wdv4_config_vs_fallback_parity() -> None:
    if not _pytorch_timm_available():
        pytest.skip("PyTorch + timm is required for config/fallback parity test.")

    repo_id = os.getenv("VIBE_REAL_WORLD_WDV4_PARITY_REPO", "animetimm/caformer_b36.dbv4-full").strip()
    model_id = _repo_to_model_id(repo_id)
    image = Image.new("RGB", (896, 640), (120, 170, 220))

    with vibe.load(
        model_id,
        source=f"hf:{repo_id}",
        backend="pytorch",
        auto_download=True,
    ) as config_session:
        config_result = config_session.infer(image).first()

    with vibe.load(
        model_id,
        source=f"hf:{repo_id}",
        backend="pytorch",
        auto_download=True,
        file_name_map={"config.json": "_forced_missing_config_for_fallback_.json"},
    ) as fallback_session:
        fallback_result = fallback_session.infer(image).first()

    config_scores = config_result.as_score_dict()
    fallback_scores = fallback_result.as_score_dict()

    common_tags = [tag for tag in config_scores if tag in fallback_scores]
    assert common_tags, "No overlapping tags between config and fallback outputs."

    diffs = np.array([abs(config_scores[tag] - fallback_scores[tag]) for tag in common_tags], dtype=np.float32)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))

    assert mean_diff <= 0.002, (
        f"Average score drift between config/fallback paths is too large: mean_diff={mean_diff:.6f}"
    )
    assert max_diff <= 0.02, f"Max score drift between config/fallback paths is too large: max_diff={max_diff:.6f}"
