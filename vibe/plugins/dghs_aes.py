# Note: Scores deviate a negligible amount from deepgh-imgutils implementation

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    ExecutionPlan,
    FileRole,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
    RuntimeExecutor,
)
from vibe.backends.runtime.onnx import ONNXBackend
from vibe.backends.runtime.pytorch import PyTorchBackend
from vibe.contracts import CalibrationProvider, LabelCatalogProvider
from vibe.metadata import CalibrationTable, LabelCatalog, LabelInfo
from vibe.model_profiles import build_multi_scorer_profile
from vibe.plugins.shared.scores_utils import (
    load_samples_file,
    normalize_scalar,
    softmax,
)
from vibe.plugins.shared.tagger_shared import _to_rgb_with_background
from vibe.results import MultiScoreResult, ScoreEntry
from vibe.settings import InferenceRequest

logger = logging.getLogger(__name__)


# region Helpers


def _calculate_aesthetic_rank_score(entries: list[ScoreEntry]) -> float:
    """
    Calculate the expected ranking score over DeepGHS categorical aesthetic labels.
    DeepGHS meta.json labels run in descending order: masterpiece (weight 6) down to worst (weight 0).
    """
    total = len(entries)
    if total == 0:
        return 0.0
    return float(sum((total - 1 - i) * entry.normalized_score for i, entry in enumerate(entries)))


def _normalize_aesthetic_score(entries: list[ScoreEntry]) -> float:
    """Normalize categorical aesthetic expectations to a [0, 1] range."""
    if not entries:
        return 0.0
    rank_score = _calculate_aesthetic_rank_score(entries)
    max_v = float(max(len(entries) - 1, 1))
    return float(np.clip(rank_score / max_v, 0.0, 1.0))


def _build_variants(hf_subdir: str | None = None) -> tuple[ModelVariant, ...]:
    return (
        ModelVariant(
            backend=Backend.PYTORCH,
            hf_subdir=hf_subdir,
            artifacts=(
                ArtifactSpec(id="model_pt", name="model.ckpt", role=FileRole.WEIGHTS),
                ArtifactSpec(id="meta", name="meta.json", role=FileRole.CONFIG),
                ArtifactSpec(id="samples", name="samples.npz", role=FileRole.MAPPING),
            ),
        ),
        ModelVariant(
            backend=Backend.ONNX,
            hf_subdir=hf_subdir,
            artifacts=(
                ArtifactSpec(id="model_onnx", name="model.onnx", role=FileRole.WEIGHTS),
                ArtifactSpec(id="meta", name="meta.json", role=FileRole.CONFIG),
                ArtifactSpec(id="samples", name="samples.npz", role=FileRole.MAPPING),
            ),
        ),
    )


# endregion Helpers


# region Base Plugin


class DeepGHSAnimeAesPlugin(ModelPlugin):
    """Shared implementation for DeepGHS anime aesthetic scorers."""

    family_name = "DeepGHS Anime Aesthetic Scorers"

    IMAGE_SIZE = 448
    SCORE_MIN = 0.0
    SCORE_MAX = 1.0

    profile = build_multi_scorer_profile(
        output_extras={"percentile": "Score percentile calibrated against training dataset samples."}
    )
    implements = (LabelCatalogProvider, CalibrationProvider)

    catalog: LabelCatalog
    calibration: CalibrationTable
    _labels: list[str]
    _image_size: int

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        meta_path = artifacts.get("meta")
        meta = self._read_meta_json(meta_path)

        labels = meta.get("labels")
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise RuntimeError("meta.json must contain a list of label strings.")
        self._labels = [str(label) for label in labels]

        label_infos = [LabelInfo(index=idx, name=label, category="general") for idx, label in enumerate(self._labels)]
        self.catalog = LabelCatalog(labels=tuple(label_infos), categories=("general",))

        self._image_size = self._resolve_image_size(meta)

        samples_path = artifacts.get("samples")
        x, y = load_samples_file(samples_path)
        self.calibration = CalibrationTable(x=x, y=y, source="samples.npz")

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        if plan.backend == Backend.ONNX:
            onnx_path = artifacts.get("model_onnx")
            backend = ONNXBackend()
            backend.load(onnx_path, plan)
            return backend

        if plan.backend == Backend.PYTORCH:
            ckpt_path = artifacts.get("model_pt")
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("PyTorch is required.") from exc

            # DeepGHS checkpoints are compiled TorchScript modules (.ckpt)
            try:
                model = torch.jit.load(str(ckpt_path), map_location="cpu")
            except Exception as exc:
                raise RuntimeError(f"Failed to load DeepGHS TorchScript checkpoint '{ckpt_path}': {exc}") from exc

            backend = PyTorchBackend()
            backend.load(model, plan)
            return backend

        raise ValueError(f"Unsupported backend '{plan.backend}'.")

    def preprocess(self, image: Any, request: InferenceRequest | None = None) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = _to_rgb_with_background(image)
        image = image.resize((self._image_size, self._image_size), Image.Resampling.BICUBIC)

        image_array = np.asarray(image, dtype=np.float32)
        image_array = (image_array / 127.5) - 1.0
        image_array = np.ascontiguousarray(np.transpose(image_array, (2, 0, 1)))
        return np.expand_dims(image_array, axis=0)

    def postprocess(self, raw_output: Any) -> MultiScoreResult:
        raw_scores = self._flatten_scores(raw_output)

        if len(raw_scores) != len(self._labels):
            logger.warning(
                "Score length mismatch for model '%s': expected %d scores for labels %s, got %d.",
                self.identity.model_id,
                len(self._labels),
                self._labels,
                len(raw_scores),
            )

        # PyTorch TorchScript models output raw logits; ONNX exports may or may not embed Softmax.
        # If values fall outside [0, 1] or their sum deviates from 1.0, convert logits via Softmax.
        is_logits = (
            self.active_backend == Backend.PYTORCH
            or np.any(raw_scores < -1e-4)
            or np.any(raw_scores > 1.0 + 1e-4)
            or abs(float(np.sum(raw_scores)) - 1.0) > 0.05
        )

        probabilities = softmax(raw_scores) if is_logits else np.clip(raw_scores, 0.0, 1.0)

        entries = [
            ScoreEntry(
                label=label,
                score=float(prob),
                score_min=self.SCORE_MIN,
                score_max=self.SCORE_MAX,
                normalized_score=normalize_scalar(float(prob), self.SCORE_MIN, self.SCORE_MAX),
            )
            for label, prob in zip(self._labels, probabilities, strict=False)
        ]

        generic_normalized = _normalize_aesthetic_score(entries)

        # Calculate calibrated dataset percentile
        extras = {}
        if self.calibration is not None:
            rank_score = _calculate_aesthetic_rank_score(entries)
            extras["percentile"] = float(np.interp(rank_score, self.calibration.x, self.calibration.y))

        return MultiScoreResult(entries=entries, normalized_score=generic_normalized, extras=extras)

    def _read_meta_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to read DeepGHS meta.json: {exc}") from exc

    def _resolve_image_size(self, meta: dict[str, Any]) -> int:
        for key in ("image_size", "input_size", "img_size"):
            value = meta.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return self.IMAGE_SIZE

    def _flatten_scores(self, raw_output: Any) -> np.ndarray:
        if isinstance(raw_output, (tuple, list)):
            raw_output = raw_output[0]

        scores = np.asarray(raw_output, dtype=np.float32)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        elif scores.ndim > 1:
            if scores.shape[0] == 1:
                scores = np.squeeze(scores, axis=0)
            scores = np.ravel(scores)
        return scores.astype(np.float32, copy=False)


# endregion Base Plugin


# region Model Variants


class DGHSAesSwinV2xPlugin(DeepGHSAnimeAesPlugin):
    identity = ModelIdentity(
        model_id="dghs-aes-swinv2pv3-ls0.2-x",
        display_name="DeepGHS Aesthetic SwinV2 PV3 x",
        description="Anime image aesthetic scorer.",
    )
    default_repo_id = "deepghs/anime_aesthetic"
    variants = _build_variants("swinv2pv3_v0_448_ls0.2_x")


class DGHSAesSwinV2Plugin(DeepGHSAnimeAesPlugin):
    identity = ModelIdentity(
        model_id="dghs-aes-swinv2pv3-ls0.2",
        display_name="DeepGHS Aesthetic SwinV2 PV3",
        description="Anime image aesthetic scorer.",
    )
    default_repo_id = "deepghs/anime_aesthetic"
    variants = _build_variants("swinv2pv3_v0_448_ls0.2")


class DGHSAesCaformerS36Plugin(DeepGHSAnimeAesPlugin):
    identity = ModelIdentity(
        model_id="dghs-aes-caformer-s36-ls0.2",
        display_name="DeepGHS Aesthetic CaFormer S36",
        description="Anime image aesthetic scorer.",
    )
    default_repo_id = "deepghs/anime_aesthetic"
    IMAGE_SIZE = 384
    variants = _build_variants("caformer_s36_v0_ls0.2")


# endregion Model Variants
