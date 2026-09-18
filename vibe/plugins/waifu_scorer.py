from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

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
from vibe.backends.runtime.pytorch import PyTorchBackend
from vibe.contracts import LabelCatalogProvider
from vibe.metadata import LabelCatalog, LabelInfo
from vibe.model_profiles import ScoreSemantics, build_scorer_profile
from vibe.plugins.shared.scores_utils import normalize_scalar
from vibe.results import ScoreResult
from vibe.settings import InferenceRequest

logger = logging.getLogger(__name__)


# region Runtime Model Definition


_WAIFU_SCORER_RUNTIME_CLS: type | None = None


def _get_runtime_model_cls(nn_module: Any) -> type:
    """Dynamically define the combined CLIP + MLP scorer module."""
    global _WAIFU_SCORER_RUNTIME_CLS
    if _WAIFU_SCORER_RUNTIME_CLS is not None:
        return _WAIFU_SCORER_RUNTIME_CLS

    class WaifuScorerRuntimeModel(nn_module.Module):
        """Combined CLIP image encoder + MLP scorer."""

        def __init__(self, *, clip_model: Any, mlp: Any) -> None:
            super().__init__()
            self.clip_model = clip_model
            self.mlp = mlp

        def forward(self, images: Any) -> Any:
            features = self.clip_model.get_image_features(images)
            # todo: check if this works with transformers v4.x and v5.x
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "image_embeds"):
                features = features.image_embeds
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return self.mlp(features).clamp(0, 10)

    _WAIFU_SCORER_RUNTIME_CLS = WaifuScorerRuntimeModel
    return _WAIFU_SCORER_RUNTIME_CLS


# endregion


# region Plugin Base


class WaifuScorerBasePlugin(ModelPlugin):
    """Shared implementation for the Eugeoter waifu scorer models."""

    family_name = "Eugeoter Aesthetic Scorers"

    SCORE_MIN = 0.0
    SCORE_MAX = 10.0
    INPUT_SIZE = 768

    # Explicitly declare that WaifuScorer outputs an uncalibrated 0-10 regression score
    profile = build_scorer_profile(score_semantics=ScoreSemantics.UNCALIBRATED)
    implements = (LabelCatalogProvider,)

    # NOTE: if user overrides source with local dir for example, then user needs to
    # use filename_map (or source_map) to allow for the same-filename files to load
    # (rename one weight file and use filename_map to point to it)
    variants = (
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(
                    id="mlp_weights",
                    name="model.safetensors",
                    role=FileRole.WEIGHTS,
                ),
                ArtifactSpec(
                    id="clip_weights",
                    name="model.safetensors",
                    role=FileRole.WEIGHTS,
                    repo_id="openai/clip-vit-large-patch14",
                ),
                ArtifactSpec(
                    id="clip_config",
                    name="config.json",
                    role=FileRole.CONFIG,
                    repo_id="openai/clip-vit-large-patch14",
                ),
                ArtifactSpec(
                    id="clip_preprocessor",
                    name="preprocessor_config.json",
                    role=FileRole.CONFIG,
                    repo_id="openai/clip-vit-large-patch14",
                ),
            ),
        ),
    )

    catalog: LabelCatalog
    _clip_preprocess: Any | None = None

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Initialize the CLIP preprocessor from its exact resolved artifact path."""
        preprocessor_path = artifacts.get("clip_preprocessor")

        try:
            from transformers import CLIPImageProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required for WaifuScorer.") from exc

        try:
            with preprocessor_path.open("r", encoding="utf-8") as f:
                prep_dict = json.load(f)
            self._clip_preprocess = CLIPImageProcessor.from_dict(prep_dict)
        except Exception as exc:
            raise RuntimeError(f"Failed to load CLIP preprocessor from '{preprocessor_path}': {exc}") from exc

        self.catalog = LabelCatalog(
            labels=(LabelInfo(index=0, name="score", category="general"),),
            categories=("general",),
        )

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        """Construct the combined PyTorch model graph using exact artifact paths."""
        if plan.backend != Backend.PYTORCH:
            raise ValueError(f"WaifuScorer only supports PyTorch backend, got '{plan.backend}'.")

        try:
            from safetensors.torch import load_file
            from torch import nn
            from transformers import CLIPConfig, CLIPModel
        except ImportError as exc:
            raise RuntimeError("PyTorch, safetensors, and transformers are required.") from exc

        # Load CLIP Config from exact artifact path
        config_path = artifacts.get("clip_config")
        try:
            with config_path.open("r", encoding="utf-8") as f:
                config_dict = json.load(f)
            clip_config = CLIPConfig.from_dict(config_dict)
        except Exception as exc:
            raise RuntimeError(f"Failed to load CLIP config from '{config_path}': {exc}") from exc

        # Instantiate CLIPModel from config and load exact weights file
        clip_model = CLIPModel(clip_config)
        clip_weights_path = artifacts.get("clip_weights")
        try:
            if clip_weights_path.suffix.lower() == ".safetensors":
                clip_state = load_file(clip_weights_path, device="cpu")
            else:
                import torch

                clip_state = torch.load(clip_weights_path, map_location="cpu")
            clip_model.load_state_dict(clip_state, strict=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load CLIP weights from '{clip_weights_path}': {exc}") from exc

        clip_model.eval()
        clip_model.requires_grad_(False)

        # Build and load MLP Head
        mlp = self._build_mlp(nn)
        mlp_path = artifacts.get("mlp_weights")
        try:
            mlp_state = load_file(mlp_path, device="cpu")
            normalized_state = self._normalize_mlp_state_dict(mlp_state)
            mlp.load_state_dict(normalized_state, strict=True)
            mlp.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load MLP weights from '{mlp_path}': {exc}") from exc

        # Assemble Combined Runtime Model
        runtime_cls = _get_runtime_model_cls(nn)
        model = runtime_cls(clip_model=clip_model, mlp=mlp)

        backend = PyTorchBackend()
        backend.load(model, plan)
        return backend

    def preprocess(self, image: Any, request: InferenceRequest | None = None) -> Any:
        if self._clip_preprocess is None:
            raise RuntimeError("Waifu scorer preprocessor is not loaded.")

        try:
            batch = self._clip_preprocess(image, return_tensors="pt")
            return batch["pixel_values"]
        except Exception as exc:
            raise RuntimeError(f"Waifu scorer preprocess failed: {exc}") from exc

    def postprocess(self, raw_output: Any) -> ScoreResult:
        scores = np.asarray(raw_output, dtype=np.float32).reshape(-1)
        score = 0.0 if scores.size == 0 else float(np.clip(scores[0], self.SCORE_MIN, self.SCORE_MAX))

        return ScoreResult(
            score=score,
            score_min=self.SCORE_MIN,
            score_max=self.SCORE_MAX,
            normalized_score=normalize_scalar(score, self.SCORE_MIN, self.SCORE_MAX),
        )

    def _normalize_mlp_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if not state_dict:
            return state_dict

        # Strip non-parameter batchnorm tracking counters
        clean = {k: v for k, v in state_dict.items() if not k.endswith(".num_batches_tracked")}

        # Prefixes added by PyTorch Lightning, torch.compile, or nested module wrappers
        prefixes = ("_orig_mod.", "model.", "mlp.", "layers.")

        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if any(k.startswith(prefix) for k in clean):
                    clean = {(k.removeprefix(prefix)): v for k, v in clean.items()}
                    changed = True

        return clean

    def _build_mlp(self, nn: Any) -> Any:
        return nn.Sequential(
            nn.Linear(self.INPUT_SIZE, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )


# endregion


# region Concrete Plugins


class WaifuScorerV3Plugin(WaifuScorerBasePlugin):
    identity = ModelIdentity(
        model_id="waifu-scorer-v3",
        display_name="Waifu Scorer v3",
        description="Anime image aesthetic scorer using CLIP ViT-L/14 image encoder and Waifu Scorer v3 MLP head.",
    )
    default_repo_id = "Eugeoter/waifu-scorer-v3"


class WaifuScorerV4Plugin(WaifuScorerBasePlugin):
    identity = ModelIdentity(
        model_id="waifu-scorer-v4-beta",
        display_name="Waifu Scorer v4 Beta",
        description="Anime image aesthetic scorer using CLIP ViT-L/14 image encoder and Waifu Scorer v4-beta MLP head.",
    )
    default_repo_id = "Eugeoter/waifu-scorer-v4-beta"


# endregion
