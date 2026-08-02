from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    ExecutionRequest,
    FileRole,
    ModelCapabilities,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
    RuntimeExecutor,
)
from vibe.backends.runtime.pytorch import PyTorchBackend
from vibe.plugins.shared.scores_utils import normalize_scalar
from vibe.results import OutputType, ScoreResult

logger = logging.getLogger(__name__)


# region Runtime Model Definition


def _get_runtime_model_cls(nn_module: Any) -> type:
    """Dynamically define the combined CLIP + MLP scorer module."""

    class WaifuScorerRuntimeModel(nn_module.Module):
        """Combined CLIP image encoder + MLP scorer."""

        def __init__(self, *, clip_model: Any, mlp: Any) -> None:
            super().__init__()
            self.clip_model = clip_model
            self.mlp = mlp

        def forward(self, images: Any) -> Any:
            features = self.clip_model.get_image_features(images).pooler_output
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return self.mlp(features).clamp(0, 10)

    return WaifuScorerRuntimeModel


# endregion


# region Plugin Base


class WaifuScorerBasePlugin(ModelPlugin):
    """Shared implementation for the Eugeoter waifu scorer models."""

    family_name = "Eugeoter Aesthetic Scorers"

    SCORE_MIN = 0.0
    SCORE_MAX = 10.0
    INPUT_SIZE = 768

    capabilities = ModelCapabilities(
        output_type=OutputType.SCORE,
        output_categories=(),
        transforms=(),
    )

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

    _clip_preprocess: Any | None = None

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Initialize the CLIP preprocessor for image normalization."""
        clip_weights = artifacts.get("clip_weights")
        clip_dir = clip_weights.parent

        try:
            from transformers import CLIPImageProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required for WaifuScorer.") from exc

        self._clip_preprocess = CLIPImageProcessor.from_pretrained(str(clip_dir), local_files_only=True)

    def build_runtime(self, artifacts: ArtifactMap, request: ExecutionRequest) -> RuntimeExecutor:
        """Construct the combined PyTorch model graph and return a PyTorchBackend executor."""
        if request.backend != Backend.PYTORCH:
            raise ValueError(f"WaifuScorer only supports PyTorch backend, got '{request.backend}'.")

        try:
            from safetensors.torch import load_file
            from torch import nn
            from transformers import CLIPModel
        except ImportError as exc:
            raise RuntimeError("PyTorch, safetensors, and transformers are required.") from exc

        # Load CLIP Model
        clip_weights = artifacts.get("clip_weights")
        clip_dir = clip_weights.parent
        clip_model = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True)
        clip_model.eval()
        clip_model.requires_grad_(False)

        # Build MLP Head
        mlp = self._build_mlp(nn)
        mlp_path = artifacts.get("mlp_weights")
        mlp_state = load_file(mlp_path, device="cpu")
        normalized_state = self._normalize_mlp_state_dict(mlp_state)
        mlp.load_state_dict(normalized_state, strict=True)
        mlp.eval()

        # Assemble Combined Runtime Model
        runtime_cls = _get_runtime_model_cls(nn)
        model = runtime_cls(clip_model=clip_model, mlp=mlp)

        backend = PyTorchBackend()
        backend.load(model, request)
        return backend

    def preprocess(self, image: Any) -> Any:
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

        if any(key.startswith("layers.") for key in state_dict):
            state_dict = {
                key[len("layers.") :]: value for key, value in state_dict.items() if key.startswith("layers.")
            }

        if all(key.startswith("model.") for key in state_dict):
            state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}

        return {k: v for k, v in state_dict.items() if not k.endswith(".num_batches_tracked")}

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
