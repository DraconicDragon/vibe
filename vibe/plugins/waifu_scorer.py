from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    FileRole,
    ModelCapabilities,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
)
from vibe.result_transforms import NormalizedScore
from vibe.results import OutputType, ScoreResult

logger = logging.getLogger(__name__)


def _get_runtime_model_cls(nn_module: Any) -> type:
    """Dynamically define and return the WaifuScorerRuntimeModel class given torch.nn."""

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


class WaifuScorerBasePlugin(ModelPlugin):
    """Shared implementation for the Eugeoter waifu scorer models."""

    family_name = "Eugeoter Aesthetic Scorers"

    SCORE_MIN = 0.0
    SCORE_MAX = 10.0
    INPUT_SIZE = 768

    capabilities = ModelCapabilities(
        output_type=OutputType.SCORE,
        output_categories=(),
        transforms=(NormalizedScore,),
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

    _backend: Backend | None = None
    _backend_instance: Any | None = None
    _clip_model: Any | None = None
    _clip_preprocess: Any | None = None

    def configure(self, **kwargs: Any) -> None:
        self._backend = kwargs.get("backend")
        self._backend_instance = kwargs.get("backend_instance")

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        if self._backend != Backend.PYTORCH:
            return

        backend = self._backend_instance
        if backend is None:
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to use the waifu scorer plugin.") from exc

        model_or_state = getattr(backend, "raw", None)
        if isinstance(model_or_state, nn.Module):
            return
        if not isinstance(model_or_state, dict):
            raise RuntimeError("Waifu scorer weights must be a PyTorch state dict or nn.Module.")

        device = getattr(backend, "device", "cpu")
        clip_model, clip_preprocess = self._load_clip_model(device, artifacts)
        mlp = self._build_mlp()
        normalized_state = self._normalize_mlp_state_dict(model_or_state)
        self._load_state_dict(mlp, normalized_state)

        # Build class dynamically without importing torch at module import time
        runtime_cls = _get_runtime_model_cls(nn)
        model = runtime_cls(clip_model=clip_model, mlp=mlp)
        backend._model = model

        apply_precision = getattr(backend, "_apply_precision_plan", None)
        if callable(apply_precision):
            apply_precision(torch)
        else:
            model.to(device=device)

        model.eval()

        self._clip_model = clip_model
        self._clip_preprocess = clip_preprocess

    def preprocess(self, image: Any) -> Any:
        if self._clip_preprocess is None:
            raise RuntimeError("Waifu scorer preprocess is unavailable until the plugin is loaded.")

        try:
            batch = self._clip_preprocess(image, return_tensors="pt")
            return batch["pixel_values"]
        except Exception as exc:
            raise RuntimeError("Waifu scorer preprocess could not prepare image tensors.") from exc

    def postprocess(self, raw_output: Any) -> ScoreResult:
        scores = np.asarray(raw_output, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            score = 0.0
        else:
            score = float(np.clip(scores[0], self.SCORE_MIN, self.SCORE_MAX))

        return ScoreResult(
            score=score,
            score_min=self.SCORE_MIN,
            score_max=self.SCORE_MAX,
        )

    def _load_clip_model(self, device: str, artifacts: ArtifactMap) -> tuple[Any, Any]:
        try:
            from transformers import CLIPImageProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers could not be imported for the waifu scorer model. "
                "This is usually a dependency mismatch between transformers and huggingface_hub. "
                "Try upgrading or reinstalling both packages together."
            ) from exc

        clip_weights = artifacts.get("clip_weights")
        clip_config = artifacts.get("clip_config")
        clip_preprocessor = artifacts.get("clip_preprocessor")

        clip_dir = clip_weights.parent
        if clip_config.parent != clip_dir or clip_preprocessor.parent != clip_dir:
            raise RuntimeError(
                "CLIP files must be located in the same folder to load locally. "
                "Use source_map or file_name_map to align file locations."
            )

        processor = CLIPImageProcessor.from_pretrained(str(clip_dir), local_files_only=True)
        clip_model = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True)
        clip_model = clip_model.to(device=device)  # ty:ignore[missing-argument]
        clip_model.eval()
        clip_model.requires_grad_(False)

        return clip_model, processor

    def _normalize_mlp_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if not state_dict:
            return state_dict

        # MLP state dict from reference script defines self.layers = nn.Sequential(...)
        # So we strip 'layers.' to map it to our unwrapped nn.Sequential head:
        if any(key.startswith("layers.") for key in state_dict):
            state_dict = {
                key[len("layers.") :]: value for key, value in state_dict.items() if key.startswith("layers.")
            }

        if all(key.startswith("model.") for key in state_dict):
            state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}

        # Remove num_batches_tracked so strict loading works since our Sequential model may not track them
        # identically or HF checkpoint has them extra.
        return {k: v for k, v in state_dict.items() if not k.endswith(".num_batches_tracked")}

    def _build_mlp(self) -> Any:
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to build the waifu scorer MLP.") from exc

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

    def _load_state_dict(self, model: Any, state_dict: dict[str, Any]) -> None:
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load waifu scorer weights: {exc}") from exc

        if missing:
            logger.warning("Waifu scorer missing keys for model_id=%s: %s", self.identity.model_id, missing[:8])
        if unexpected:
            logger.warning("Waifu scorer unexpected keys for model_id=%s: %s", self.identity.model_id, unexpected[:8])


# region Model Variants


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


# endregion Model Variants
