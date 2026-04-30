# Todo: find out if possible to use timm or open_clip_torch instead of clip library OR
# find out if clip lib can take external weights instead of downloading by itself

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.results import OutputType, ScoreResult

logger = logging.getLogger(__name__)


@dataclass
class WaifuScorerResult(ScoreResult):
    """Single-value score result for the Waifu scorer models."""


class WaifuScorerBasePlugin(ModelPlugin):
    """Shared implementation for the Eugeoter waifu scorer models."""

    _abstract = True

    SCORE_MIN = 0.0
    SCORE_MAX = 10.0
    INPUT_SIZE = 768

    output_type = OutputType.SCORE
    supported_backends = [Backend.PYTORCH]
    supported_processors = []

    MLP_WEIGHTS_KEY = "mlp_weights"
    CLIP_WEIGHTS_KEY = "clip_weights"
    CLIP_CONFIG_KEY = "clip_config"
    CLIP_PREPROCESSOR_KEY = "clip_preprocessor"

    required_files = [
        FileSpec(
            name="model.safetensors",
            key=MLP_WEIGHTS_KEY,
            role=FileRole.WEIGHTS,
            backends=[Backend.PYTORCH],
        ),
        FileSpec(
            name="model.safetensors",
            key=CLIP_WEIGHTS_KEY,
            role=FileRole.WEIGHTS,
            backends=[Backend.PYTORCH],
            repo_id="openai/clip-vit-large-patch14",
        ),
        FileSpec(
            name="config.json",
            key=CLIP_CONFIG_KEY,
            role=FileRole.CONFIG,
            repo_id="openai/clip-vit-large-patch14",
        ),
        FileSpec(
            name="preprocessor_config.json",
            key=CLIP_PREPROCESSOR_KEY,
            role=FileRole.CONFIG,
            repo_id="openai/clip-vit-large-patch14",
        ),
    ]

    _backend: Backend | None = None
    _backend_instance: Any | None = None
    _clip_model: Any | None = None
    _clip_preprocess: Any | None = None

    def configure(self, **kwargs: Any) -> None:
        self._backend = kwargs.get("backend")
        self._backend_instance = kwargs.get("backend_instance")

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
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
        clip_model, clip_preprocess = self._load_clip_model(device, file_map)
        mlp = self._build_mlp()
        normalized_state = self._normalize_mlp_state_dict(model_or_state)
        self._load_state_dict(mlp, normalized_state)

        model = WaifuScorerRuntimeModel(clip_model=clip_model, mlp=mlp)
        backend._model = model

        apply_precision = getattr(backend, "_apply_precision_plan", None)
        if callable(apply_precision):
            apply_precision(torch)
        else:
            device = getattr(backend, "device", "cpu")
            model.to(device=device)

        model.eval()

        self._clip_model = clip_model
        self._clip_preprocess = clip_preprocess

    def preprocess(self, image: Any) -> Any:
        if self._clip_preprocess is None:
            raise RuntimeError("Waifu scorer preprocess is unavailable until the plugin is loaded.")

        if hasattr(self._clip_preprocess, "__call__"):
            batch = self._clip_preprocess(images=image, return_tensors="pt")
            try:
                return batch["pixel_values"]
            except Exception:
                pass

        raise RuntimeError("Waifu scorer preprocess could not prepare image tensors.")

    def postprocess(self, raw_output: Any) -> ScoreResult:
        scores = np.asarray(raw_output, dtype=np.float32).reshape(-1)
        if scores.size == 0:
            score = 0.0
        else:
            score = float(np.clip(scores[0], self.SCORE_MIN, self.SCORE_MAX))

        return WaifuScorerResult(
            score=score,
            score_min=self.SCORE_MIN,
            score_max=self.SCORE_MAX,
        )

    def _load_clip_model(self, device: str, file_map: dict[str, Path]) -> tuple[Any, Any]:
        try:
            from transformers import CLIPImageProcessor, CLIPModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to run the waifu scorer models since they are made for CLIP.\n"
                + "Install it with: pip install transformers"
            ) from exc

        clip_weights = file_map.get(self.CLIP_WEIGHTS_KEY)
        clip_config = file_map.get(self.CLIP_CONFIG_KEY)
        clip_preprocessor = file_map.get(self.CLIP_PREPROCESSOR_KEY)

        if not clip_weights or not clip_config or not clip_preprocessor:
            raise RuntimeError("Waifu scorer is missing CLIP files; check resolved sources.")

        clip_dir = clip_weights.parent
        if clip_config.parent != clip_dir or clip_preprocessor.parent != clip_dir:
            raise RuntimeError(
                "CLIP files must be located in the same folder to load locally. "
                "Use source_map or file_name_map to align file locations."
            )

        processor = CLIPImageProcessor.from_pretrained(str(clip_dir), local_files_only=True)
        clip_model = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True)
        clip_model = clip_model.to(device=device)
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
            logger.warning("Waifu scorer missing keys for model_id=%s: %s", self.model_id, missing[:8])
        if unexpected:
            logger.warning("Waifu scorer unexpected keys for model_id=%s: %s", self.model_id, unexpected[:8])


try:
    import torch.nn as nn
except ImportError:  # pragma: no cover - module discovery stays importable without torch
    nn = None  # type: ignore[assignment]


if nn is not None:

    class WaifuScorerRuntimeModel(nn.Module):
        """Combined CLIP image encoder + MLP scorer."""

        def __init__(self, *, clip_model: Any, mlp: Any) -> None:
            super().__init__()
            self.clip_model = clip_model
            self.mlp = mlp

        def forward(self, images: Any) -> Any:
            features = self.clip_model.get_image_features(images).pooler_output
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            return self.mlp(features).clamp(0, 10) / 10.0

else:  # pragma: no cover - only used when torch is missing entirely

    class WaifuScorerRuntimeModel:
        """Fallback placeholder when torch is unavailable."""

        def __init__(self, *, clip_model: Any, mlp: Any) -> None:
            self.clip_model = clip_model
            self.mlp = mlp

        def forward(self, images: Any) -> Any:
            raise RuntimeError("PyTorch is required to run the waifu scorer model.")


# region Model Variants


class WaifuScorerV3Plugin(WaifuScorerBasePlugin):
    model_id = "waifu-scorer-v3"
    aliases = []
    display_name = "Waifu Scorer v3"
    description = "Waifu scorer using an open-clip ViT-L/14 image encoder and an MLP head."
    default_hf_repo = "Eugeoter/waifu-scorer-v3"


class WaifuScorerV4Plugin(WaifuScorerBasePlugin):
    model_id = "waifu-scorer-v4-beta"
    aliases = []
    display_name = "Waifu Scorer v4 Beta"
    description = "Waifu scorer using an open-clip ViT-L/14 image encoder and an MLP head."
    default_hf_repo = "Eugeoter/waifu-scorer-v4-beta"


# endregion Model Variants
