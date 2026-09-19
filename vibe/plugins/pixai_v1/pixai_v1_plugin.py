"""
PixAI Tagger v1.0 ModelPlugin implementation.
High-resolution (1008x1008) multi-label anime tagger based on SAM-3 ViTDet by PixAI Labs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as vF
from PIL import Image
from torchvision import transforms

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
from vibe.metadata import LabelCatalog, LabelInfo, TagFilterRecommendation
from vibe.model_profiles import build_tagger_profile
from vibe.plugins.pixai_v1.model import ViTDetCls, ViTDetClsConfig
from vibe.plugins.shared.tagger_shared import (
    build_categorized_tag_result,
    normalize_output_scores,
)
from vibe.results import TagResult
from vibe.settings import InferenceRequest
from vibe.tag_categories import TagCategory

logger = logging.getLogger(__name__)

_PIXAI_CATEGORIES = (
    TagCategory.GENERAL,
    TagCategory.CHARACTER,
    TagCategory.COPYRIGHT,
    TagCategory.STYLE,
    TagCategory.META,
    TagCategory.RATING,
)


def _rescale_pad_tensor(image: torch.Tensor, output_size: int = 1008) -> torch.Tensor:
    """
    Aspect-preserving resize and zero-pad to output_size.
    Matches PixAI's rescale_pad function from tagger_pipeline.py.
    """
    h, w = image.shape[-2:]
    if h != output_size or w != output_size:
        r = min(output_size / h, output_size / w)
        new_h, new_w = int(h * r), int(w * r)
        ph = output_size - new_h
        pw = output_size - new_w
        left = pw // 2
        right = pw - left
        top = ph // 2
        bottom = ph - top
        image = transforms.functional.resize(image, [new_h, new_w])
        image = transforms.functional.pad(image, [left, top, right, bottom], 0)
    return image


def _preprocess_pixai_image(image: Any, target_size: int = 1008) -> torch.Tensor:
    """
    Preprocess PIL Image or NumPy array into a normalized PyTorch tensor [1, 3, 1008, 1008].
    Matches PixAI's RescalePadProcessor from tagger_pipeline.py.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    if image.mode != "RGB":
        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

    tensor = vF.to_tensor(image)  # Converts to float32 [0.0, 1.0] [C, H, W]
    padded = _rescale_pad_tensor(tensor, target_size)
    normalized = vF.normalize(padded, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return normalized.unsqueeze(0)


class PixAITaggerPlugin(ModelPlugin):
    """PixAI Tagger v1.0 ModelPlugin implementation."""

    family_name = "PixAI Taggers"

    identity = ModelIdentity(
        model_id="pixai-tagger-v1.0",
        display_name="PixAI Tagger v1.0",
        description="Anime image tagger by PixAI Labs based on SAM-3 ViTDet with 30k+ tags. Dataset cutoff: May 2026",
    )
    default_repo_id = "pixai-labs/pixai-tagger-v1.0"

    profile = build_tagger_profile(
        categories=_PIXAI_CATEGORIES,
        recommended_filter=TagFilterRecommendation(
            global_threshold=0.20,
            category_thresholds={
                TagCategory.GENERAL: 0.17,
                TagCategory.CHARACTER: 0.27,
                TagCategory.COPYRIGHT: 0.24,
                TagCategory.STYLE: 0.15,
                TagCategory.META: 0.17,
                TagCategory.RATING: 0.41,
            },
        ),
    )
    implements = (LabelCatalogProvider,)

    variants = (
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(
                    id="model_pt",
                    name="model.safetensors",
                    role=FileRole.WEIGHTS,
                ),
                ArtifactSpec(
                    id="config",
                    name="config.json",
                    role=FileRole.CONFIG,
                ),
            ),
        ),
    )

    catalog: LabelCatalog
    _raw_tag_names: list[str]
    _category_indices: dict[str, list[int]]

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Parse tags and category block boundaries directly from config.json."""
        config_path = artifacts.get("config")
        with config_path.open("r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        self._raw_tag_names = cfg_data.get("tags", [])
        tags_split: list[list[Any]] = cfg_data.get("tags_split", [])

        self._category_indices = {}
        label_infos: list[LabelInfo] = []
        seen_categories: set[str] = set()

        curr_idx = 0
        for cat_raw, count in tags_split:
            cat_name = str(cat_raw).strip()
            seen_categories.add(cat_name)
            indices = list(range(curr_idx, curr_idx + count))
            self._category_indices[cat_name] = indices

            for idx in indices:
                tag_name = self._raw_tag_names[idx]
                label_infos.append(LabelInfo(index=idx, name=tag_name, category=cat_name))

            curr_idx += count

        self.catalog = LabelCatalog(labels=tuple(label_infos), categories=tuple(seen_categories))
        logger.info(
            "PixAI Tagger v1.0 vocabulary loaded: %d tags across %d categories",
            len(self._raw_tag_names),
            len(self._category_indices),
        )

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        """Instantiate ViTDetCls from config.json and load safetensors weights."""
        if plan.backend != Backend.PYTORCH:
            raise ValueError(f"PixAI Tagger currently only supports PyTorch, got '{plan.backend}'.")

        config_path = artifacts.get("config")
        with config_path.open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        # Build ViTDetClsConfig directly using the parameters in config.json
        cfg = ViTDetClsConfig(**cfg_dict)

        logger.info("Instantiating PixAI ViTDetCls architecture...")
        model = ViTDetCls(cfg)

        weights_path = artifacts.get("model_pt")
        from safetensors.torch import load_file

        logger.info("Loading PyTorch weights from %s...", weights_path)
        state_dict = load_file(str(weights_path), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            logger.warning("PixAI load_state_dict: missing=%s, unexpected=%s", missing[:5], unexpected[:5])

        model.eval()
        backend = PyTorchBackend()
        backend.load(model, plan)
        return backend

    def preprocess(self, image: Any, request: InferenceRequest | None = None) -> torch.Tensor:
        del request
        return _preprocess_pixai_image(image, target_size=1008)

    def postprocess(self, raw_output: Any) -> TagResult:
        probs = normalize_output_scores(
            raw_output,
            is_logits=True,
            expected_count=len(self._raw_tag_names),
        )
        return build_categorized_tag_result(self._raw_tag_names, probs, self._category_indices)
