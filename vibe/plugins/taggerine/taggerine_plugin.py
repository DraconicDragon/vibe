"""
Taggerine ModelPlugin implementation.
DINOv3 ViT-H/16+ with an added linear projection head by lodestones.
"""

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
from vibe.metadata import LabelCatalog, LabelInfo, TagFilterRecommendation
from vibe.model_profiles import build_tagger_profile
from vibe.plugins.shared.tagger_shared import (
    build_categorized_tag_result,
    normalize_output_scores,
    resolve_category_name,
)
from vibe.results import TagResult
from vibe.settings import InferenceRequest
from vibe.tag_categories import E621_CATEGORY_LABELS, TagCategory

logger = logging.getLogger(__name__)

_TAGGERINE_CATEGORIES = (
    TagCategory.GENERAL,
    TagCategory.ARTIST,
    TagCategory.COPYRIGHT,
    TagCategory.CHARACTER,
    TagCategory.SPECIES,
    TagCategory.INVALID,
    TagCategory.META,
    TagCategory.LORE,
)

_MAX_SIZE = 1024
_PATCH_SIZE = 16
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _snap(x: int, m: int) -> int:
    return max(m, (x // m) * m)


class TaggerinePlugin(ModelPlugin):
    family_name = "Lodestone's Taggerine"

    identity = ModelIdentity(
        model_id="taggerine",
        display_name="Taggerine",
        description="E621 + Danbooru tagger by Lodestones using DINOv3 ViT-H/16+.",
    )
    default_repo_id = "lodestones/taggerine"

    profile = build_tagger_profile(
        categories=_TAGGERINE_CATEGORIES,
        recommended_filter=TagFilterRecommendation(global_threshold=0.35),
    )
    implements = (LabelCatalogProvider,)

    variants = (
        # Index 0: Default PyTorch variant
        ModelVariant(
            variant_id="bf16-mixed",
            backend=Backend.PYTORCH,
            description="Mixed precision checkpoint (BF16 backbone + FP32 head). More efficient compared to original since the backbone was never trained in FP32.",
            repo_id="DraconicDragon/taggerine-mixed-bf16",
            artifacts=(
                ArtifactSpec(
                    id="model_pt",
                    name="tagger_proto.safetensors",
                    role=FileRole.WEIGHTS,
                ),
                ArtifactSpec(
                    id="tag_list",
                    name="tagger_vocab_with_categories_and_alias_updated.json",
                    role=FileRole.TAG_LIST,
                ),
            ),
        ),
        # Index 1: Opt-in official
        ModelVariant(
            variant_id="original",
            backend=Backend.PYTORCH,
            description="Original checkpoint in full FP32.",
            artifacts=(
                ArtifactSpec(
                    id="model_pt",
                    name="tagger_proto.safetensors",
                    role=FileRole.WEIGHTS,
                ),
                ArtifactSpec(
                    id="tag_list",
                    name="tagger_vocab_with_categories_and_alias_updated.json",
                    role=FileRole.TAG_LIST,
                ),
            ),
        ),
    )

    catalog: LabelCatalog
    _raw_tag_names: list[str]
    _category_indices: dict[str, list[int]]

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Parse vocabulary JSON to populate labels and categories."""
        vocab_path = artifacts.get("tag_list")

        with vocab_path.open("r", encoding="utf-8") as handle:
            vocab_data = json.load(handle)

        self._raw_tag_names = vocab_data.get("idx2tag", [])
        self._category_indices = {}

        label_infos: list[LabelInfo] = []
        seen_categories: set[str] = set()

        if "tag2category" in vocab_data:
            tag2category = vocab_data["tag2category"]
            for idx, tag in enumerate(self._raw_tag_names):
                cat_id = tag2category.get(tag, 0)
                cat_name = resolve_category_name(cat_id, E621_CATEGORY_LABELS, namespace="e621")
                self._category_indices.setdefault(cat_name, []).append(idx)
                seen_categories.add(cat_name)
                label_infos.append(LabelInfo(index=idx, name=tag, category=cat_name))
        elif "idx2category" in vocab_data:
            for idx, cat_id in enumerate(vocab_data["idx2category"]):
                cat_name = resolve_category_name(cat_id, E621_CATEGORY_LABELS, namespace="e621")
                tag = self._raw_tag_names[idx]
                self._category_indices.setdefault(cat_name, []).append(idx)
                seen_categories.add(cat_name)
                label_infos.append(LabelInfo(index=idx, name=tag, category=cat_name))
        else:
            cat_name = TagCategory.GENERAL.value
            self._category_indices[cat_name] = list(range(len(self._raw_tag_names)))
            seen_categories.add(cat_name)
            for idx, tag in enumerate(self._raw_tag_names):
                label_infos.append(LabelInfo(index=idx, name=tag, category=cat_name))

        self.catalog = LabelCatalog(labels=tuple(label_infos), categories=tuple(seen_categories))
        logger.info("Taggerine vocab loaded: %d tags", len(self._raw_tag_names))

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        """Build the DINOv3Tagger model and load weights."""
        if plan.backend != Backend.PYTORCH:
            raise ValueError(f"Taggerine only supports PyTorch, got '{plan.backend}'.")

        weights_path = artifacts.get("model_pt")
        from safetensors.torch import load_file

        from .model import DINOv3Tagger, _build_head_from_checkpoint, split_and_clean_state_dict

        sd = load_file(weights_path, device="cpu")
        backbone_sd, head_sd = split_and_clean_state_dict(sd)

        if not head_sd:
            raise RuntimeError("Taggerine checkpoint contains no head keys.")

        model = DINOv3Tagger()
        head_module, head_sd_remapped = _build_head_from_checkpoint(
            head_sd, in_dim=6400, num_tags=len(self._raw_tag_names)
        )
        model.head = head_module

        model.backbone.load_state_dict(backbone_sd, strict=True)
        model.head.load_state_dict(head_sd_remapped, strict=True)
        model.eval()

        backend = PyTorchBackend()
        backend.load(model, plan)
        return backend

    def preprocess(self, image: Any, request: InferenceRequest | None = None) -> Any:
        import torch
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        img = image.convert("RGB")
        w, h = img.size

        # Target long-edge (snapped to patch multiple)
        long_edge = max(w, h)
        target_long = _snap(min(long_edge, _MAX_SIZE), _PATCH_SIZE)
        scale = target_long / long_edge

        new_w = _snap(max(_PATCH_SIZE, round(w * scale)), _PATCH_SIZE)
        new_h = _snap(max(_PATCH_SIZE, round(h * scale)), _PATCH_SIZE)

        # Standard PIL Lanczos resize
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to numpy, scale to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize (ImageNet)
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC to CHW
        arr = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))

        # Vibe expects all preprocessed tensors to have a batch dimension of 1
        return torch.from_numpy(arr).unsqueeze(0)

    def postprocess(self, raw_output: Any) -> TagResult:
        probs = normalize_output_scores(raw_output, is_logits=True, expected_count=len(self._raw_tag_names))
        return build_categorized_tag_result(self._raw_tag_names, probs, self._category_indices)
