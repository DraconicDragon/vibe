from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    FileRole,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
)
from vibe.contracts import LabelCatalogProvider
from vibe.metadata import LabelCatalog, ModelProfile, TagFilterRecommendation
from vibe.model_profiles import build_tagger_profile
from vibe.plugins.shared.generic_timm_pipeline import TimmPipelineMixin
from vibe.plugins.shared.tagger_shared import (
    ParsedTagData,
    build_categorized_tag_result,
    load_tag_metadata,
    normalize_output_scores,
    preprocess_tagger_image,
)
from vibe.results import TagResult
from vibe.settings import InferenceRequest
from vibe.tag_categories import DANBOORU_CATEGORY_LABELS, TagCategory

logger = logging.getLogger(__name__)

_WD_CATEGORIES = (
    TagCategory.RATING,
    TagCategory.GENERAL,
    TagCategory.CHARACTER,
)


def _build_wd_profile(
    recommended_filter: TagFilterRecommendation | None = None,
) -> ModelProfile:
    """Helper to build standard WD tagger profile with fixed WD categories."""
    return build_tagger_profile(
        categories=_WD_CATEGORIES,
        recommended_filter=recommended_filter,
    )


class WDTaggerBasePlugin(TimmPipelineMixin, ModelPlugin):
    """Shared implementation for WaifuDiffusion taggers by SmilingWolf."""

    family_name = "SmilingWolf WD Taggers"

    profile = _build_wd_profile()
    implements = (LabelCatalogProvider,)

    variants = (
        ModelVariant(
            backend=Backend.ONNX,
            artifacts=(
                ArtifactSpec(id="model_onnx", name="model.onnx", role=FileRole.WEIGHTS),
                ArtifactSpec(id="tag_list", name="selected_tags.csv", role=FileRole.TAG_LIST),
            ),
        ),
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(id="model_pt", name="model.safetensors", role=FileRole.WEIGHTS),
                ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG),
                ArtifactSpec(id="tag_list", name="selected_tags.csv", role=FileRole.TAG_LIST),
            ),
        ),
    )

    IMAGE_SIZE = 448

    catalog: LabelCatalog
    _tag_data: ParsedTagData

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Load tag metadata from selected_tags.csv."""
        csv_path = artifacts.get("tag_list")
        logger.info("Loading tag list from %s", csv_path)
        self._tag_data = load_tag_metadata(
            csv_path,
            category_labels=DANBOORU_CATEGORY_LABELS,
            namespace="danbooru",
        )
        self.catalog = self._tag_data.catalog
        self._num_classes = len(self._tag_data.raw_tag_names)

    def preprocess(self, image: Any, request: InferenceRequest | None = None) -> np.ndarray:
        """Convert image to layout expected by the active backend."""
        backend = self.active_backend
        if backend is None:
            raise RuntimeError(
                f"Plugin '{self.identity.model_id}' has no active backend bound to this session. "
                "This usually means the execution state was not initialized before pooled runtime reuse."
            )

        # NOTE: PyTorch - models expect standard (1, C, H, W) NCHW format
        # NOTE: ONNX - models expect (1, H, W, C) NHWC format
        layout = "NCHW" if backend == Backend.PYTORCH else "NHWC"

        if backend == Backend.PYTORCH:
            # NOTE: PyTorch expects BGR normalized to [-1, 1] range: (x - 127.5) / 127.5
            return preprocess_tagger_image(
                image,
                image_size=self.IMAGE_SIZE,
                input_layout=layout,
                rgb_to_bgr=True,
                normalize_to_unit=True,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )

        # NOTE: ONNX expects unnormalized, raw BGR [0, 255] float32
        arr = preprocess_tagger_image(
            image,
            image_size=self.IMAGE_SIZE,
            input_layout=layout,
            rgb_to_bgr=True,
            normalize_to_unit=False,
        )
        return np.ascontiguousarray(arr)

    def postprocess(self, raw_output: Any) -> TagResult:
        """Return full scored output grouped by WD tag category."""
        # PyTorch timm models output raw logits; SmilingWolf ONNX models bake Sigmoid into the graph
        is_logits = self.active_backend == Backend.PYTORCH

        scores = normalize_output_scores(
            raw_output,
            is_logits=is_logits,
            expected_count=len(self._tag_data.raw_tag_names),
        )
        return build_categorized_tag_result(self._tag_data.raw_tag_names, scores, self._tag_data.category_indices)


class WDEva02Plugin(WDTaggerBasePlugin):
    identity = ModelIdentity(
        model_id="wd-eva02-large-v3",
        display_name="WD Eva02-large Tagger v3",
        description="Danbooru tag prediction using Eva02 ViT-L architecture.",
    )
    default_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
    profile = _build_wd_profile(
        TagFilterRecommendation(
            global_threshold=0.53,
            category_thresholds={TagCategory.CHARACTER: 0.75},
        )
    )


class WDSwinV2Plugin(WDTaggerBasePlugin):
    identity = ModelIdentity(
        model_id="wd-swinv2-v3",
        display_name="WD SwinV2 Tagger v3",
        description="Danbooru tag prediction using SwinV2 architecture.",
    )
    default_repo_id = "SmilingWolf/wd-swinv2-tagger-v3"
    profile = _build_wd_profile(
        TagFilterRecommendation(
            global_threshold=0.265,
            category_thresholds={TagCategory.CHARACTER: 0.75},
        )
    )


class WDConvNextPlugin(WDTaggerBasePlugin):
    identity = ModelIdentity(
        model_id="wd-convnext-v3",
        display_name="WD ConvNeXt Tagger v3",
        description="Danbooru tag prediction using ConvNeXt architecture.",
    )
    default_repo_id = "SmilingWolf/wd-convnext-tagger-v3"
    profile = _build_wd_profile(
        TagFilterRecommendation(
            global_threshold=0.27,
            category_thresholds={TagCategory.CHARACTER: 0.75},
        )
    )


class WDVitPlugin(WDTaggerBasePlugin):
    identity = ModelIdentity(
        model_id="wd-vit-v3",
        display_name="WD ViT Tagger v3",
        description="Danbooru tag prediction using ViT architecture.",
    )
    default_repo_id = "SmilingWolf/wd-vit-tagger-v3"
    profile = _build_wd_profile(
        TagFilterRecommendation(
            global_threshold=0.26,
            category_thresholds={TagCategory.CHARACTER: 0.75},
        )
    )
