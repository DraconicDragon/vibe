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
from vibe.plugins.shared.generic_timm_pipeline import TimmPipelineMixin
from vibe.plugins.shared.tagger_shared import (
    build_categorized_tag_result,
    load_tag_metadata,
    normalize_output_scores,
    preprocess_tagger_image,
    resolve_category_indices,
)
from vibe.result_transforms import CharacterIPMapping, CleanTags, ScoreThresholds
from vibe.results import OutputType, TagResult
from vibe.tag_categories import DANBOORU_CATEGORY_LABELS, TagCategory

logger = logging.getLogger(__name__)


class WDTaggerBasePlugin(TimmPipelineMixin, ModelPlugin):
    """Shared implementation for WaifuDiffusion taggers by SmilingWolf."""

    family_name = "SmilingWolf WD Taggers"

    capabilities = ModelCapabilities(
        output_type=OutputType.TAGS,
        output_categories=(
            TagCategory.RATING,
            TagCategory.GENERAL,
            TagCategory.CHARACTER,
        ),
        transforms=(
            CleanTags,
            ScoreThresholds(
                threshold=0.35,
                category_thresholds={TagCategory.CHARACTER: 0.75},
            ),
            CharacterIPMapping,
        ),
    )

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

    # Internal state loaded by load_ancillary()
    _raw_tag_names: list[str]
    _category_indices: dict[str, list[int]]

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Load tag metadata from selected_tags.csv."""
        csv_path = artifacts.get("tag_list")

        logger.info("Loading tag list from %s", csv_path)
        metadata = load_tag_metadata(csv_path)

        self._raw_tag_names = metadata.raw_tag_names
        self._category_indices = resolve_category_indices(
            metadata.category_indices,
            DANBOORU_CATEGORY_LABELS,
            namespace="danbooru",
        )

    def preprocess(self, image: Any) -> np.ndarray:
        """Convert image to layout expected by the active backend."""
        # NOTE: PyTorch - models expect standard (1, C, H, W) NCHW format
        # NOTE: ONNX - models expect (1, H, W, C) NHWC format
        layout = "NCHW" if self._active_backend == Backend.PYTORCH else "NHWC"

        if self._active_backend == Backend.PYTORCH:
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
        scores = normalize_output_scores(
            raw_output,
            is_logits=True,
            expected_count=len(self._raw_tag_names),
        )
        return build_categorized_tag_result(self._raw_tag_names, scores, self._category_indices)


# region Model Variants


class WDEva02Plugin(WDTaggerBasePlugin):
    """WD Eva02 Large tagger."""

    identity = ModelIdentity(
        model_id="wd-eva02-large-v3",
        display_name="WD Eva02-large Tagger v3",
        description="Danbooru tag prediction using Eva02 ViT-L architecture.",
    )
    default_repo_id = "SmilingWolf/wd-eva02-large-tagger-v3"
    capabilities = WDTaggerBasePlugin.capabilities.with_transforms(
        ScoreThresholds(
            threshold=0.53,
            category_thresholds={TagCategory.CHARACTER: 0.75},
        )
    )


class WDSwinV2Plugin(WDTaggerBasePlugin):
    """WD SwinV2 tagger."""

    identity = ModelIdentity(
        model_id="wd-swinv2-v3",
        display_name="WD SwinV2 Tagger v3",
        description="Danbooru tag prediction using SwinV2 architecture.",
    )
    default_repo_id = "SmilingWolf/wd-swinv2-tagger-v3"
    capabilities = WDTaggerBasePlugin.capabilities.with_transforms(
        ScoreThresholds(
            threshold=0.265,
            category_thresholds={TagCategory.CHARACTER: 0.75},
        )
    )


class WDConvNextPlugin(WDTaggerBasePlugin):
    """WD ConvNeXt tagger."""

    identity = ModelIdentity(
        model_id="wd-convnext-v3",
        display_name="WD ConvNeXt Tagger v3",
        description="Danbooru tag prediction using ConvNeXt architecture.",
    )
    default_repo_id = "SmilingWolf/wd-convnext-tagger-v3"
    capabilities = WDTaggerBasePlugin.capabilities.with_transforms(
        ScoreThresholds(
            threshold=0.27,
            category_thresholds={TagCategory.CHARACTER: 0.75},
        )
    )


class WDVitPlugin(WDTaggerBasePlugin):
    """WD ViT tagger (normal version)."""

    identity = ModelIdentity(
        model_id="wd-vit-v3",
        display_name="WD ViT Tagger v3",
        description="Danbooru tag prediction using ViT architecture.",
    )
    default_repo_id = "SmilingWolf/wd-vit-tagger-v3"
    capabilities = WDTaggerBasePlugin.capabilities.with_transforms(
        ScoreThresholds(
            threshold=0.26,
            category_thresholds={TagCategory.CHARACTER: 0.75},
        )
    )


# endregion Model Variants
