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
    build_entries_for_indices,
    load_tag_metadata,
    normalize_output_scores,
    preprocess_tagger_image,
)
from vibe.result_transforms import CharacterIPMapping, CleanTags, ScoreThresholds
from vibe.results import OutputType, TagEntry, TagResult
from vibe.tag_categories import DanbooruTagCategory, TagCategory

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
    _per_tag_thresholds: list[float | None]
    _rating_indices: list[int]
    _general_indices: list[int]
    _character_indices: list[int]

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Load tag metadata from selected_tags.csv."""
        csv_path = artifacts.get("tag_list")

        logger.info("Loading tag list from %s", csv_path)
        metadata = load_tag_metadata(csv_path)

        self._raw_tag_names = metadata.raw_tag_names
        self._per_tag_thresholds = metadata.per_tag_thresholds
        self._rating_indices = metadata.indices_for(int(DanbooruTagCategory.RATING))
        self._general_indices = metadata.indices_for(int(DanbooruTagCategory.GENERAL))
        self._character_indices = metadata.indices_for(int(DanbooruTagCategory.CHARACTER))

        logger.info(
            "Loaded WD tags: total=%d general=%d character=%d rating=%d",
            len(self._raw_tag_names), # num classes
            len(self._general_indices),
            len(self._character_indices),
            len(self._rating_indices),
        )

        config_path = artifacts.get_optional("config")
        if config_path:
            config = self.read_timm_config_json(config_path)
            self.prepare_timm_runtime_preprocess(config)

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
        scores = normalize_output_scores(raw_output)

        usable_count = min(len(scores), len(self._raw_tag_names))
        if usable_count != len(self._raw_tag_names):
            logger.error(
                "Score length mismatch: got %d scores for %d tags.",
                len(scores),
                len(self._raw_tag_names),
            )

        rating = self._entries_for_indices(self._rating_indices, scores, usable_count)
        general = self._entries_for_indices(self._general_indices, scores, usable_count)
        character = self._entries_for_indices(self._character_indices, scores, usable_count)

        return TagResult(
            tags={
                TagCategory.RATING: rating,
                TagCategory.GENERAL: general,
                TagCategory.CHARACTER: character,
            }
        )

    def _entries_for_indices(
        self,
        indices: list[int],
        scores: np.ndarray,
        usable_count: int,
    ) -> list[TagEntry]:
        return build_entries_for_indices(
            tag_names=self._raw_tag_names,
            indices=indices,
            scores=scores,
            usable_count=usable_count,
        )


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
