from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.plugins.shared.generic_timm_pipeline import TimmPipelineMixin
from vibe.plugins.shared.tagger_shared import (
    build_entries_for_indices,
    load_tag_metadata,
    normalize_output_scores,
    preprocess_tagger_image,
)
from vibe.result_processors import CharacterIPMapping, CleanTags, ScoreThresholds
from vibe.results import OutputType, TagEntry, TagResult
from vibe.tag_categories import DanbooruTagCategory

logger = logging.getLogger(__name__)


class WDTaggerBasePlugin(TimmPipelineMixin, ModelPlugin):
    """Shared implementation for WaifuDiffusion taggers by SmilingWolf."""

    _abstract = True
    family_name = "SmilingWolf WD Taggers"

    output_type = OutputType.TAGS
    supported_backends = (
        Backend.ONNX,
        Backend.PYTORCH,
    )
    supported_processors = (
        CleanTags,
        ScoreThresholds,
        CharacterIPMapping,
    )

    required_files = (
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            backends=(Backend.ONNX,),
        ),
        FileSpec(
            name="model.safetensors",
            role=FileRole.WEIGHTS,
            backends=(Backend.PYTORCH,),
        ),
        FileSpec(
            name="config.json",
            role=FileRole.CONFIG,
            backends=(Backend.PYTORCH,),
        ),
        FileSpec(
            name="selected_tags.csv",
            role=FileRole.TAG_LIST,
            backends=(),  # empty = needed for all backends
        ),
    )

    # Image input size for this model (most WD models use 448x448)
    IMAGE_SIZE = 448

    # Internal state loaded by load_ancillary()
    _raw_tag_names: list[str]
    _per_tag_thresholds: list[float | None]
    _rating_indices: list[int]
    _general_indices: list[int]
    _character_indices: list[int]

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        """Load tag metadata from selected_tags.csv and handle PyTorch bootstrapping."""
        csv_path = file_map["selected_tags.csv"]

        logger.info("Loading tag list from %s", csv_path)
        metadata = load_tag_metadata(csv_path)

        self._raw_tag_names = metadata.raw_tag_names
        self._per_tag_thresholds = metadata.per_tag_thresholds
        self._rating_indices = metadata.indices_for(int(DanbooruTagCategory.RATING))
        self._general_indices = metadata.indices_for(int(DanbooruTagCategory.GENERAL))
        self._character_indices = metadata.indices_for(int(DanbooruTagCategory.CHARACTER))

        logger.info(
            "Loaded WD tags: total=%d general=%d character=%d rating=%d",
            len(self._raw_tag_names),
            len(self._general_indices),
            len(self._character_indices),
            len(self._rating_indices),
        )

        # If using PyTorch, reconstruct the timm architecture using the loaded config.json
        if self._backend == Backend.PYTORCH:
            config_path = file_map.get("config.json")
            config = self.read_timm_config_json(config_path) if config_path else {}

            # Reconstruct the PyTorch model architecture and load the state dict
            self.maybe_prepare_timm_pytorch_model(config=config, num_classes=len(self._raw_tag_names))

    def preprocess(self, image: Any) -> np.ndarray:
        """Convert image to layout expected by the active backend."""
        # Under PyTorch, models expect standard (1, C, H, W) NCHW format
        # Under ONNX, these models expect (1, H, W, C) NHWC format
        layout = "NCHW" if self._backend == Backend.PYTORCH else "NHWC"

        if self._backend == Backend.PYTORCH:
            # PyTorch expects BGR normalized to [-1, 1] range: (x - 127.5) / 127.5
            return preprocess_tagger_image(
                image,
                image_size=self.IMAGE_SIZE,
                input_layout=layout,
                rgb_to_bgr=True,
                normalize_to_unit=True,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
        else:
            # ONNX expects unnormalized, raw BGR [0, 255] float32
            arr = preprocess_tagger_image(
                image,
                image_size=self.IMAGE_SIZE,
                input_layout=layout,
                rgb_to_bgr=True,
                normalize_to_unit=False,
            )
            return np.ascontiguousarray(arr)

    def postprocess(
        self,
        raw_output: Any,
    ) -> TagResult:
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
                "rating": rating,
                "general": general,
                "character": character,
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

    model_id = "wd-eva02-large-v3"
    aliases = (
        "eva02-v3",
        "eva02-tagger-v3",
        "wd-eva02-v3",
        "wd-eva02-large-tagger-v3",
    )
    display_name = "WD Eva02-large Tagger v3"
    description = "Danbooru tag prediction using Eva02 ViT-L architecture."
    default_hf_repo = "SmilingWolf/wd-eva02-large-tagger-v3"


class WDSwinV2Plugin(WDTaggerBasePlugin):
    """WD SwinV2 tagger."""

    model_id = "wd-swinv2-v3"
    aliases = (
        "swinv2-v3",
        "swinv2-tagger-v3",
        "wd-swinv2-tagger-v3",
    )
    display_name = "WD SwinV2 Tagger v3"
    description = "Danbooru tag prediction using SwinV2 architecture."
    default_hf_repo = "SmilingWolf/wd-swinv2-tagger-v3"


class WDConvNextPlugin(WDTaggerBasePlugin):
    """WD ConvNeXt tagger."""

    model_id = "wd-convnext-v3"
    aliases = (
        "convnext-v3",
        "convnext-tagger-v3",
        "wd-convnext-tagger-v3",
    )
    display_name = "WD ConvNeXt Tagger v3"
    description = "Danbooru tag prediction using ConvNeXt architecture."
    default_hf_repo = "SmilingWolf/wd-convnext-tagger-v3"


class WDVitPlugin(WDTaggerBasePlugin):
    """WD ViT tagger (normal version)."""

    model_id = "wd-vit-v3"
    aliases = (
        "vit-tagger-v3",
        "wd-vit-tagger-v3",
    )
    display_name = "WD ViT Tagger v3"
    description = "Danbooru tag prediction using ViT architecture."
    default_hf_repo = "SmilingWolf/wd-vit-tagger-v3"


# endregion Model Variants
