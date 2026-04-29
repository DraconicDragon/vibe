# Todo: make this use timm
# models come with timm config files but no preprocess but config alone should be sufficient

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.plugins.shared.tagger_shared import (
    build_entries_for_indices,
    load_tag_metadata,
    normalize_output_scores,
    preprocess_tagger_image,
)
from vibe.result_processors import CharacterIPMapping, CleanTags
from vibe.results import OutputType, TagEntry, TagResult
from vibe.tag_categories import DanbooruTagCategory

logger = logging.getLogger(__name__)


@dataclass
class WDTagResult(TagResult):
    """WD model output schema with explicit category fields."""

    rating: list[TagEntry] = field(default_factory=list)
    general: list[TagEntry] = field(default_factory=list)
    character: list[TagEntry] = field(default_factory=list)

    def categories(self) -> dict[str, list[TagEntry]]:
        return {
            "rating": self.rating,
            "general": self.general,
            "character": self.character,
        }


class WDTaggerBasePlugin(ModelPlugin):
    """Shared implementation for WaifuDiffusion taggers by SmilingWolf."""

    _abstract = True

    output_type = OutputType.TAGS
    supported_backends = [Backend.ONNX, Backend.PYTORCH]
    supported_processors = [CleanTags, CharacterIPMapping]

    required_files = [
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            backends=[Backend.ONNX],
        ),
        FileSpec(
            name="model.safetensors",
            role=FileRole.WEIGHTS,
            backends=[Backend.PYTORCH],
        ),
        FileSpec(
            name="selected_tags.csv",
            role=FileRole.TAG_LIST,
            backends=[],  # empty = needed for all backends
        ),
    ]

    # Image input size for this model
    IMAGE_SIZE = 448
    INPUT_LAYOUT = "NHWC"  # Most WD ONNX exports use NHWC

    # Internal state loaded by load_ancillary()
    _raw_tag_names: list[str]
    _per_tag_thresholds: list[float | None]
    _rating_indices: list[int]
    _general_indices: list[int]
    _character_indices: list[int]

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        """Load tag metadata from selected_tags.csv."""
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

    def preprocess(self, image: Any) -> np.ndarray:
        """Convert image to float32 BGR array for ONNX runtime."""
        return preprocess_tagger_image(
            image,
            image_size=self.IMAGE_SIZE,
            input_layout=self.INPUT_LAYOUT,
            rgb_to_bgr=True,
            normalize_to_unit=False,
        )

    def postprocess(
        self,
        raw_output: Any,
    ) -> WDTagResult:
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

        return WDTagResult(
            rating=rating,
            general=general,
            character=character,
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
    aliases = ["eva02-v3", "eva02-tagger-v3", "wd-eva02-v3", "wd-eva02-large-tagger-v3"]
    display_name = "WD Eva02 Large Tagger"
    description = "Danbooru tag prediction using Eva02 ViT-L architecture."
    default_hf_repo = "SmilingWolf/wd-eva02-large-tagger-v3"


class WDSwinV2Plugin(WDTaggerBasePlugin):
    """WD SwinV2 tagger."""

    model_id = "wd-swinv2-v3"
    aliases = ["swinv2-v3", "swinv2-tagger-v3", "wd-swinv2-tagger-v3"]
    display_name = "WD SwinV2 Tagger"
    description = "Danbooru tag prediction using SwinV2 architecture."
    default_hf_repo = "SmilingWolf/wd-swinv2-tagger-v3"


class WDConvNextPlugin(WDTaggerBasePlugin):
    """WD ConvNeXt tagger."""

    model_id = "wd-convnext-v3"
    aliases = ["convnext-v3", "convnext-tagger-v3", "wd-convnext-tagger-v3", "wd-convnext-tagger-v3"]
    display_name = "WD ConvNeXt Tagger"
    description = "Danbooru tag prediction using ConvNeXt architecture."
    default_hf_repo = "SmilingWolf/wd-convnext-tagger-v3"


class WDVitPlugin(WDTaggerBasePlugin):
    """WD ViT tagger (normal version)."""

    model_id = "wd-vit-v3"
    aliases = ["vit-tagger-v3", "wd-vit-tagger-v3"]
    display_name = "WD ViT Tagger"
    description = "Danbooru tag prediction using ViT architecture."
    default_hf_repo = "SmilingWolf/wd-vit-tagger-v3"


# endregion Model Variants
