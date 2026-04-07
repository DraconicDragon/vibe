from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.result_processors import CharacterIPMapping, CleanTags
from vibe.results import OutputType, TagEntry, TagResult

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
            required=True,
            backends=[Backend.ONNX],
        ),
        FileSpec(
            name="model.safetensors",
            role=FileRole.WEIGHTS,
            required=True,
            backends=[Backend.PYTORCH],
        ),
        FileSpec(
            name="selected_tags.csv",
            role=FileRole.TAG_LIST,
            required=True,
            backends=[],  # empty = needed for all backends
        ),
    ]

    # Image input size for this model
    IMAGE_SIZE = 448
    INPUT_LAYOUT = "NHWC"  # Most WD ONNX exports use NHWC

    # Internal state loaded by load_ancillary()
    _raw_tag_names: list[str]
    _rating_indices: list[int]
    _general_indices: list[int]
    _character_indices: list[int]

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        """Load tag metadata"""
        csv_path = file_map["selected_tags.csv"]

        logger.info("Loading tag list from %s", csv_path)
        self._raw_tag_names = []
        self._rating_indices = []
        self._general_indices = []
        self._character_indices = []

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                raw_name = row.get("name", "")
                self._raw_tag_names.append(raw_name)

                try:
                    category = int(row.get("category", "0"))
                except ValueError:
                    category = 0

                if category == 9:
                    self._rating_indices.append(idx)
                elif category == 4:
                    self._character_indices.append(idx)
                elif category == 0:
                    self._general_indices.append(idx)

        logger.info(
            "Loaded WD tags: total=%d general=%d character=%d rating=%d",
            len(self._raw_tag_names),
            len(self._general_indices),
            len(self._character_indices),
            len(self._rating_indices),
        )

    def preprocess(self, image: Any) -> np.ndarray:
        """Convert image to float32 BGR array for ONNX runtime."""
        from PIL import Image

        # Ensure PIL image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert("RGB")

        width, height = image.size
        if width != height:
            size = max(width, height)
            squared = Image.new("RGB", (size, size), (255, 255, 255))
            squared.paste(image, ((size - width) // 2, (size - height) // 2))
            image = squared

        image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.Resampling.BICUBIC)

        arr = np.asarray(image, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR

        if self.INPUT_LAYOUT == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))

        arr = np.expand_dims(arr, axis=0).astype(np.float32)
        return arr

    def postprocess(
        self,
        raw_output: Any,
    ) -> WDTagResult:
        """Return full scored output grouped by WD tag category."""
        scores = np.asarray(raw_output)
        if scores.ndim > 1:
            scores = np.squeeze(scores, axis=0)
        scores = scores.astype(np.float32)

        # Most exports already return probabilities, but apply sigmoid for logits.
        if np.min(scores) < 0.0 or np.max(scores) > 1.0:
            scores = 1.0 / (1.0 + np.exp(-scores))

        usable_count = min(len(scores), len(self._raw_tag_names))
        if usable_count != len(self._raw_tag_names):
            logger.warning(
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

    # todo: move to mixin or some other shared utility file since this is pretty universal
    def _entries_for_indices(
        self,
        indices: list[int],
        scores: np.ndarray,
        usable_count: int,
    ) -> list[TagEntry]:
        entries = [TagEntry(tag=self._raw_tag_names[i], score=float(scores[i])) for i in indices if i < usable_count]
        entries.sort(key=lambda item: item.score, reverse=True)
        return entries


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
