from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from autotagger.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from autotagger.params import ParamDef, ParamSchema
from autotagger.results import OutputType, TagEntry, TagResult

logger = logging.getLogger(__name__)


# region Params


class WDTaggerParams(TypedDict, total=False):
    """Typed WD inference params for IDE autocomplete."""

    general_threshold: float
    character_threshold: float
    return_all_scores: bool
    return_character_mapping: bool
    clean_tags: bool


def wd_tagger_params(
    *,
    general_threshold: float | None = None,
    character_threshold: float | None = None,
    return_all_scores: bool | None = None,
    return_character_mapping: bool | None = None,
    clean_tags: bool | None = None,
) -> WDTaggerParams:
    """Build typed WD params while keeping the session dict API unchanged."""
    params: WDTaggerParams = {}
    if general_threshold is not None:
        params["general_threshold"] = general_threshold
    if character_threshold is not None:
        params["character_threshold"] = character_threshold
    if return_all_scores is not None:
        params["return_all_scores"] = return_all_scores
    if return_character_mapping is not None:
        params["return_character_mapping"] = return_character_mapping
    if clean_tags is not None:
        params["clean_tags"] = clean_tags
    return params


# endregion Params


class WDTaggerBasePlugin(ModelPlugin):
    """Shared implementation for WD ONNX taggers."""

    _abstract = True

    output_type = OutputType.TAGS
    supported_backends = [Backend.ONNX, Backend.PYTORCH]

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

    param_schema = ParamSchema(
        [
            ParamDef(
                name="general_threshold",
                type=float,
                default=0.35,
                range=(0.0, 1.0),
                label="General tag threshold",
                description=("Confidence threshold for general tags. Lower = more tags. Typical range: 0.2–0.5."),
            ),
            ParamDef(
                name="character_threshold",
                type=float,
                default=0.85,
                range=(0.0, 1.0),
                label="Character tag threshold",
                description="Confidence threshold for character tags.",
            ),
            ParamDef(
                name="return_all_scores",
                type=bool,
                default=False,
                label="Return all scores",
                description="Include every tag's raw score in the result.",
            ),
            ParamDef(
                name="return_character_mapping",
                type=bool,
                default=False,
                label="Return character mapping",
                description=(
                    "If a character IP mapping file is available, include mapped "
                    "copyright/IP tags for predicted character tags."
                ),
            ),
            ParamDef(
                name="clean_tags",
                type=bool,
                default=False,
                label="Clean tags",
                description=("Normalize underscore-delimited tags into readable text while preserving kaomojis."),
            ),
        ]
    )

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
        params: dict[str, Any],
    ) -> TagResult:
        """Apply thresholding by category and emit TagResult."""
        general_thresh = params["general_threshold"]
        char_thresh = params["character_threshold"]
        return_all = params["return_all_scores"]

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

        predicted: list[TagEntry] = []
        for i in self._general_indices:
            if i < usable_count and scores[i] >= general_thresh:
                predicted.append(TagEntry(tag=self._raw_tag_names[i], score=float(scores[i])))
        for i in self._character_indices:
            if i < usable_count and scores[i] >= char_thresh:
                predicted.append(TagEntry(tag=self._raw_tag_names[i], score=float(scores[i])))

        predicted.sort(key=lambda t: t.score, reverse=True)

        all_entries: list[TagEntry] | None = None
        if return_all:
            all_entries = [TagEntry(tag=self._raw_tag_names[i], score=float(scores[i])) for i in range(usable_count)]
            all_entries.sort(key=lambda t: t.score, reverse=True)

        return TagResult(
            tags=predicted,
            all_scores=all_entries,
        )


# region Model Variants


class WDEva02Plugin(WDTaggerBasePlugin):
    """WD Eva02 Large tagger."""

    model_id = "wd-eva02-large"
    aliases = ["wd-eva02", "eva02-tagger", "wd-eva02-large-tagger", "wd-eva02-v3"]
    display_name = "WD Eva02 Large Tagger"
    description = "Danbooru tag prediction using Eva02 ViT-L architecture."
    default_hf_repo = "SmilingWolf/wd-eva02-large-tagger-v3"


class WDSwinV2Plugin(WDTaggerBasePlugin):
    """WD SwinV2 tagger."""

    model_id = "wd-swinv2"
    aliases = ["wd-swinv2-v3", "wd-swinv2-tagger", "swinv2-tagger"]
    display_name = "WD SwinV2 Tagger"
    description = "Danbooru tag prediction using SwinV2 architecture."
    default_hf_repo = "SmilingWolf/wd-swinv2-tagger-v3"


class WDConvNextPlugin(WDTaggerBasePlugin):
    """WD ConvNeXt tagger (legacy)."""

    model_id = "wd-convnext"
    aliases = ["wd-convnext-v3", "wd-convnext-tagger", "convnext-tagger"]
    display_name = "WD ConvNeXt Tagger"
    description = "Danbooru tag prediction using ConvNeXt architecture."
    default_hf_repo = "SmilingWolf/wd-convnext-tagger-v3"


class WDVitPlugin(WDTaggerBasePlugin):
    """WD ViT tagger (legacy)."""

    model_id = "wd-vit"
    aliases = ["wd-vit-v3", "wd-vit-tagger", "vit-tagger"]
    display_name = "WD ViT Tagger"
    description = "Danbooru tag prediction using ViT architecture."
    default_hf_repo = "SmilingWolf/wd-vit-tagger-v3"


# endregion Model Variants
