"""
JTP-3 & Hydra 3.5 ModelPlugin implementation.
Models are based on SigLip2 So400M NaFlex. Recommended seq_len: 1024
Patch size is 16, so 1024 tokens = ~0.25 MP which is for reference a 512x512 image.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    ExecutionPlan,
    FileRole,
    ModelCapabilities,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
    PluginOptionSpec,
    RuntimeExecutor,
)
from vibe.backends.runtime.pytorch import PyTorchBackend
from vibe.plugins.shared.tagger_shared import (
    build_categorized_tag_result,
    normalize_output_scores,
)
from vibe.result_transforms import (
    CharacterIPMapping,
    CleanTags,
    PluginData,
    ScoreThresholds,
    TagLevelThresholds,
    TagThresholds,
)
from vibe.results import OutputType, TagResult
from vibe.tag_categories import E621_CATEGORY_LABELS, TagCategory

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# region Constants & Helpers


class JTPHydraBatch(NamedTuple):
    """Preprocessed image data ready for JTP-3 Hydra inference."""

    patches: Tensor  # uint8; shape (max_seq, patch_size*patch_size*3)
    sizes: Tensor  # uint16; shape (2,)


def _preprocess_image_jtp3(image: Any, seqlen: int) -> JTPHydraBatch:
    """Convert an Image to a JTPHydraBatch using the NaFlex patch pipeline."""
    import torch
    from PIL import Image

    from .image import stack
    from .model import ImageConfig, open_image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    config = ImageConfig({"classifier.background": "black", "classifier.resize": "lanczos"})
    config.max_seqlen = seqlen

    # Load sRGB image as HWC PyTorch tensor [H, W, 3] and ensure writable copy
    img_tensor = torch.from_numpy(np.asarray(open_image(image, config)).copy())

    # Slice image to patches
    patches, sizes = stack([img_tensor], 16, seqlen)

    return JTPHydraBatch(patches.squeeze(0), sizes.squeeze(0))


def _parse_jtp_val_csv(csv_path: Path) -> dict[str, float]:
    """Parse JTP3 validation CSV and compute the optimal F1 threshold per tag."""
    tag_best_f1: dict[str, float] = {}
    tag_best_thresh: dict[str, float] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("tag", "").strip()
            if not tag:
                continue
            thresh = float(row.get("threshold", 0.0))
            tp = float(row.get("tp", 0.0))
            fp = float(row.get("fp", 0.0))
            fn = float(row.get("fn", 0.0))

            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
            if tag not in tag_best_f1 or f1 > tag_best_f1[tag]:
                tag_best_f1[tag] = f1
                tag_best_thresh[tag] = thresh

    return tag_best_thresh


# endregion Constants & Helpers


# region Abstract Base Plugin


class JTPHydraBasePlugin(ModelPlugin):
    """Shared implementation for JTP 3 / Hydra taggers by RedRocket."""

    family_name = "RedRocket JTP Hydra Taggers"

    capabilities = ModelCapabilities(
        output_type=OutputType.TAGS,
        output_categories=(
            TagCategory.GENERAL,
            TagCategory.ARTIST,
            TagCategory.CONTRIBUTOR,
            TagCategory.COPYRIGHT,
            TagCategory.CHARACTER,
            TagCategory.SPECIES,
            TagCategory.META,
            TagCategory.LORE,
        ),
        options=(
            PluginOptionSpec(
                key="jtp_hydra_seqlen",
                display_name="Sequence Length",
                type=int,
                default=1024,
                min_val=64,
                max_val=2048,
                description="Maximum visual tokens used to represent an image. Higher values preserve fine detail but use more VRAM.",
            ),
        ),
        transforms=(
            CleanTags,
            CharacterIPMapping,
            ScoreThresholds(threshold=0.35),
            TagLevelThresholds,
        ),
    )

    _raw_tag_names: list[str]
    _category_indices: dict[str, list[int]]
    _tag_thresholds: dict[str, float]
    _seqlen: int = 1024
    _preloaded_model: Any | None = None

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Parse tag labels, category indices, and optimal thresholds directly from weights or optional validation CSV."""
        weights_path = artifacts.get("model_pt")
        val_csv_path = artifacts.get_optional("val_csv")

        from .model import load_model

        # Load PyTorch model to extract embedded labels
        model = load_model(str(weights_path), logit=True)
        self._raw_tag_names = [label.label for label in model.labels]

        cat_to_name = {cat_id: str(name) for cat_id, name in E621_CATEGORY_LABELS.items()}
        self._category_indices = {}
        for idx, label in enumerate(model.labels):
            cat_name = cat_to_name.get(label.category, str(label.category))
            self._category_indices.setdefault(cat_name, []).append(idx)

        # Parse optimal tag thresholds
        self._tag_thresholds = {}

        # 1. Check for embedded 'validation' tensor in weights (Hydra 3.5 format)
        try:
            from safetensors import safe_open

            with safe_open(str(weights_path), framework="numpy") as f:
                keys = f.keys()
                if "validation" in keys:
                    val_data = f.get_tensor("validation")  # Shape: (8886, 99, 4)
                    tp = val_data[:, :, 0]
                    fp = val_data[:, :, 1]
                    fn = val_data[:, :, 3]

                    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
                    best_bins = np.argmax(f1, axis=1)
                    steps = np.linspace(0.01, 0.99, 99)
                    best_thresholds = steps[best_bins]

                    self._tag_thresholds = {
                        tag: float(thresh) for tag, thresh in zip(self._raw_tag_names, best_thresholds, strict=False)
                    }
                    logger.info(
                        "Extracted optimal thresholds from embedded validation tensor for %s", self.identity.model_id
                    )
        except Exception as exc:
            logger.debug("Failed to extract validation tensor from safetensors: %s", exc)

        # 2. Fall back to parsing optional val_csv if no embedded tensor was present
        if not self._tag_thresholds and val_csv_path is not None:
            self._tag_thresholds = _parse_jtp_val_csv(val_csv_path)
            logger.info("Parsed optimal thresholds from val_csv for %s", self.identity.model_id)

        self._seqlen = int(self.get_option("jtp_hydra_seqlen"))
        self._preloaded_model = model

    def provide_transform_data(self) -> tuple[PluginData, ...]:
        if self._tag_thresholds:
            return (TagThresholds(values=self._tag_thresholds),)
        return ()

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        """Build the native JTP-3 / Hydra model graph."""
        if plan.backend != Backend.PYTORCH:
            raise ValueError(f"JTP/Hydra models only support PyTorch, got '{plan.backend}'.")

        if self._preloaded_model is not None:
            model = self._preloaded_model
            self._preloaded_model = None
        else:
            weights_path = artifacts.get("model_pt")

            from .model import load_model

            model = load_model(str(weights_path), logit=True)

        attn_pool = getattr(model, "attn_pool", None)
        inference_fn = getattr(attn_pool, "inference", None)
        if callable(inference_fn):
            inference_fn()

        backend = PyTorchBackend()
        backend.load(model, plan)
        return backend

    def collate_batch(self, samples: list[Any]) -> Any:
        """Custom collator for JTPHydraBatch named tuples."""
        import torch

        try:
            patches = torch.stack([item.patches for item in samples], dim=0)
            sizes = torch.stack([item.sizes for item in samples], dim=0)
            return JTPHydraBatch(patches, sizes)
        except Exception as exc:
            raise ValueError(f"Failed to collate JTPHydraBatch: {exc}") from exc

    def preprocess(self, image: Any) -> JTPHydraBatch:
        return _preprocess_image_jtp3(image, self._seqlen)

    def postprocess(self, raw_output: Any) -> TagResult:
        probs = normalize_output_scores(raw_output, expected_count=len(self._raw_tag_names))
        return build_categorized_tag_result(self._raw_tag_names, probs, self._category_indices)


# endregion


# region Concrete Plugins


class JTP3Plugin(JTPHydraBasePlugin):
    """JTP-3 Hydra tagger."""

    identity = ModelIdentity(
        model_id="jtp-3",
        display_name="JTP-3 Hydra",
        description="E621 tag prediction using JTP-3 Hydra.",
    )
    default_repo_id = "RedRocket/Hydra"

    variants = (
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(
                    id="model_pt",
                    name="jtp-3-hydra.safetensors",
                    role=FileRole.WEIGHTS,
                    hf_subdir="models",
                ),
                ArtifactSpec(
                    id="val_csv",
                    name="jtp-3-hydra-val.csv",
                    role=FileRole.MAPPING,
                    hf_subdir="data",
                    required=False,
                ),
            ),
        ),
    )


class Hydra35Plugin(JTPHydraBasePlugin):
    """Hydra 3.5 Tagger - successor to JTP 3 Hydra."""

    identity = ModelIdentity(
        model_id="hydra-3.5",
        display_name="Hydra 3.5",
        description="E621 tag prediction using Hydra 3.5 - successor to JTP 3 Hydra.",
    )
    default_repo_id = "RedRocket/Hydra"

    variants = (
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(
                    id="model_pt",
                    name="hydra-3.5.safetensors",
                    role=FileRole.WEIGHTS,
                    hf_subdir="models",
                ),
            ),
        ),
    )


# endregion
