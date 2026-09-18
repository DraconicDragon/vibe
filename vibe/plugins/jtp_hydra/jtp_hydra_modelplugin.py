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
from pydantic import BaseModel, Field

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
from vibe.contracts import LabelCatalogProvider, ThresholdProvider
from vibe.metadata import LabelCatalog, LabelInfo, TagFilterRecommendation, ThresholdTable
from vibe.model_profiles import build_tagger_profile
from vibe.plugins.shared.tagger_shared import (
    build_categorized_tag_result,
    normalize_output_scores,
    resolve_category_name,
)
from vibe.results import TagResult
from vibe.settings import InferenceRequest, SettingGroupSpec
from vibe.tag_categories import E621_CATEGORY_LABELS, TagCategory

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class JTPHydraSettings(BaseModel):
    """Visual token budget configuration for JTP / Hydra models."""

    seqlen: int = Field(
        default=1024,
        ge=64,
        le=2048,
        multiple_of=64,
        title="Sequence Length",
        description="Maximum visual tokens used to represent an image. Higher values preserve fine detail but use more VRAM.",
        json_schema_extra={"scope": "infer"},
    )


class JTPHydraBatch(NamedTuple):
    """Preprocessed image data ready for JTP-3 Hydra inference."""

    patches: Tensor  # uint8; shape (batch_size, max_seq, patch_size*patch_size*3)
    sizes: Tensor  # uint16; shape (batch_size, 2)


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

    # Slice image to patches (returns batched [1, S, D] and [1, 2])
    patches, sizes = stack([img_tensor], 16, seqlen)

    return JTPHydraBatch(patches, sizes)


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
            # Only record thresholds if the tag achieves a positive F1 score
            if f1 > 0.0 and (tag not in tag_best_f1 or f1 > tag_best_f1[tag]):
                tag_best_f1[tag] = f1
                tag_best_thresh[tag] = thresh

    return tag_best_thresh


class JTPHydraBasePlugin(ModelPlugin):
    """Shared implementation for JTP 3 / Hydra taggers by RedRocket."""

    family_name = "RedRocket JTP Hydra Taggers"

    profile = build_tagger_profile()
    implements = (LabelCatalogProvider, ThresholdProvider)

    settings = (
        SettingGroupSpec.from_model(
            JTPHydraSettings,
            id="jtp_hydra",
            display_name="Sequence Length",
            description="Visual token sequence budget for NaFlex patch stacker.",
            recommended=JTPHydraSettings(seqlen=1024),
        ),
    )

    catalog: LabelCatalog
    thresholds: ThresholdTable
    _raw_tag_names: list[str]
    _category_indices: dict[str, list[int]]

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Parse tag labels, category indices, and optimal thresholds directly from safetensors metadata or validation CSV."""
        weights_path = artifacts.get("model_pt")
        val_csv_path = artifacts.get_optional("val_csv")

        self._raw_tag_names = []
        self._category_indices = {}
        raw_tag_categories: list[str] = []
        tag_thresholds_map: dict[str, float] = {}
        thresholds_source = "empty"
        label_infos: list[LabelInfo] = []
        seen_categories: set[str] = set()

        # Parse tag labels and embedded validation tensor directly via safe_open (0 MB RAM load)
        try:
            from safetensors import safe_open

            with safe_open(str(weights_path), framework="numpy") as f:
                meta = f.metadata() or {}

                if "classifier.labels" in meta:
                    for idx, line in enumerate(meta["classifier.labels"].splitlines()):
                        line = line.strip()
                        if not line:
                            tag = f"unknown_{idx}"
                            cat_name = "unknown"
                        else:
                            # Split into at most 3 parts: [tag, category, implications]
                            parts = line.split(maxsplit=2)
                            tag = parts[0]
                            cat_raw = parts[1] if len(parts) > 1 else "general"
                            cat_name = resolve_category_name(cat_raw, E621_CATEGORY_LABELS, namespace="e621")

                        self._raw_tag_names.append(tag)
                        raw_tag_categories.append(cat_name)
                        self._category_indices.setdefault(cat_name, []).append(idx)
                        seen_categories.add(cat_name)

                # Extract embedded 'validation' tensor (Hydra 3.5 format)
                keys = f.keys()
                if "validation" in keys:
                    val_data = f.get_tensor("validation")  # Shape: (8886, 99, 4)
                    tp = val_data[:, :, 0]
                    fp = val_data[:, :, 1]
                    fn = val_data[:, :, 3]

                    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)

                    # Guard against argmax(zeros) trap on rare tags with zero true positives
                    max_f1 = np.max(f1, axis=1)
                    valid_mask = max_f1 > 0.0

                    best_bins = np.argmax(f1, axis=1)
                    steps = np.linspace(0.01, 0.99, 99)
                    best_thresholds = steps[best_bins]

                    if self._raw_tag_names:
                        tag_thresholds_map = {
                            tag: float(thresh)
                            for tag, thresh, is_valid in zip(
                                self._raw_tag_names, best_thresholds, valid_mask, strict=False
                            )
                            if is_valid
                        }
                    thresholds_source = "embedded"
                    logger.info(
                        "Extracted %d calibrated thresholds from embedded validation tensor for %s (omitted %d untrainable tags)",
                        len(tag_thresholds_map),
                        self.identity.model_id,
                        len(self._raw_tag_names) - len(tag_thresholds_map),
                    )

        except Exception as exc:
            logger.debug("Failed to extract metadata/tensors directly via safe_open: %s", exc)

        # Fallback to validation CSV if embedded tensor is not present (JTP-3 format)
        if not tag_thresholds_map and val_csv_path is not None:
            tag_thresholds_map = _parse_jtp_val_csv(val_csv_path)
            thresholds_source = "val_csv"
            logger.info("Parsed optimal thresholds from val_csv for %s", self.identity.model_id)

        for idx, tag in enumerate(self._raw_tag_names):
            cat_name = raw_tag_categories[idx] if idx < len(raw_tag_categories) else "unknown"
            thresh = tag_thresholds_map.get(tag)
            label_infos.append(LabelInfo(index=idx, name=tag, category=cat_name, threshold=thresh))

        self.catalog = LabelCatalog(labels=tuple(label_infos), categories=tuple(seen_categories))
        self.thresholds = ThresholdTable(values=tag_thresholds_map, source=thresholds_source)

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        """Build the native JTP-3 / Hydra model graph."""
        if plan.backend != Backend.PYTORCH:
            raise ValueError(f"JTP/Hydra models only support PyTorch, got '{plan.backend}'.")

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
            patches = torch.cat([item.patches for item in samples], dim=0)
            sizes = torch.cat([item.sizes for item in samples], dim=0)
            return JTPHydraBatch(patches, sizes)
        except Exception as exc:
            raise ValueError(f"Failed to collate JTPHydraBatch: {exc}") from exc

    def preprocess(self, image: Any, request: InferenceRequest | None = None) -> JTPHydraBatch:
        settings = request.get(JTPHydraSettings) if request is not None else None
        seqlen = settings.seqlen if settings is not None else 1024
        return _preprocess_image_jtp3(image, seqlen)

    def postprocess(self, raw_output: Any) -> TagResult:
        probs = normalize_output_scores(raw_output, is_logits=True, expected_count=len(self._raw_tag_names))
        return build_categorized_tag_result(self._raw_tag_names, probs, self._category_indices)


class JTP3Plugin(JTPHydraBasePlugin):
    """JTP-3 Hydra tagger."""

    identity = ModelIdentity(
        model_id="jtp-3",
        display_name="JTP-3 Hydra",
        description="E621 tag prediction using JTP-3 Hydra.",
    )
    default_repo_id = "RedRocket/Hydra"

    profile = build_tagger_profile(
        # thresholds are calculated from validation data f1.0@0.1
        recommended_filter=TagFilterRecommendation(
            global_threshold=0.75,  # 0.7506
            category_thresholds={
                TagCategory.CHARACTER: 0.71,  # 0.7098
                TagCategory.COPYRIGHT: 0.78,  # 0.7800
                TagCategory.GENERAL: 0.74,  # 0.7402
                TagCategory.LORE: 0.64,  # 0.6397
                TagCategory.META: 0.76,  # 0.7607
                TagCategory.SPECIES: 0.77,  # 0.7705
            },
        ),
    )

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

    # thresholds are calculated from validation data f1.0@0.1
    profile = build_tagger_profile(
        recommended_filter=TagFilterRecommendation(
            global_threshold=0.72,  # 0.7202
            category_thresholds={
                TagCategory.CHARACTER: 0.65,  # 0.6496
                TagCategory.COPYRIGHT: 0.71,  # 0.7098
                #TagCategory.GENERAL: 0.72,  # 0.7202 # commented in since already exists in form of global threshold
                TagCategory.LORE: 0.59,  # 0.5898
                #TagCategory.META: 0.72,  # 0.7202
                TagCategory.SPECIES: 0.74,  # 0.7402
            },
        ),
    )

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
