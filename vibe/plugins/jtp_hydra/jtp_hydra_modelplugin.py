"""JTP-3 & Hydra 3.5 ModelPlugin implementation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    ExecutionRequest,
    FileRole,
    ModelCapabilities,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
    RuntimeExecutor,
)
from vibe.backends.runtime.pytorch import PyTorchBackend
from vibe.plugins.shared.tagger_shared import (
    build_categorized_tag_result,
    logits_to_probabilities,
)
from vibe.result_transforms import CharacterIPMapping, CleanTags, ScoreThresholds
from vibe.results import OutputType, TagResult
from vibe.tag_categories import E621_CATEGORY_LABELS, TagCategory

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# region Constants & Helpers

_DEFAULT_SEQLEN: int = 1024
_SEQLEN_MIN: int = 64
_SEQLEN_MAX: int = 2048


def _resolve_seqlen() -> int:
    raw = os.environ.get("JTP_HYDRA_SEQLEN")
    if raw is None:
        return _DEFAULT_SEQLEN
    try:
        value = int(raw)
    except ValueError:
        logger.warning("JTP_HYDRA_SEQLEN=%r is not a valid integer; using default %d.", raw, _DEFAULT_SEQLEN)
        return _DEFAULT_SEQLEN
    if not (_SEQLEN_MIN <= value <= _SEQLEN_MAX):
        logger.warning(
            "JTP_HYDRA_SEQLEN=%d is outside valid range [%d, %d]; using default %d.",
            value,
            _SEQLEN_MIN,
            _SEQLEN_MAX,
            _DEFAULT_SEQLEN,
        )
        return _DEFAULT_SEQLEN
    return value


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


# endregion


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
        transforms=(
            CleanTags,
            CharacterIPMapping,
            ScoreThresholds(threshold=0.35),
        ),
    )

    _raw_tag_names: list[str]
    _category_indices: dict[str, list[int]]
    _seqlen: int = _DEFAULT_SEQLEN
    _preloaded_model: Any | None = None

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Parse tag labels and category indices from weights or CSV metadata."""
        weights_path = artifacts.get("model_pt")
        csv_path = artifacts.get_optional("tag_list")

        legacy_dir = str(csv_path.parent) if csv_path is not None else None

        from .model import load_model

        # Load PyTorch model once to extract metadata and store for build_runtime()
        model = load_model(str(weights_path), logit=True, legacy_metadata_dir=legacy_dir)
        self._raw_tag_names = [label.label for label in model.labels]

        cat_to_name = {cat_id: str(name) for cat_id, name in E621_CATEGORY_LABELS.items()}
        self._category_indices = {}
        for idx, label in enumerate(model.labels):
            cat_name = cat_to_name.get(label.category, str(label.category))
            self._category_indices.setdefault(cat_name, []).append(idx)

        self._seqlen = _resolve_seqlen()
        self._preloaded_model = model

    def build_runtime(self, artifacts: ArtifactMap, request: ExecutionRequest) -> RuntimeExecutor:
        """Build the native JTP-3 / Hydra model graph."""
        if request.backend != Backend.PYTORCH:
            raise ValueError(f"JTP/Hydra models only support PyTorch, got '{request.backend}'.")

        if self._preloaded_model is not None:
            model = self._preloaded_model
            self._preloaded_model = None
        else:
            weights_path = artifacts.get("model_pt")
            csv_path = artifacts.get_optional("tag_list")
            legacy_dir = str(csv_path.parent) if csv_path is not None else None

            from .model import load_model

            model = load_model(str(weights_path), logit=True, legacy_metadata_dir=legacy_dir)

        attn_pool = getattr(model, "attn_pool", None)
        inference_fn = getattr(attn_pool, "inference", None)
        if callable(inference_fn):
            inference_fn()

        backend = PyTorchBackend()
        backend.load(model, request)
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
        probs = logits_to_probabilities(raw_output)
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
                    role=FileRole.TAG_LIST,
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
