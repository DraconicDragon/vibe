"""JTP-3 & Hydra 3.5 ModelPlugin implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.plugins.shared.tagger_shared import build_entries_for_indices
from vibe.result_processors import CharacterIPMapping, CleanTags
from vibe.results import OutputType, TagResult
from vibe.tag_categories import E621_CATEGORY_LABELS

logger = logging.getLogger(__name__)

# region constants

_PATCH_SIZE: int = 16
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
    if value != _DEFAULT_SEQLEN:
        logger.info("JTP_HYDRA_SEQLEN override: using seqlen=%d (default=%d).", value, _DEFAULT_SEQLEN)
    return value


class JTPHydraBatch(NamedTuple):
    """Preprocessed image data ready for JTP-3 Hydra inference."""

    patches: Tensor  # uint8, shape (max_seq, patch_size*patch_size*3)
    sizes: Tensor  # uint16, shape (2,)


def _preprocess_image_jtp3(image: Image.Image, seqlen: int) -> JTPHydraBatch:
    """Convert a PIL Image to a JTPHydraBatch using the NaFlex patch pipeline."""
    from .image import stack
    from .model import ImageConfig, open_image

    config = ImageConfig({"classifier.background": "black", "classifier.resize": "lanczos"})
    config.max_seqlen = seqlen

    # Load sRGB image as HWC PyTorch tensor [H, W, 3] and ensure writable copy
    img_tensor = torch.from_numpy(np.asarray(open_image(image, config)).copy())

    # Slice image to patches
    patches, sizes = stack([img_tensor], 16, seqlen)

    return JTPHydraBatch(patches.squeeze(0), sizes.squeeze(0))


class JTPHydraBasePlugin(ModelPlugin):
    """Shared implementation for JTP 3 / Hydra taggers by RedRocket."""

    _abstract = True
    family_name = "RedRocket JTP Hydra Taggers"

    output_type = OutputType.TAGS
    supported_backends = (Backend.PYTORCH,)
    supported_processors = (
        CleanTags,
        CharacterIPMapping,
    )

    _raw_tag_names: list[str]
    _indices_by_category: dict[int, list[int]]
    _backend_instance: Any | None
    _jtp3_model: Any
    _device: str
    _seqlen: int

    def configure(self, **kwargs: Any) -> None:
        self._device = str(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self._backend_instance = kwargs.get("backend_instance")

        env_seqlen = _resolve_seqlen()
        if env_seqlen != _DEFAULT_SEQLEN:
            self._seqlen = env_seqlen
        elif "seqlen" in kwargs:
            self._seqlen = int(kwargs["seqlen"])
        else:
            self._seqlen = _DEFAULT_SEQLEN

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        """Load target files and build the architecture natively."""
        # Find model weights
        weights_path = next((p for p in file_map.values() if p.suffix.lower() == ".safetensors"), None)
        if weights_path is None:
            raise RuntimeError("No weights file (.safetensors) resolved in file_map.")

        logger.info("Loading model weights from %s", weights_path)

        # Look for tag/validation CSV files in the resolved file map (JTP-3 fallback)
        csv_path = next((p for p in file_map.values() if p.suffix.lower() == ".csv"), None)
        if csv_path is not None:
            # We found a CSV. JTP-3 calibration is required, so pass the data folder
            legacy_dir = str(csv_path.parent)
            logger.info("JTP-3 tag lists found in %s; legacy metadata enabled.", legacy_dir)
        else:
            # No CSV found. Hydra 3.5 uses built-in safetensors metadata
            legacy_dir = None
            logger.info("No legacy metadata directory provided; loading directly from safetensors.")

        if not hasattr(self, "_device"):
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        from .model import load_model

        # Load the architecture natively using the official loader
        model = load_model(
            str(weights_path),
            logit=True,  # Get raw logits for postprocessing
            legacy_metadata_dir=legacy_dir,  # Points to CSV folder for JTP-3, or None for Hydra 3.5
        )

        # Extract the official labels parsed directly by the model
        self._raw_tag_names = [label.label for label in model.labels]

        # Re-build self._indices_by_category dynamically from the loaded model labels
        from vibe.tag_categories import E621_CATEGORY_LABELS

        cat_to_id = {name: cat_id for cat_id, name in E621_CATEGORY_LABELS.items()}

        self._indices_by_category = {}
        for idx, label in enumerate(model.labels):
            cat_id = cat_to_id.get(label.category, -1)
            self._indices_by_category.setdefault(cat_id, []).append(idx)

        # Safely activate pool inference mode
        if hasattr(model, "attn_pool") and hasattr(model.attn_pool, "inference"):
            model.attn_pool.inference()  # ty:ignore[call-non-callable]

        self._jtp3_model = model

        # Hand the model to the shared backend so run() can use it.
        backend = getattr(self, "_backend_instance", None)
        if backend is not None:
            backend.attach_model(model)
        else:
            logger.warning("JTP-3 load_ancillary: no backend_instance available; model stored on plugin only.")

        logger.info("Model ready: device=%s seqlen=%d labels=%d", self._device, self._seqlen, len(model.labels))

    def preprocess(self, image: Any) -> JTPHydraBatch:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        seqlen = getattr(self, "_seqlen", _DEFAULT_SEQLEN)
        return _preprocess_image_jtp3(image, seqlen)

    def postprocess(self, raw_output: Any) -> TagResult:
        if isinstance(raw_output, np.ndarray):
            scores_np = raw_output.ravel().astype(np.float32)
        elif isinstance(raw_output, Tensor):
            scores_np = raw_output.float().cpu().numpy().ravel()
        else:
            raise TypeError(f"JTP 3 / Hydra postprocess received unexpected type {type(raw_output).__name__}.")

        scores_np = 1.0 / (1.0 + np.exp(-scores_np))

        usable_count = min(len(scores_np), len(self._raw_tag_names))
        if usable_count != len(self._raw_tag_names):
            logger.error("Score length mismatch: got %d scores for %d tags.", len(scores_np), len(self._raw_tag_names))

        result_tags = {
            name: build_entries_for_indices(
                tag_names=self._raw_tag_names,
                indices=indices,
                scores=scores_np,
                usable_count=usable_count,
            )
            for cat_id, name in E621_CATEGORY_LABELS.items()
            if (indices := self._indices_by_category.get(int(cat_id)))
        }

        return TagResult(tags=result_tags)


# region Model Variants


class JTP3Plugin(JTPHydraBasePlugin):
    """JTP-3 Hydra tagger."""

    model_id = "jtp-3"
    display_name = "JTP-3 Hydra"
    description = "E621 tag prediction using JTP-3 Hydra."
    default_hf_repo = "RedRocket/Hydra"

    required_files = (
        FileSpec(
            name="jtp-3-hydra.safetensors",
            role=FileRole.WEIGHTS,
            backends=(Backend.PYTORCH,),
            hf_subdir="models",
        ),
        FileSpec(
            name="jtp-3-hydra-tags.csv",
            role=FileRole.TAG_LIST,
            hf_subdir="data",
        ),
        FileSpec(
            name="jtp-3-hydra-val.csv",
            role=FileRole.TAG_LIST,
            hf_subdir="data",
        ),
    )


class Hydra35Plugin(JTPHydraBasePlugin):
    """Hydra 3.5 Tagger - successor to JTP 3 Hydra."""

    model_id = "hydra-3.5"
    display_name = "Hydra 3.5"
    description = "E621 tag prediction using Hydra 3.5 - successor to JTP 3 Hydra."
    default_hf_repo = "RedRocket/Hydra"

    required_files = (
        FileSpec(
            name="hydra-3.5.safetensors",
            role=FileRole.WEIGHTS,
            backends=(Backend.PYTORCH,),
            hf_subdir="models",
        ),
        # NOTE: hydra3.5 has tag meta inside the safetensors, makes sense since the arch also actually requires the tag meta
        # Leaving this here because in theory i can support both i guess
        # FileSpec(
        #     name="jtp-3.5-hydra-tags.csv",
        #     role=FileRole.TAG_LIST,
        #     hf_subdir="data",
        # ),
    )
