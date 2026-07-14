"""Taggerine ModelPlugin implementation.

DINOv3 ViT-H/16+ Tagger by lodestones.
Uses an unpadded, aspect-ratio preserving approach.
"""

from __future__ import annotations
from vibe import ScoreThresholds

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.plugins.shared.tagger_shared import build_entries_for_indices
from vibe.result_processors import CharacterIPMapping, CleanTags
from vibe.results import OutputType, TagResult
from vibe.tag_categories import E621_CATEGORY_LABELS

logger = logging.getLogger(__name__)

# Taggerine constants
_MAX_SIZE = 1024
_PATCH_SIZE = 16
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _snap(x: int, m: int) -> int:
    return max(m, (x // m) * m)


class TaggerinePlugin(ModelPlugin):
    model_id = "taggerine"
    display_name = "Taggerine (DINOv3)"
    description = "E621 + Danbooru tagger by Lodestone."
    default_hf_repo = "lodestones/taggerine"

    output_type = OutputType.TAGS
    supported_backends = (Backend.PYTORCH,)
    supported_processors = (
        CleanTags,
        CharacterIPMapping,
        ScoreThresholds,
    )

    required_files = (
        FileSpec(
            name="tagger_proto.safetensors",
            role=FileRole.WEIGHTS,
            backends=(Backend.PYTORCH,),
        ),
        FileSpec(
            name="tagger_vocab_with_categories_and_alias_updated.json",
            role=FileRole.TAG_LIST,
        ),
    )

    _raw_tag_names: list[str]
    _indices_by_category: dict[int, list[int]]

    def configure(self, **kwargs: Any) -> None:
        self._device = str(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self._backend_instance = kwargs.get("backend_instance")

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        # 1. Parse Vocab JSON
        vocab_path = next((p for p in file_map.values() if p.suffix.lower() == ".json"), None)
        if not vocab_path:
            raise RuntimeError("Vocabulary JSON file not resolved in file_map.")

        with vocab_path.open("r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self._raw_tag_names = vocab_data.get("idx2tag", [])
        self._indices_by_category = {}

        # Look for tag2category mapping first
        if "tag2category" in vocab_data:
            tag2category = vocab_data["tag2category"]
            for idx, tag in enumerate(self._raw_tag_names):
                cat_id = tag2category.get(tag, 0)  # Default to 0 (general) if tag isn't mapped
                self._indices_by_category.setdefault(int(cat_id), []).append(idx)
        elif "idx2category" in vocab_data:
            for idx, cat_id in enumerate(vocab_data["idx2category"]):
                self._indices_by_category.setdefault(int(cat_id), []).append(idx)
        else:
            # Fallback if categories are not provided: put everything in general (0)
            self._indices_by_category[0] = list(range(len(self._raw_tag_names)))

        logger.info("Taggerine vocab loaded: %d tags", len(self._raw_tag_names))

        # 2. Build and Load Model Architecture
        weights_path = next((p for p in file_map.values() if p.suffix.lower() == ".safetensors"), None)
        if not weights_path:
            raise RuntimeError("Weights .safetensors file not resolved in file_map.")

        from .model import DINOv3Tagger, _build_head_from_checkpoint, split_and_clean_state_dict

        # Reuse the state dict the backend already loaded to CPU, instead of
        # re-reading the same safetensors file from disk a second time.
        backend = getattr(self, "_backend_instance", None)
        sd = getattr(backend, "raw", None)
        if not isinstance(sd, dict):
            from safetensors.torch import load_file

            logger.info("Loading Taggerine weights from %s", weights_path)
            sd = load_file(weights_path, device="cpu")
        else:
            logger.info("Reusing Taggerine weights already loaded in CPU RAM")

        backbone_sd, head_sd = split_and_clean_state_dict(sd)

        if not head_sd:
            raise RuntimeError("Checkpoint contains no non-backbone keys — cannot build head.")

        model = DINOv3Tagger()
        head_module, head_sd_remapped = _build_head_from_checkpoint(
            head_sd, in_dim=6400, num_tags=len(self._raw_tag_names)
        )
        model.head = head_module

        model.backbone.load_state_dict(backbone_sd, strict=True)
        model.head.load_state_dict(head_sd_remapped, strict=True)

        model.eval()

        backend = getattr(self, "_backend_instance", None)
        if backend is not None:
            backend.attach_model(model)

        logger.info("Taggerine model ready: device=%s", self._device)

    def preprocess(self, image: Any) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        img = image.convert("RGB")
        w, h = img.size

        # Target long-edge (snapped to patch multiple)
        long_edge = max(w, h)
        target_long = _snap(min(long_edge, _MAX_SIZE), _PATCH_SIZE)
        scale = target_long / long_edge

        new_w = _snap(max(_PATCH_SIZE, round(w * scale)), _PATCH_SIZE)
        new_h = _snap(max(_PATCH_SIZE, round(h * scale)), _PATCH_SIZE)

        # Standard PIL Lanczos resize
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert to numpy, scale to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize (ImageNet)
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC to CHW
        arr = np.transpose(arr, (2, 0, 1))

        # Vibe expects all preprocessed tensors to have a batch dimension of 1
        return torch.from_numpy(arr).unsqueeze(0)

    def postprocess(self, raw_output: Any) -> TagResult:
        if isinstance(raw_output, np.ndarray):
            scores_np = raw_output.ravel().astype(np.float32)
        elif isinstance(raw_output, torch.Tensor):
            scores_np = raw_output.float().cpu().numpy().ravel()
        else:
            raise TypeError(f"Postprocess received unexpected type {type(raw_output).__name__}.")

        # Taggerine outputs raw logits, we apply Sigmoid
        scores_np = 1.0 / (1.0 + np.exp(-scores_np))

        usable_count = min(len(scores_np), len(self._raw_tag_names))

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
