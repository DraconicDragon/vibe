"""JTP-3 Hydra ModelPlugin implementation.

E621 tagger by RedRocket, based on SigLIP2-so400m-patch16-naflex with a NaFlex
ViT backbone, cross-attention HydraPool classifier head with learned per-tag
queries, SwiGLU feedforward, and per-tag SwiGLU output heads.

Architecture notes:
  - Input is a variable-length sequence of 16x16 patches (NaFlex / naflex).
  - Sequence length is dynamic: 64–2048 patches, recommended/default 1024.
  - The model was primarily trained at seqlen=1024; deviating reduces accuracy.
  - seqlen is exposed via JTP3_SEQLEN env var for experimental use only.
  - Extension (LoRA-like per-label) support exists in the architecture but is
    intentionally not wired up here — a future addition.

Env vars (experimental, not part of the stable API):
  JTP3_SEQLEN   int, 64–2048  NaFlex sequence length  (default: 1024)
"""

from __future__ import annotations

import csv
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
from vibe.tag_categories import E621TagCategory, E621_CATEGORY_LABELS

logger = logging.getLogger(__name__)

# region constants

#: NaFlex patch size. Fixed by the SigLIP2 backbone; do not change.
_PATCH_SIZE: int = 16

#: Default NaFlex sequence length (number of patches).
#: Model page recommends 1024; higher/lower may reduce accuracy somewhat.
#: Exposed via JTP3_SEQLEN env var for experimental tuning only.
_DEFAULT_SEQLEN: int = 1024
_SEQLEN_MIN: int = 64
_SEQLEN_MAX: int = 2048


def _resolve_seqlen() -> int:
    raw = os.environ.get("JTP3_SEQLEN")
    if raw is None:
        return _DEFAULT_SEQLEN
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "JTP3_SEQLEN=%r is not a valid integer; using default %d.",
            raw,
            _DEFAULT_SEQLEN,
        )
        return _DEFAULT_SEQLEN
    if not (_SEQLEN_MIN <= value <= _SEQLEN_MAX):
        logger.warning(
            "JTP3_SEQLEN=%d is outside valid range [%d, %d]; using default %d.",
            value,
            _SEQLEN_MIN,
            _SEQLEN_MAX,
            _DEFAULT_SEQLEN,
        )
        return _DEFAULT_SEQLEN
    if value != _DEFAULT_SEQLEN:
        logger.info(
            "JTP3_SEQLEN override: using seqlen=%d (default=%d). "
            "Note: accuracy may be reduced at non-default sequence lengths.",
            value,
            _DEFAULT_SEQLEN,
        )
    return value


# region CSV loading helpers


class _JTP3TagMetadata(NamedTuple):
    raw_tag_names: list[str]
    indices_by_category: dict[int, list[int]]


def _load_jtp3_tag_csv(path: Path) -> _JTP3TagMetadata:
    """Parse jtp-3-hydra-tags.csv into tag names and per-category index lists.

    Expected columns: tag, category[, implications, ...]
    The 'category' column holds an integer matching E621TagCategory values.
    """
    raw_tag_names: list[str] = []
    indices_by_category: dict[int, list[int]] = {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)

        if reader.fieldnames is None:
            raise RuntimeError(f"jtp-3-hydra-tags.csv at {path} appears empty.")

        missing = {"tag", "category"} - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"jtp-3-hydra-tags.csv is missing required columns: {missing}. Found: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader):
            tag = row["tag"]
            raw_tag_names.append(tag)

            try:
                cat_id = int(row["category"])
            except (ValueError, KeyError):
                cat_id = -1

            indices_by_category.setdefault(cat_id, []).append(idx)

    logger.info(
        "Loaded JTP-3 tag CSV from %s: total=%d, categories=%s",
        path,
        len(raw_tag_names),
        {k: len(v) for k, v in sorted(indices_by_category.items())},
    )
    return _JTP3TagMetadata(raw_tag_names, indices_by_category)


# region Preprocessing helpers


class JTP3Batch(NamedTuple):
    """Preprocessed image data ready for JTP-3 inference.

    This is returned by preprocess() and consumed by the custom run path in
    PyTorchBackend (or directly in _run_pytorch_jtp3 if the backend exposes
    the model). It is distinct from the normal single-tensor preprocess output
    of other plugins.
    """

    patches: Tensor  # uint8, shape (max_seq, patch_size*patch_size*3)
    patch_coords: Tensor  # int16, shape (max_seq, 2)
    patch_valid: Tensor  # bool,  shape (max_seq,)


def _preprocess_image_jtp3(image: Image.Image, seqlen: int) -> JTP3Batch:
    """Convert a PIL Image to a JTP3Batch using the NaFlex patch pipeline."""
    # Local import to avoid pulling in torch at module load if not used.
    from .model import patchify_image, process_image

    # process_image handles sRGB conversion, aspect-ratio-preserving resize to
    # fit within seqlen patches at patch_size=16 px.
    processed = process_image(image, _PATCH_SIZE, seqlen)
    patches, patch_coords, patch_valid = patchify_image(processed, _PATCH_SIZE, seqlen, share_memory=False)
    return JTP3Batch(patches, patch_coords, patch_valid)


class JTP3BasePlugin(ModelPlugin):
    """Shared implementation for JTP-3 Hydra taggers by RedRocket.

    JTP-3 is fundamentally different from the WD/AnimeTimm family in that:
      - It uses a NaFlex (variable sequence length) ViT backbone.
      - Preprocessing produces (patches, patch_coords, patch_valid) rather than
        a single fixed-size tensor.
      - The model is always loaded via safetensors + custom architecture code
        (model.py / siglip2.py / hydra_pool.py); there is no ONNX export.
      - The forward pass requires three separate tensors with specific dtypes.

    Because of these differences the plugin builds the architecture itself and
    then attaches the model to the shared PyTorch backend, which runs inference
    and applies the usual precision plan.
    """

    _abstract = True
    family_name = "RedRocket JTP-3 Hydra Tagger"

    output_type = OutputType.TAGS
    supported_backends = (Backend.PYTORCH,)
    supported_processors = (
        CleanTags,
        CharacterIPMapping,
    )

    required_files = (
        FileSpec(
            name="model.safetensors",
            role=FileRole.WEIGHTS,
            backends=(Backend.PYTORCH,),
            hf_subdir="models",
        ),
        FileSpec(
            name="jtp-3-hydra-tags.csv",
            role=FileRole.TAG_LIST,
            hf_subdir="data",
        ),
        # FileSpec( # todo: may be possible to get best_threshold from this?
        #     name="jtp-3-hydra-val.csv",
        #     role=FileRole.TAG_LIST,
        #     hf_subdir="data",
        # ),
    )

    # Internal state — populated by load_ancillary()
    _raw_tag_names: list[str]
    _indices_by_category: dict[int, list[int]]
    _backend_instance: Any | None
    _jtp3_model: Any  # NaFlexVit, set after load_ancillary
    _device: str
    _seqlen: int

    def configure(self, **kwargs: Any) -> None:
        """Accept optional runtime configuration.

        Recognised keys:
          device (str)   — torch device string, e.g. "cuda", "cpu".  Defaults
                           to "cuda" if available, else "cpu".
          seqlen (int)   — NaFlex sequence length override (experimental).
                           Ignored if set via JTP3_SEQLEN env var; the env var
                           takes precedence so that it can be used without
                           touching calling code.
        """
        # Device is passed in by the session layer via the backend's device.
        # We capture it here for use in _run_jtp3_pytorch.
        self._device = str(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self._backend_instance = kwargs.get("backend_instance")

        # seqlen: env var takes precedence; kwarg is a secondary programmatic
        # override; module default is the final fallback.
        env_seqlen = _resolve_seqlen()
        if env_seqlen != _DEFAULT_SEQLEN:
            self._seqlen = env_seqlen
        elif "seqlen" in kwargs:
            value = int(kwargs["seqlen"])
            if not (_SEQLEN_MIN <= value <= _SEQLEN_MAX):
                logger.warning(
                    "configure(seqlen=%d) is outside valid range [%d, %d]; using default %d.",
                    value,
                    _SEQLEN_MIN,
                    _SEQLEN_MAX,
                    _DEFAULT_SEQLEN,
                )
                self._seqlen = _DEFAULT_SEQLEN
            else:
                self._seqlen = value
        else:
            self._seqlen = _DEFAULT_SEQLEN

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        """Load tag metadata and build + load the NaFlexVit model."""
        # Tag CSV
        csv_path = file_map["jtp-3-hydra-tags.csv"]
        logger.info("Loading JTP-3 tag list from %s", csv_path)
        metadata = _load_jtp3_tag_csv(csv_path)
        self._raw_tag_names = metadata.raw_tag_names
        self._indices_by_category = metadata.indices_by_category

        cat_info = " ".join(
            f"{label}={len(self._indices_by_category.get(int(cat_id), []))}"
            for cat_id, label in E621_CATEGORY_LABELS.items()
            if cat_id != E621TagCategory.INVALID
        )
        logger.info("Loaded JTP-3 tags: total=%d %s", len(self._raw_tag_names), cat_info)

        # -- Model weights --
        weights_path = file_map["model.safetensors"]
        logger.info("Loading JTP-3 model from %s", weights_path)

        if not hasattr(self, "_device"):
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(self, "_seqlen"):
            self._seqlen = _resolve_seqlen()

        from .model import load_model

        load_dtype = torch.float32 if self._device == "cpu" else torch.bfloat16
        model, labels, _ext_info = load_model(
            str(weights_path),
            extensions=(),  # extension (LoRA-like) support reserved for future use
            device=self._device,
            dtype=load_dtype,
        )

        if len(labels) != len(self._raw_tag_names):
            logger.warning(
                "Model label count (%d) does not match tag CSV count (%d). "
                "Inference will use min(model_labels, csv_tags) entries.",
                len(labels),
                len(self._raw_tag_names),
            )

        model.attn_pool.inference()  # type: ignore[attr-defined]
        self._jtp3_model = model

        # Hand the model to the shared backend so run() can use it.
        # _backend_instance is set by configure() when the session layer passes it in.
        backend = getattr(self, "_backend_instance", None)
        if backend is not None:
            backend.attach_model(model)
        else:
            logger.warning(
                "JTP-3 load_ancillary: no backend_instance available; "
                "model stored on plugin only. Inference will still work via postprocess()."
            )

        logger.info("JTP-3 model ready: device=%s seqlen=%d labels=%d", self._device, self._seqlen, len(labels))

    def preprocess(self, image: Any) -> JTP3Batch:
        """Convert a PIL Image to a JTP3Batch for the NaFlex forward pass.

        Returns a JTP3Batch (patches, patch_coords, patch_valid) rather than
        a single tensor. The session layer must call postprocess() with this
        batch directly — see note in postprocess().
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        seqlen = getattr(self, "_seqlen", _DEFAULT_SEQLEN)
        return _preprocess_image_jtp3(image, seqlen)

    def postprocess(self, raw_output: Any) -> TagResult:
        """Convert backend output (raw pre-sigmoid scores) to a TagResult."""
        if isinstance(raw_output, np.ndarray):
            scores_np = raw_output.ravel().astype(np.float32)
        elif isinstance(raw_output, Tensor):
            scores_np = raw_output.float().cpu().numpy().ravel()
        else:
            raise TypeError(
                f"JTP3 postprocess received unexpected type {type(raw_output).__name__}. Expected np.ndarray or Tensor."
            )

        # sigmoid application here
        scores_np = 1.0 / (1.0 + np.exp(-scores_np))

        usable_count = min(len(scores_np), len(self._raw_tag_names))
        if usable_count != len(self._raw_tag_names):
            logger.error(
                "JTP-3 score length mismatch: got %d scores for %d tags.", len(scores_np), len(self._raw_tag_names)
            )

        result_tags = {
            name: build_entries_for_indices(
                tag_names=self._raw_tag_names,
                indices=indices,
                scores=scores_np,
                usable_count=usable_count,
            )
            for cat_id, name in E621_CATEGORY_LABELS.items()
            if cat_id != E621TagCategory.INVALID and (indices := self._indices_by_category.get(int(cat_id)))
        }

        return TagResult(tags=result_tags)


# region Model Variants


class JTP3Plugin(JTP3BasePlugin):
    """JTP-3 Hydra tagger (e621 tags, SigLIP2-so400m-patch16-naflex backbone)."""

    model_id = "jtp-3"
    aliases = ("jtp-3-hydra",)
    display_name = "JTP-3 Hydra"
    description = "E621 tag prediction using JTP-3 Hydra (NaFlex ViT + HydraPool)."
    default_hf_repo = "RedRocket/JTP-3"
