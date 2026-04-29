from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.results import OutputType, MultiScoreResult

logger = logging.getLogger(__name__)


@dataclass
class DeepGHSAnimeAesResult(MultiScoreResult):
    """Single-value score result for the Waifu scorer models."""


class DeepGHSAnimeAesPlugin(ModelPlugin):
    """Shared implementation for the Eugeoter waifu scorer models."""

    _abstract = True


    output_type = OutputType.MULTI_SCORE
    supported_backends = [Backend.ONNX]
    supported_processors = []

    required_files = [
        FileSpec(
            name="model.ckpt",
            role=FileRole.WEIGHTS,
            backends=[Backend.PYTORCH],
        ),
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            backends=[Backend.ONNX],
        ),
    ]

    def load_ancillary(self, file_map: dict[str, Path]) -> None:

    def preprocess(self, image: Any) -> Any:
        return

    def postprocess(self, raw_output: Any) -> MultiScoreResult:

        return DeepGHSAnimeAesResult(
        )



# region Model Variants


class DGHSAesSwinV2xPlugin(DeepGHSAnimeAesPlugin):
    model_id = "aesv2-temp"
    aliases = []
    display_name = ""
    description = ""
    default_hf_repo = ""



# endregion Model Variants
