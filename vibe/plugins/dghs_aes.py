# Note: Scores deviate a negligible amount from deepgh-imgutils

# todo: specify subfolders and all that

# todo: way of not computing percentile/weighted score

# todo: remove percentile/weighted mean from here and move to a result processor

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.results import MultiScoreResult, OutputType

logger = logging.getLogger(__name__)


class DeepGHSAnimeAesPlugin(ModelPlugin):
    """Shared implementation for DeepGHS anime aesthetic scorers."""

    _abstract = True

    IMAGE_SIZE = 448
    SCORE_MIN = 0.0
    SCORE_MAX = 1.0

    output_type = OutputType.MULTI_SCORE
    supported_backends = [Backend.PYTORCH, Backend.ONNX]
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
        FileSpec(
            name="meta.json",
            role=FileRole.CONFIG,
        ),
        FileSpec(
            name="samples.npz",
            role=FileRole.MAPPING,
        ),
    ]

    _labels: list[str]
    _mark_table: tuple[np.ndarray, np.ndarray] | None = None
    _image_size: int

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        meta_path = file_map.get("meta.json")
        if meta_path is None:
            raise RuntimeError("Missing meta.json for DeepGHS aesthetic model.")

        meta = self._read_meta_json(meta_path)
        labels = meta.get("labels")
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise RuntimeError("meta.json must contain a list of label strings.")
        self._labels = list(labels)

        self._image_size = self._resolve_image_size(meta)

        samples_path = file_map.get("samples.npz")
        if samples_path is None:
            raise RuntimeError("Missing samples.npz for DeepGHS aesthetic model.")
        self._mark_table = self._load_mark_table(samples_path)

    def preprocess(self, image: Any) -> np.ndarray:
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        image = image.resize((self._image_size, self._image_size), Image.Resampling.BICUBIC)

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = (image_array / 255.0).astype(np.float32)
        mean = np.asarray([0.5], dtype=np.float32).reshape((-1, 1, 1))
        std = np.asarray([0.5], dtype=np.float32).reshape((-1, 1, 1))
        image_array = (image_array - mean) / std
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def postprocess(self, raw_output: Any) -> MultiScoreResult:
        scores = self._flatten_scores(raw_output)
        usable_count = min(len(scores), len(self._labels))
        if usable_count != len(self._labels):
            logger.warning(
                "Score length mismatch for model_id=%s: got %d scores for %d labels.",
                self.model_id,
                len(scores),
                len(self._labels),
            )

        labels = self._labels[:usable_count]
        values = scores[:usable_count]

        score_map = {label: float(values[idx]) for idx, label in enumerate(labels)}

        weighted_score = self._weighted_mean(labels, values)
        percentile = self._score_to_percentile(weighted_score)

        metrics: dict[str, float] = {}
        if weighted_score is not None:
            metrics["weighted_score"] = weighted_score
        if percentile is not None:
            metrics["percentile"] = percentile

        return MultiScoreResult(
            scores=score_map,
            score_min=self.SCORE_MIN,
            score_max=self.SCORE_MAX,
            label_order=list(labels),
            metrics=metrics,
        )

    def _read_meta_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            raise RuntimeError(f"Failed to read DeepGHS meta.json: {exc}") from exc

    def _resolve_image_size(self, meta: dict[str, Any]) -> int:
        for key in ("image_size", "input_size", "img_size"):
            value = meta.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return self.IMAGE_SIZE

    def _load_mark_table(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        try:
            with np.load(path, allow_pickle=False) as data:
                arr = data["arr_0"]

                # format: (2, N)
                scores = np.asarray(arr[0], dtype=np.float32)
                percentiles = np.asarray(arr[1], dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to read DeepGHS samples.npz: {exc}") from exc

        if scores.size != percentiles.size:
            raise RuntimeError("samples.npz score and percentile arrays must be the same length.")

        order = np.argsort(scores)
        scores = scores[order]
        percentiles = percentiles[order]

        x = np.concatenate(([0.0], scores, [6.0])).astype(np.float32, copy=False)
        y = np.concatenate(([0.0], percentiles, [1.0])).astype(np.float32, copy=False)
        return (x, y)

    def _flatten_scores(self, raw_output: Any) -> np.ndarray:
        scores = np.asarray(raw_output, dtype=np.float32)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        elif scores.ndim > 1:
            if scores.shape[0] == 1:
                scores = np.squeeze(scores, axis=0)
            scores = np.ravel(scores)
        return scores.astype(np.float32, copy=False)

    def _weighted_mean(self, labels: list[str], values: np.ndarray) -> float | None:
        if values.size == 0:
            return None
        weights = np.arange(len(labels), dtype=np.float32)
        return float(np.sum(weights[: values.size] * values))

    def _score_to_percentile(self, weighted_score: float | None) -> float | None:
        if weighted_score is None or self._mark_table is None:
            return None

        xs, ys = self._mark_table
        if xs.size == 0 or ys.size == 0:
            return None
        clipped = float(np.clip(weighted_score, xs[0], xs[-1]))

        idx = int(np.searchsorted(xs, clipped, side="right")) - 1
        idx = np.clip(idx, 0, xs.size - 2)

        x0, x1 = xs[idx], xs[idx + 1]
        y0, y1 = ys[idx], ys[idx + 1]

        if x1 == x0:
            return float(np.clip(y0, 0.0, 1.0))
        ratio = (clipped - x0) / (x1 - x0)
        return float(np.clip(y0 + ratio * (y1 - y0), 0.0, 1.0))


# region Model Variants


class DGHSAesSwinV2xPlugin(DeepGHSAnimeAesPlugin):
    model_id = "dghs-aes-swinv2pv3-ls0.2-x"
    aliases = [
        "swinv2pv3_v0_448_ls0.2_x",
        "dghs-aes-swinv2pv3-x",
    ]
    display_name = "DeepGHS Anime Aesthetic SwinV2 PV3 448 x"
    description = "Anime aesthetic scorer using DeepGHS SwinV2 PV3 448 (ls0.2 x) model."
    default_hf_repo = "deepghs/anime_aesthetic"


class DGHSAesSwinV2Plugin(DeepGHSAnimeAesPlugin):
    model_id = "dghs-aes-swinv2pv3-ls0.2"
    aliases = [
        "swinv2pv3_v0_448_ls0.2",
        "dghs-aes-swinv2pv3",
    ]
    display_name = "DeepGHS Anime Aesthetic SwinV2 PV3 448"
    description = "Anime aesthetic scorer using DeepGHS SwinV2 PV3 448 (ls0.2) model."
    default_hf_repo = "deepghs/anime_aesthetic"


class DGHSAesCaformerS36Plugin(DeepGHSAnimeAesPlugin):
    model_id = "dghs-aes-caformer-s36-ls0.2"
    aliases = [
        "caformer_s36_v0_ls0.2",
        "dghs-aes-caformer-s36",
    ]
    IMAGE_SIZE = 384
    display_name = "DeepGHS Anime Aesthetic CaFormer S36"
    description = "Anime aesthetic scorer using DeepGHS CaFormer S36 (ls0.2) model."
    default_hf_repo = "deepghs/anime_aesthetic"


# endregion Model Variants
