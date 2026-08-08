"""Generic timm classifier/scorer plugin implementations for arbitrary timm models."""
# todo: needs real world testing

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    FileRole,
    ModelCapabilities,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
)
from vibe.plugins.shared.generic_timm_pipeline import TimmPipelineMixin
from vibe.plugins.shared.scores_utils import normalize_scalar
from vibe.results import MultiScoreResult, OutputType, ScoreEntry, ScoreResult, TagEntry, TagResult
from vibe.tag_categories import TagCategory

logger = logging.getLogger(__name__)


class GenericTimmBasePlugin(TimmPipelineMixin, ModelPlugin):
    """Generic timm classifier/scorer for arbitrary timm-style repos."""

    family_name = "Generic Timm Models"

    capabilities = ModelCapabilities(
        output_type=OutputType.TAGS,
        output_categories=(TagCategory.GENERAL,),
    )

    variants = (
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(id="model_pt", name="model.safetensors", role=FileRole.WEIGHTS),
                ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG, required=False),
                ArtifactSpec(id="preprocess", name="preprocess.json", role=FileRole.CONFIG, required=False),
            ),
        ),
        ModelVariant(
            backend=Backend.ONNX,
            artifacts=(
                ArtifactSpec(id="model_onnx", name="model.onnx", role=FileRole.WEIGHTS),
                ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG, required=False),
                ArtifactSpec(id="preprocess", name="preprocess.json", role=FileRole.CONFIG, required=False),
            ),
        ),
    )

    _labels: list[str] | None = None
    _num_classes: int | None = None

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        logger.info(
            "Loading generic timm model plugin. Preprocessing and label resolution use heuristics from config.json."
        )
        config_path = artifacts.get_optional("config")
        config = self.read_timm_config_json(config_path) if config_path else {}

        self._labels = self._resolve_labels(config)
        self._num_classes = self._resolve_num_classes(config)

        preprocess_path = artifacts.get_optional("preprocess")
        self.prepare_timm_runtime_preprocess(config, preprocess_path, prefer_timm=True)

    def postprocess(self, raw_output: Any) -> ScoreResult | MultiScoreResult | TagResult:
        scores = self._flatten_scores(raw_output)
        output_type = self.capabilities.output_type

        # 1. Single Scalar Score Result
        if output_type == OutputType.SCORE:
            val = float(scores[0]) if len(scores) > 0 else 0.0
            if val < 0.0 or val > 1.0:
                val = float(1.0 / (1.0 + np.exp(-np.clip(val, -80.0, 80.0))))
            label = self._labels[0] if self._labels else "score"
            return ScoreResult(
                score=val,
                score_min=0.0,
                score_max=1.0,
                normalized_score=normalize_scalar(val, 0.0, 1.0),
                label=label,
            )

        labels = self._labels
        if labels is None or len(labels) != len(scores):
            labels = [f"class_{index}" for index in range(len(scores))]

        # Apply sigmoid to raw logits if values fall outside [0, 1]
        if np.min(scores) < 0.0 or np.max(scores) > 1.0:
            clipped = np.clip(scores, -80.0, 80.0)
            scores = 1.0 / (1.0 + np.exp(-clipped))

        score_values = [float(val) for val in scores]

        # 2. Tag Result
        if output_type == OutputType.TAGS:
            cat_name = (
                self.capabilities.output_categories[0]
                if self.capabilities.output_categories
                else TagCategory.GENERAL.value
            )
            tag_entries = [TagEntry(tag=label, score=score) for label, score in zip(labels, score_values, strict=False)]
            tag_entries.sort(key=lambda e: e.score, reverse=True)
            return TagResult(tags={cat_name: tag_entries})

        # 3. Multi-Score Result
        entries = [
            ScoreEntry(
                label=label,
                score=score,
                score_min=0.0,
                score_max=1.0,
                normalized_score=normalize_scalar(score, 0.0, 1.0),
            )
            for label, score in zip(labels, score_values, strict=False)
        ]
        entries.sort(key=lambda e: e.score, reverse=True)

        return MultiScoreResult(
            entries=entries,
            normalized_score=float(np.mean(score_values)) if score_values else 0.0,
        )

    def _flatten_scores(self, raw_output: Any) -> np.ndarray:
        """Extract flat 1D score array from arbitrary backend model outputs."""
        if isinstance(raw_output, (tuple, list)):
            raw_output = raw_output[0]

        scores = np.asarray(raw_output, dtype=np.float32)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        elif scores.ndim > 1:
            if scores.shape[0] == 1:
                scores = np.squeeze(scores, axis=0)
            scores = np.ravel(scores)
        return scores.astype(np.float32, copy=False)

    def _resolve_labels(self, config: dict[str, Any]) -> list[str] | None:
        """Attempt to extract label list from various common HF/timm config keys."""
        for key in ("label_names", "labels", "classes", "categories"):
            raw = config.get(key)
            if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                return [str(item) for item in raw]

        id2label = config.get("id2label")
        if isinstance(id2label, dict):
            pairs: list[tuple[int, str]] = []
            for key, value in id2label.items():
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    continue
                pairs.append((index, str(value)))
            if pairs:
                return [label for _, label in sorted(pairs, key=lambda x: x[0])]

        for cfg_key in ("pretrained_cfg", "pretrained_cfg_overlay"):
            sub_cfg = config.get(cfg_key)
            if isinstance(sub_cfg, dict):
                label_list = sub_cfg.get("label_names") or sub_cfg.get("classes")
                if isinstance(label_list, list) and all(isinstance(item, str) for item in label_list):
                    return [str(item) for item in label_list]

        return None

        return None

    def _resolve_num_classes(self, config: dict[str, Any]) -> int | None:
        for source in (config, config.get("model_args")):
            if not isinstance(source, dict):
                continue
            value = source.get("num_classes")
            if isinstance(value, int) and value > 0:
                return value
        if self._labels:
            return len(self._labels)
        return None


class GenericTimmMultiScorerPlugin(GenericTimmBasePlugin):
    identity = ModelIdentity(
        model_id="generic-timm-multi-score",
        display_name="Generic timm scorer (multi-score)",
        description="Experimental generic timm loader that returns vector outputs as multi-score results.",
    )
    default_repo_id = ""
    capabilities = ModelCapabilities(output_type=OutputType.MULTI_SCORE)


class GenericTimmScorerPlugin(GenericTimmBasePlugin):
    identity = ModelIdentity(
        model_id="generic-timm-score",
        display_name="Generic timm scorer",
        description="Experimental generic timm loader that returns the first output as a scalar score.",
    )
    default_repo_id = ""
    capabilities = ModelCapabilities(output_type=OutputType.SCORE)


class GenericTimmTaggerPlugin(GenericTimmBasePlugin):
    identity = ModelIdentity(
        model_id="generic-timm-tags",
        display_name="Generic timm tagger",
        description="Experimental generic timm loader that returns vector outputs as flat tag results.",
    )
    default_repo_id = ""
    capabilities = ModelCapabilities(
        output_type=OutputType.TAGS,
        output_categories=(TagCategory.GENERAL,),
    )
