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
from vibe.plugins.shared.tagger_shared import normalize_output_scores
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
    _is_logits: bool | None = None

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        logger.info(
            "Loading generic timm model plugin. Preprocessing and label resolution use heuristics from config.json."
        )
        config_path = artifacts.get_optional("config")
        config = self.read_timm_config_json(config_path) if config_path else {}

        self._labels = self._resolve_labels(config)
        self._num_classes = self._resolve_num_classes(config)
        self._is_logits = self._resolve_is_logits_from_config(config)

        preprocess_path = artifacts.get_optional("preprocess")
        self.prepare_timm_runtime_preprocess(config, preprocess_path, prefer_timm=True)

    def postprocess(self, raw_output: Any) -> ScoreResult | MultiScoreResult | TagResult:
        expected_count = self._num_classes if self._num_classes else (len(self._labels) if self._labels else None)

        # Detect or confirm whether the output tensor represents raw logits
        is_logits = self._determine_is_logits(raw_output)

        scores = normalize_output_scores(raw_output, is_logits=is_logits, expected_count=expected_count)

        output_type = self.capabilities.output_type

        # 1. Single Scalar Score Result
        if output_type == OutputType.SCORE:
            val = float(scores[0]) if len(scores) > 0 else 0.0
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

    def _determine_is_logits(self, raw_output: Any) -> bool:
        """Determine whether the raw output represents logits, locking the decision once proven."""
        # 1. Return cached determination if already locked
        if self._is_logits is not None:
            return self._is_logits

        # 2. PyTorch timm models natively output raw linear logits
        if self._active_backend == Backend.PYTORCH:
            self._is_logits = True
            return True

        # 3. Dynamic inspection for ONNX or unconfigured models
        arr = np.asarray(raw_output, dtype=np.float32)
        min_val = float(np.min(arr)) if arr.size > 0 else 0.0
        max_val = float(np.max(arr)) if arr.size > 0 else 0.0

        # Irrefutable proof: values < 0 or > 1 cannot be probabilities
        if min_val < -1e-4 or max_val > 1.0 + 1e-4:
            logger.debug(
                "Detected raw logits for model '%s' (min=%.4f, max=%.4f)", self.identity.model_id, min_val, max_val
            )
            self._is_logits = True
            return True

        # For models with many classes (>10), if all values fall in [0, 1], it is almost certainly probabilities
        num_classes = self._num_classes or (len(self._labels) if self._labels else 0)
        if num_classes > 10:
            logger.debug(
                "Detected probabilities for model '%s' (%d classes, all in [0, 1])", self.identity.model_id, num_classes
            )
            self._is_logits = False
            return False

        # Fallback for ambiguous cases (e.g. single-scalar score ONNX models)
        return False

    def _resolve_is_logits_from_config(self, config: dict[str, Any]) -> bool | None:
        """Inspect config.json for explicit activation settings."""
        for key in ("classifier_activation", "activation_fn", "activation"):
            act = config.get(key)
            if isinstance(act, str):
                act_lower = act.lower().strip()
                if act_lower in ("sigmoid", "softmax"):
                    return False
                if act_lower in ("none", "identity", "linear"):
                    return True

        for cfg_key in ("pretrained_cfg", "pretrained_cfg_overlay"):
            sub_cfg = config.get(cfg_key)
            if isinstance(sub_cfg, dict):
                act = sub_cfg.get("classifier_activation")
                if isinstance(act, str):
                    act_lower = act.lower().strip()
                    if act_lower in ("sigmoid", "softmax"):
                        return False
                    if act_lower in ("none", "identity", "linear"):
                        return True

        return None

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
