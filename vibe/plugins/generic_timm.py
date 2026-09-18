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
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
)
from vibe.contracts import LabelCatalogProvider
from vibe.metadata import LabelCatalog, LabelInfo, OutputKind
from vibe.model_profiles import build_multi_scorer_profile, build_scorer_profile, build_tagger_profile
from vibe.plugins.shared.generic_timm_pipeline import TimmPipelineMixin
from vibe.plugins.shared.scores_utils import normalize_scalar
from vibe.plugins.shared.tagger_shared import detect_is_logits, normalize_output_scores
from vibe.results import MultiScoreResult, ScoreEntry, ScoreResult, TagEntry, TagResult

logger = logging.getLogger(__name__)


class GenericTimmBasePlugin(TimmPipelineMixin, ModelPlugin):
    """Generic timm classifier/scorer for arbitrary timm-style repos."""

    family_name = "Generic Timm Models"
    custom_only = True
    profile = build_tagger_profile(categories=("general",))
    implements = (LabelCatalogProvider,)

    variants = (
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(id="model_pt", name="model.safetensors", role=FileRole.WEIGHTS),
                ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG),
                ArtifactSpec(id="preprocess", name="preprocess.json", role=FileRole.CONFIG, required=False),
            ),
        ),
        ModelVariant(
            backend=Backend.ONNX,
            artifacts=(
                ArtifactSpec(id="model_onnx", name="model.onnx", role=FileRole.WEIGHTS),
                ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG),
                ArtifactSpec(id="preprocess", name="preprocess.json", role=FileRole.CONFIG, required=False),
            ),
        ),
    )

    catalog: LabelCatalog
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

        if not self._labels:
            num_c = self._num_classes or 1000
            self._labels = [f"class_{i}" for i in range(num_c)]

        label_infos = [LabelInfo(index=idx, name=name, category="general") for idx, name in enumerate(self._labels)]
        self.catalog = LabelCatalog(labels=tuple(label_infos), categories=("general",))

        preprocess_path = artifacts.get_optional("preprocess")
        self.prepare_timm_runtime_preprocess(config, preprocess_path, prefer_timm=True)

    def postprocess(self, raw_output: Any) -> ScoreResult | MultiScoreResult | TagResult:
        expected_count = self._num_classes if self._num_classes else (len(self._labels) if self._labels else None)

        # Detect or confirm whether the output tensor represents raw logits
        is_logits = self._determine_is_logits(raw_output, expected_count=expected_count)

        output_kind = self.profile.output_spec.kind

        # Mutually exclusive categorical scorers use Softmax; multi-label taggers use Sigmoid
        use_softmax = output_kind == OutputKind.MULTI_SCORE

        scores = normalize_output_scores(
            raw_output,
            is_logits=is_logits,
            expected_count=expected_count,
            use_softmax=use_softmax,
        )

        if output_kind == OutputKind.SCORE:
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

        if output_kind == OutputKind.TAGS:
            tag_entries = [TagEntry(tag=label, score=score) for label, score in zip(labels, score_values, strict=False)]
            tag_entries.sort(key=lambda e: e.score, reverse=True)
            return TagResult(categories={"general": tag_entries})

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

    def _determine_is_logits(self, raw_output: Any, expected_count: int | None = None) -> bool:
        """Determine whether the raw output represents logits, locking the decision once proven."""
        # 1. Return cached determination if already locked
        if self._is_logits is not None:
            return self._is_logits

        # 2. PyTorch timm models natively output raw linear logits
        if self.active_backend is None:
            raise RuntimeError(
                f"Plugin '{self.identity.model_id}' has no active backend bound to this session. "
                "Unable to determine logits semantics for a pooled runtime reuse without execution state."
            )

        if self.active_backend == Backend.PYTORCH:
            self._is_logits = True
            return True

        detected = detect_is_logits(raw_output, min_classes_for_prob_inference=10)
        if detected is not None:
            logger.debug(
                "Heuristic logit detection locked for model '%s': is_logits=%s", self.identity.model_id, detected
            )
            self._is_logits = detected
            return detected

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
                except (TypeError, ValueError) as exc:
                    logger.debug("Skipping non-integer key '%s' in id2label: %s", key, exc)
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
    profile = build_multi_scorer_profile()


class GenericTimmScorerPlugin(GenericTimmBasePlugin):
    identity = ModelIdentity(
        model_id="generic-timm-score",
        display_name="Generic timm scorer",
        description="Experimental generic timm loader that returns the first output as a scalar score.",
    )
    default_repo_id = ""
    profile = build_scorer_profile()


class GenericTimmTaggerPlugin(GenericTimmBasePlugin):
    identity = ModelIdentity(
        model_id="generic-timm-tags",
        display_name="Generic timm tagger",
        description="Experimental generic timm loader that returns vector outputs as flat tag results.",
    )
    default_repo_id = ""
    profile = build_tagger_profile(categories=("general",))
