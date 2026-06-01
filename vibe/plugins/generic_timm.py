# todo: needs real world testing

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.plugins.shared.generic_timm_pipeline import TimmPipelineMixin, flatten_timm_output
from vibe.results import MultiScoreResult, OutputType, ScoreResult, TagEntry, TagResult

logger = logging.getLogger(__name__)


class GenericTimmBasePlugin(TimmPipelineMixin, ModelPlugin):
    """Generic timm classifier/scorer for timm-style repos."""

    _abstract = True
    display_name = "Generic timm"
    family_name = "Generic Timm Models"
    description = "Loads arbitrary models so long as they are in timm format and contain timm config files."
    supported_backends = (
        Backend.ONNX,
        Backend.PYTORCH,
    )

    required_files = (
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            backends=(Backend.ONNX,),
        ),
        FileSpec(
            name="model.safetensors",
            role=FileRole.WEIGHTS,
            backends=(Backend.PYTORCH,),
        ),
        FileSpec(
            name="config.json",
            role=FileRole.CONFIG,
        ),
        FileSpec(
            name="preprocess.json",
            role=FileRole.CONFIG,
            required=False,
        ),
    )

    _labels: list[str] | None = None

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        logger.warning(
            "The generic-timm model plugin is experimental. Results and preprocessing may need model-specific tuning."
        )
        config = self.read_timm_config_json(file_map["config.json"])
        self._labels = self._resolve_labels(config)
        num_classes = self._resolve_num_classes(config)
        self.maybe_prepare_timm_pytorch_model(config=config, num_classes=num_classes)
        self.prepare_timm_runtime_preprocess(
            config,
            file_map.get("preprocess.json"),
            prefer_timm=self._backend == Backend.PYTORCH,
        )

    def postprocess(self, raw_output: Any) -> ScoreResult | MultiScoreResult | TagResult:
        scores = flatten_timm_output(raw_output)
        if self.output_type == OutputType.SCORE:
            value = float(scores[0]) if len(scores) else 0.0
            label = self._labels[0] if self._labels else "score"
            return ScoreResult(score=value, score_min=0.0, score_max=1.0, label=label)

        labels = self._labels
        if labels is None or len(labels) != len(scores):
            labels = [f"class_{index}" for index in range(len(scores))]

        score_values = [float(value) for value in scores]
        if self.output_type == OutputType.TAGS:
            return TagResult(
                tags={
                    "tags": [
                        TagEntry(tag=label, score=score) for label, score in zip(labels, score_values, strict=False)
                    ]
                }
            )

        label_map = {index: label for index, label in enumerate(labels[: len(score_values)])}
        return MultiScoreResult(
            scores=score_values,
            label_map=label_map,
            label_order=list(label_map.values()),
        )

    def _resolve_labels(self, config: dict[str, Any]) -> list[str] | None:
        for key in ("label_names", "labels", "classes", "categories"):
            raw = config.get(key)
            if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                return list(raw)

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
                return [label for _, label in sorted(pairs)]

        pretrained_cfg = config.get("pretrained_cfg")
        if isinstance(pretrained_cfg, dict):
            classifier = pretrained_cfg.get("label_names")
            if isinstance(classifier, list) and all(isinstance(item, str) for item in classifier):
                return list(classifier)
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
    """Generic timm model that returns all outputs as a MultiScoreResult."""

    model_id = "generic-timm-multi-score"
    aliases = ("generic-timm-multiscore",)
    display_name = "Generic timm scorer (multi-score)"
    description = "Experimental generic timm loader that returns vector outputs as multi-score results."
    output_type = OutputType.MULTI_SCORE


class GenericTimmScorerPlugin(GenericTimmBasePlugin):
    """Generic timm model that returns the first output as a ScoreResult."""

    model_id = "generic-timm-score"
    aliases = ()
    display_name = "Generic timm scorer"
    description = "Experimental generic timm loader that returns the first output as a scalar score."
    output_type = OutputType.SCORE


class GenericTimmTaggerPlugin(GenericTimmBasePlugin):
    """Generic timm model that returns outputs as flat tags."""

    model_id = "generic-timm-tags"
    aliases = ()
    display_name = "Generic timm tagger"
    description = "Experimental generic timm loader that returns vector outputs as flat tag results."
    output_type = OutputType.TAGS
