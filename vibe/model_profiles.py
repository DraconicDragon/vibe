"""
Reusable metadata profiles for standard model types (taggers, scorers, etc.).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

from vibe.metadata import (
    ConsumerSettingSpec,
    InputSpec,
    LabelSource,
    Modality,
    ModelProfile,
    OutputKind,
    OutputSpec,
    ScoreSemantics,
    StandardConsumerSettingId,
    TagFilterRecommendation,
)

# TODO: Add build_image_text_profile and build_embedding_profile when raw multimodal / embedding plugins are added.


def build_tagger_profile(
    *,
    categories: Sequence[str | Enum],
    score_semantics: ScoreSemantics = ScoreSemantics.PROBABILITY,
    recommended_filter: TagFilterRecommendation | None = None,
    output_extras: Mapping[str, str] | None = None,
    entry_extras: Mapping[str, str] | None = None,
) -> ModelProfile:
    """Helper to generate common tagger metadata."""
    consumer_settings: list[ConsumerSettingSpec] = []

    if recommended_filter is not None:
        consumer_settings.append(
            ConsumerSettingSpec(
                id=StandardConsumerSettingId.TAG_FILTER,
                display_name="Tag Score Filter",
                description="Recommended confidence thresholds for selecting tags.",
                schema=TagFilterRecommendation.model_json_schema(),
                default=TagFilterRecommendation().model_dump(mode="json"),
                recommended=recommended_filter,
            )
        )

    return ModelProfile(
        model_types=("tagger",),
        input_spec=InputSpec(modalities=(Modality.IMAGE,)),
        output_spec=OutputSpec(
            kind=OutputKind.TAGS,
            label_source=LabelSource.MODEL,
            score_semantics=score_semantics,
            categories=tuple(categories),
            output_extras=output_extras or {},
            entry_extras=entry_extras or {},
        ),
        consumer_settings=tuple(consumer_settings),
    )


def build_scorer_profile(
    *,
    score_semantics: ScoreSemantics = ScoreSemantics.UNCALIBRATED,
    output_extras: Mapping[str, str] | None = None,
    entry_extras: Mapping[str, str] | None = None,
) -> ModelProfile:
    """Helper to generate common scalar scorer metadata."""
    return ModelProfile(
        model_types=("scorer",),
        input_spec=InputSpec(modalities=(Modality.IMAGE,)),
        output_spec=OutputSpec(
            kind=OutputKind.SCORE,
            label_source=LabelSource.MODEL,
            score_semantics=score_semantics,
            output_extras=output_extras or {},
            entry_extras=entry_extras or {},
        ),
        consumer_settings=(),
    )


def build_multi_scorer_profile(
    *,
    score_semantics: ScoreSemantics = ScoreSemantics.PROBABILITY,
    output_extras: Mapping[str, str] | None = None,
    entry_extras: Mapping[str, str] | None = None,
) -> ModelProfile:
    """Helper to generate common multi-scorer metadata."""
    return ModelProfile(
        model_types=("multi_scorer",),
        input_spec=InputSpec(modalities=(Modality.IMAGE,)),
        output_spec=OutputSpec(
            kind=OutputKind.MULTI_SCORE,
            label_source=LabelSource.MODEL,
            score_semantics=score_semantics,
            output_extras=output_extras or {},
            entry_extras=entry_extras or {},
        ),
        consumer_settings=(),
    )
