# Notes about the models:
# - Most or all models seem to allow dynamic input sizes even though they were trained
#   with specific ones like 384x384
#   - They won't error but will have lower accuracy

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.plugins.shared.generic_timm_pipeline import TimmPipelineMixin
from vibe.plugins.shared.tagger_shared import (
    build_entries_for_indices,
    load_tag_metadata,
    normalize_output_scores,
)
from vibe.result_processors import CharacterIPMapping, CleanTags, ScoreThresholds, TagLevelThresholds
from vibe.results import OutputType, TagEntry, TagResult
from vibe.tag_categories import DanbooruTagCategory

logger = logging.getLogger(__name__)


class WDV4AnimeTimmBasePlugin(TimmPipelineMixin, ModelPlugin):
    """Shared implementation for AnimeTimm dbv4-full taggers."""

    _abstract = True
    family_name = "AnimeTimm Taggers (dbv4-full)"

    output_type = OutputType.TAGS
    supported_backends = (
        Backend.ONNX,
        Backend.PYTORCH,
    )
    supported_processors = (
        CleanTags,
        CharacterIPMapping,
        ScoreThresholds,
        TagLevelThresholds,
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
        ),
        FileSpec(
            name="selected_tags.csv",
            role=FileRole.TAG_LIST,
        ),
    )

    FALLBACK_TIMM_MODEL_ARGS: dict[str, Any] = {}

    _raw_tag_names: list[str]
    _rating_indices: list[int]
    _general_indices: list[int]
    _character_indices: list[int]
    _artist_indices: list[int]
    _backend: Backend | None = None
    _backend_instance: Any | None = None
    _runtime_preprocess_steps: list[dict[str, Any]]

    # region Session Lifecycle

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        csv_path = file_map["selected_tags.csv"]
        logger.info("Loading AnimeTimm tag list from %s", csv_path)

        metadata = load_tag_metadata(csv_path)

        self._raw_tag_names = metadata.raw_tag_names
        self._rating_indices = metadata.indices_for(int(DanbooruTagCategory.RATING))
        self._general_indices = metadata.indices_for(int(DanbooruTagCategory.GENERAL))
        self._character_indices = metadata.indices_for(int(DanbooruTagCategory.CHARACTER))
        self._artist_indices = metadata.indices_for(int(DanbooruTagCategory.ARTIST))

        config = self.read_timm_config_json(file_map["config.json"])
        self._runtime_preprocess_steps = self.resolve_timm_preprocess_steps(
            config,
            file_map.get("preprocess.json"),
        )

        self.maybe_prepare_timm_pytorch_model(config=config, num_classes=len(self._raw_tag_names))

        logger.info(
            # todo: update log message to be more model specific maybe
            "Loaded AnimeTimm tags: total=%d general=%d artist=%d character=%d rating=%d",
            len(self._raw_tag_names),
            len(self._general_indices),
            len(self._artist_indices),
            len(self._character_indices),
            len(self._rating_indices),
        )

    # endregion Session Lifecycle

    def resolve_timm_model_args(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config is not None and isinstance(config.get("model_args"), dict):
            logger.debug(
                "Ignoring config.json model_args for model_id=%s to preserve stable PyTorch reconstruction behavior.",
                self.model_id,
            )
        return dict(self.FALLBACK_TIMM_MODEL_ARGS)

    # region Preprocess & Out Mapping

    def postprocess(self, raw_output: Any) -> TagResult:
        """Return full scored output grouped by AnimeTimm categories."""
        scores = normalize_output_scores(raw_output)

        usable_count = min(len(scores), len(self._raw_tag_names))
        if usable_count != len(self._raw_tag_names):
            logger.error(
                "Score length mismatch: got %d scores for %d tags.",
                len(scores),
                len(self._raw_tag_names),
            )

        rating = self._entries_for_indices(self._rating_indices, scores, usable_count)
        general = self._entries_for_indices(self._general_indices, scores, usable_count)
        character = self._entries_for_indices(self._character_indices, scores, usable_count)
        artist = self._entries_for_indices(self._artist_indices, scores, usable_count)

        return TagResult(
            tags={
                "rating": rating,
                "general": general,
                "character": character,
                "artist": artist,
            }
        )

    def _entries_for_indices(
        self,
        indices: list[int],
        scores: np.ndarray,
        usable_count: int,
    ) -> list[TagEntry]:
        return build_entries_for_indices(
            tag_names=self._raw_tag_names,
            indices=indices,
            scores=scores,
            usable_count=usable_count,
        )

    # endregion Preprocess & Out Mapping


# region Model Variants


class WDV4ConvNextV2HugePlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ConvNeXtV2 Huge model trained on Danbooru v4-full tags."""

    model_id = "wdv4-convnextv2-huge-dbv4-full"
    aliases = (
        "convnextv2-huge-dbv4-full",
        "convnextv2_huge.dbv4-full",
        "animetimm-convnextv2-huge",
        "wdv4-convnextv2-huge",
    )
    display_name = "AnimeTimm ConvNeXtV2 Huge"
    description = "Danbooru v4-full tagger using the AnimeTimm ConvNeXtV2 Huge architecture."
    default_hf_repo = "animetimm/convnextv2_huge.dbv4-full"

    supported_backends = (Backend.PYTORCH,)

    required_files = (
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
        ),
        FileSpec(
            name="selected_tags.csv",
            role=FileRole.TAG_LIST,
        ),
    )


class WDV4CaformerB36FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm CaFormer B36 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-caformer-b36-dbv4-full"
    aliases = (
        "caformer-b36-dbv4-full",
        "caformer_b36.dbv4-full",
        "animetimm-caformer-b36",
        "wdv4-caformer-b36",
    )
    display_name = "AnimeTimm CaFormer B36"
    description = "Danbooru v4-full tagger using the AnimeTimm CaFormer B36 architecture."
    default_hf_repo = "animetimm/caformer_b36.dbv4-full"


class WDV4CaformerM36FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm CaFormer M36 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-caformer-m36-dbv4-full"
    aliases = (
        "caformer-m36-dbv4-full",
        "caformer_m36.dbv4-full",
        "animetimm-caformer-m36",
        "wdv4-caformer-m36",
    )
    display_name = "AnimeTimm CaFormer M36"
    description = "Danbooru v4-full tagger using the AnimeTimm CaFormer M36 architecture."
    default_hf_repo = "animetimm/caformer_m36.dbv4-full"


class WDV4CaformerS36FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm CaFormer S36 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-caformer-s36-dbv4-full"
    aliases = (
        "caformer-s36-dbv4-full",
        "caformer_s36.dbv4-full",
        "animetimm-caformer-s36",
        "wdv4-caformer-s36",
    )
    display_name = "AnimeTimm CaFormer S36"
    description = "Danbooru v4-full tagger using the AnimeTimm CaFormer S36 architecture."
    default_hf_repo = "animetimm/caformer_s36.dbv4-full"


class WDV4CaformerS18FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm CaFormer S18 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-caformer-s18-dbv4-full"
    aliases = (
        "caformer-s18-dbv4-full",
        "caformer_s18.dbv4-full",
        "animetimm-caformer-s18",
        "wdv4-caformer-s18",
    )
    display_name = "AnimeTimm CaFormer S18"
    description = "Danbooru v4-full tagger using the AnimeTimm CaFormer S18 architecture."
    default_hf_repo = "animetimm/caformer_s18.dbv4-full"


class WDV4ConvNextBaseFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ConvNeXt Base model trained on Danbooru v4-full tags."""

    model_id = "wdv4-convnext-base-dbv4-full"
    aliases = (
        "convnext-base-dbv4-full",
        "convnext_base.dbv4-full",
        "animetimm-convnext-base",
        "wdv4-convnext-base",
    )
    display_name = "AnimeTimm ConvNeXt Base"
    description = "Danbooru v4-full tagger using the AnimeTimm ConvNeXt Base architecture."
    default_hf_repo = "animetimm/convnext_base.dbv4-full"


class WDV4Eva02LargePatch14448FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm Eva02 Large Patch14 448 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-eva02-large-patch14-448-dbv4-full"
    aliases = (
        "eva02-large-patch14-448-dbv4-full",
        "eva02_large_patch14_448.dbv4-full",
        "animetimm-eva02-large-patch14-448",
        "wdv4-eva02-large-patch14-448",
    )
    display_name = "AnimeTimm Eva02 Large Patch14 448"
    description = "Danbooru v4-full tagger using the AnimeTimm Eva02 Large Patch14 448 architecture."
    default_hf_repo = "animetimm/eva02_large_patch14_448.dbv4-full"


class WDV4MobileNetV3Large100FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV3 Large 100 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv3-large-100-dbv4-full"
    aliases = (
        "mobilenetv3-large-100-dbv4-full",
        "mobilenetv3_large_100.dbv4-full",
        "animetimm-mobilenetv3-large-100",
        "wdv4-mobilenetv3-large-100",
    )
    display_name = "AnimeTimm MobileNetV3 Large 100"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV3 Large 100 architecture."
    default_hf_repo = "animetimm/mobilenetv3_large_100.dbv4-full"


class WDV4MobileNetV3Large150dFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV3 Large 150d model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv3-large-150d-dbv4-full"
    aliases = (
        "mobilenetv3-large-150d-dbv4-full",
        "mobilenetv3_large_150d.dbv4-full",
        "animetimm-mobilenetv3-large-150d",
        "wdv4-mobilenetv3-large-150d",
    )
    display_name = "AnimeTimm MobileNetV3 Large 150d"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV3 Large 150d architecture."
    default_hf_repo = "animetimm/mobilenetv3_large_150d.dbv4-full"


class WDV4MobileNetV4ConvAaLargeFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV4 Conv AA Large model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv4-conv-aa-large-dbv4-full"
    aliases = (
        "mobilenetv4-conv-aa-large-dbv4-full",
        "mobilenetv4_conv_aa_large.dbv4-full",
        "animetimm-mobilenetv4-conv-aa-large",
        "wdv4-mobilenetv4-conv-aa-large",
    )
    display_name = "AnimeTimm MobileNetV4 Conv AA Large"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv AA Large architecture."
    default_hf_repo = "animetimm/mobilenetv4_conv_aa_large.dbv4-full"


class WDV4MobileNetV4ConvSmallFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV4 Conv Small model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv4-conv-small-dbv4-full"
    aliases = (
        "mobilenetv4-conv-small-dbv4-full",
        "mobilenetv4_conv_small.dbv4-full",
        "animetimm-mobilenetv4-conv-small",
        "wdv4-mobilenetv4-conv-small",
    )
    display_name = "AnimeTimm MobileNetV4 Conv Small"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv Small architecture."
    default_hf_repo = "animetimm/mobilenetv4_conv_small.dbv4-full"


class WDV4MobileNetV4ConvSmall050FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV4 Conv Small 050 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv4-conv-small-050-dbv4-full"
    aliases = (
        "mobilenetv4-conv-small-050-dbv4-full",
        "mobilenetv4_conv_small_050.dbv4-full",
        "animetimm-mobilenetv4-conv-small-050",
        "wdv4-mobilenetv4-conv-small-050",
    )
    display_name = "AnimeTimm MobileNetV4 Conv Small 050"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv Small 050 architecture."
    default_hf_repo = "animetimm/mobilenetv4_conv_small_050.dbv4-full"


class WDV4ResNet101FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet101 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet101-dbv4-full"
    aliases = (
        "resnet101-dbv4-full",
        "resnet101.dbv4-full",
        "animetimm-resnet101",
        "wdv4-resnet101",
    )
    display_name = "AnimeTimm ResNet101"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet101 architecture."
    default_hf_repo = "animetimm/resnet101.dbv4-full"


class WDV4ResNet152FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet152 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet152-dbv4-full"
    aliases = (
        "resnet152-dbv4-full",
        "resnet152.dbv4-full",
        "animetimm-resnet152",
        "wdv4-resnet152",
    )
    display_name = "AnimeTimm ResNet152"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet152 architecture."
    default_hf_repo = "animetimm/resnet152.dbv4-full"


class WDV4ResNet18FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet18 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet18-dbv4-full"
    aliases = (
        "resnet18-dbv4-full",
        "resnet18.dbv4-full",
        "animetimm-resnet18",
        "wdv4-resnet18",
    )
    display_name = "AnimeTimm ResNet18"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet18 architecture."
    default_hf_repo = "animetimm/resnet18.dbv4-full"


class WDV4ResNet34FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet34 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet34-dbv4-full"
    aliases = (
        "resnet34-dbv4-full",
        "resnet34.dbv4-full",
        "animetimm-resnet34",
        "wdv4-resnet34",
    )
    display_name = "AnimeTimm ResNet34"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet34 architecture."
    default_hf_repo = "animetimm/resnet34.dbv4-full"


class WDV4ResNet50FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet50 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet50-dbv4-full"
    aliases = (
        "resnet50-dbv4-full",
        "resnet50.dbv4-full",
        "animetimm-resnet50",
        "wdv4-resnet50",
    )
    display_name = "AnimeTimm ResNet50"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet50 architecture."
    default_hf_repo = "animetimm/resnet50.dbv4-full"


class WDV4SwinV2BaseWindow8256FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm SwinV2 Base Window8 256 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-swinv2-base-window8-256-dbv4-full"
    aliases = (
        "swinv2-base-window8-256-dbv4-full",
        "swinv2_base_window8_256.dbv4-full",
        "animetimm-swinv2-base-window8-256",
        "wdv4-swinv2-base-window8-256",
    )
    display_name = "AnimeTimm SwinV2 Base Window8 256"
    description = "Danbooru v4-full tagger using the AnimeTimm SwinV2 Base Window8 256 architecture."
    default_hf_repo = "animetimm/swinv2_base_window8_256.dbv4-full"


class WDV4SwinV2BaseWindow8256Dbv4aFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm SwinV2 Base Window8 256 model trained on Danbooru v4a-full tags."""

    model_id = "wdv4-swinv2-base-window8-256-dbv4a-full"
    aliases = (
        "swinv2-base-window8-256-dbv4a-full",
        "swinv2_base_window8_256.dbv4a-full",
        "animetimm-swinv2-base-window8-256-dbv4a",
        "wdv4-swinv2-base-window8-256-dbv4a",
    )
    display_name = "AnimeTimm SwinV2 Base Window8 256"
    description = "Danbooru v4a-full tagger using the AnimeTimm SwinV2 Base Window8 256 architecture."
    default_hf_repo = "animetimm/swinv2_base_window8_256.dbv4a-full"


class WDV4VitBasePatch16224FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ViT Base Patch16 224 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-vit-base-patch16-224-dbv4-full"
    aliases = (
        "vit-base-patch16-224-dbv4-full",
        "vit_base_patch16_224.dbv4-full",
        "animetimm-vit-base-patch16-224",
        "wdv4-vit-base-patch16-224",
    )
    display_name = "AnimeTimm ViT Base Patch16 224"
    description = "Danbooru v4-full tagger using the AnimeTimm ViT Base Patch16 224 architecture."
    default_hf_repo = "animetimm/vit_base_patch16_224.dbv4-full"


# endregion Model Variants
