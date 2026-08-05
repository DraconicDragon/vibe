# Notes about the models:
# - Most or all models seem to allow dynamic input sizes even though they were trained
#   with specific ones like 384x384
#   - They won't error but will have lower accuracy

from __future__ import annotations

import logging
from typing import Any

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
from vibe.plugins.shared.tagger_shared import (
    build_categorized_tag_result,
    load_tag_metadata,
    normalize_output_scores,
)
from vibe.result_transforms import CharacterIPMapping, CleanTags, ScoreThresholds, TagLevelThresholds
from vibe.results import OutputType, TagResult
from vibe.tag_categories import DanbooruTagCategory, TagCategory

logger = logging.getLogger(__name__)


# region Base Plugin


class AnimeTimmBasePlugin(TimmPipelineMixin, ModelPlugin):
    """Shared implementation for AnimeTimm dbv4-full taggers."""

    family_name = "AnimeTimm Taggers (dbv4-full)"

    capabilities = ModelCapabilities(
        output_type=OutputType.TAGS,
        output_categories=(
            TagCategory.RATING,
            TagCategory.GENERAL,
            TagCategory.CHARACTER,
            TagCategory.ARTIST,
        ),
        transforms=(
            CleanTags,
            ScoreThresholds(threshold=0.35),
            CharacterIPMapping,
            TagLevelThresholds,
        ),
    )

    variants = (
        ModelVariant(
            backend=Backend.ONNX,
            artifacts=(
                ArtifactSpec(id="model_onnx", name="model.onnx", role=FileRole.WEIGHTS),
                ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG),
                ArtifactSpec(id="preprocess", name="preprocess.json", role=FileRole.CONFIG, required=False),
                ArtifactSpec(id="tag_list", name="selected_tags.csv", role=FileRole.TAG_LIST),
            ),
        ),
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(id="model_pt", name="model.safetensors", role=FileRole.WEIGHTS),
                ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG),
                ArtifactSpec(id="preprocess", name="preprocess.json", role=FileRole.CONFIG, required=False),
                ArtifactSpec(id="tag_list", name="selected_tags.csv", role=FileRole.TAG_LIST),
            ),
        ),
    )

    _raw_tag_names: list[str]
    _num_classes: int
    _category_indices: dict[str, list[int]]

    # region Session Lifecycle

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        csv_path = artifacts.get("tag_list")
        logger.info("Loading AnimeTimm tag list from %s", csv_path)

        metadata = load_tag_metadata(csv_path)

        self._raw_tag_names = metadata.raw_tag_names
        self._num_classes = len(self._raw_tag_names)

        self._category_indices = {
            str(TagCategory.RATING): metadata.indices_for(int(DanbooruTagCategory.RATING)),
            str(TagCategory.GENERAL): metadata.indices_for(int(DanbooruTagCategory.GENERAL)),
            str(TagCategory.CHARACTER): metadata.indices_for(int(DanbooruTagCategory.CHARACTER)),
            str(TagCategory.ARTIST): metadata.indices_for(int(DanbooruTagCategory.ARTIST)),
        }

        config_path = artifacts.get_optional("config")
        preprocess_path = artifacts.get_optional("preprocess")
        if config_path:
            config = self.read_timm_config_json(config_path)
            self.prepare_timm_runtime_preprocess(config, preprocess_path)

        logger.info(
            "Loaded AnimeTimm tags for %s: total=%d general=%d artist=%d character=%d rating=%d",
            self.identity.model_id,
            self._num_classes,
            len(self._category_indices.get(str(TagCategory.GENERAL), [])),
            len(self._category_indices.get(str(TagCategory.ARTIST), [])),
            len(self._category_indices.get(str(TagCategory.CHARACTER), [])),
            len(self._category_indices.get(str(TagCategory.RATING), [])),
        )

    # endregion Session Lifecycle

    # region Postprocess

    def postprocess(self, raw_output: Any) -> TagResult:
        """Return full scored output grouped by AnimeTimm categories."""
        scores = normalize_output_scores(raw_output)
        return build_categorized_tag_result(self._raw_tag_names, scores, self._category_indices)

    # endregion Postprocess


# endregion Base Plugin


# region Model Variants


class ATConvNextV2HugePlugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-convnextv2-huge-dbv4-full",
        display_name="AnimeTimm ConvNeXtV2 Huge",
        description="Danbooru v4-full tagger using the AnimeTimm ConvNeXtV2 Huge architecture.",
    )
    default_repo_id = "animetimm/convnextv2_huge.dbv4-full"

    # ConvNeXtV2 Huge repository only provides PyTorch safetensors
    variants = (
        ModelVariant(
            backend=Backend.PYTORCH,
            artifacts=(
                ArtifactSpec(id="model_pt", name="model.safetensors", role=FileRole.WEIGHTS),
                ArtifactSpec(id="config", name="config.json", role=FileRole.CONFIG),
                ArtifactSpec(id="preprocess", name="preprocess.json", role=FileRole.CONFIG, required=False),
                ArtifactSpec(id="tag_list", name="selected_tags.csv", role=FileRole.TAG_LIST),
            ),
        ),
    )


class ATCaformerB36Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-caformer-b36-dbv4-full",
        display_name="AnimeTimm CaFormer B36",
        description="Danbooru v4-full tagger using the AnimeTimm CaFormer B36 architecture.",
    )
    default_repo_id = "animetimm/caformer_b36.dbv4-full"


class ATCaformerM36Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-caformer-m36-dbv4-full",
        display_name="AnimeTimm CaFormer M36",
        description="Danbooru v4-full tagger using the AnimeTimm CaFormer M36 architecture.",
    )
    default_repo_id = "animetimm/caformer_m36.dbv4-full"


class ATCaformerS36Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-caformer-s36-dbv4-full",
        display_name="AnimeTimm CaFormer S36",
        description="Danbooru v4-full tagger using the AnimeTimm CaFormer S36 architecture.",
    )
    default_repo_id = "animetimm/caformer_s36.dbv4-full"


class ATCaformerS18Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-caformer-s18-dbv4-full",
        display_name="AnimeTimm CaFormer S18",
        description="Danbooru v4-full tagger using the AnimeTimm CaFormer S18 architecture.",
    )
    default_repo_id = "animetimm/caformer_s18.dbv4-full"


class ATConvNextBasePlugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-convnext-base-dbv4-full",
        display_name="AnimeTimm ConvNeXt Base",
        description="Danbooru v4-full tagger using the AnimeTimm ConvNeXt Base architecture.",
    )
    default_repo_id = "animetimm/convnext_base.dbv4-full"


class ATEva02LargePatch14448Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-eva02-large-patch14-448-dbv4-full",
        display_name="AnimeTimm Eva02 Large Patch14 448",
        description="Danbooru v4-full tagger using the AnimeTimm Eva02 Large Patch14 448 architecture.",
    )
    default_repo_id = "animetimm/eva02_large_patch14_448.dbv4-full"


class ATMobileNetV3Large100Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-mobilenetv3-large-100-dbv4-full",
        display_name="AnimeTimm MobileNetV3 Large 100",
        description="Danbooru v4-full tagger using the AnimeTimm MobileNetV3 Large 100 architecture.",
    )
    default_repo_id = "animetimm/mobilenetv3_large_100.dbv4-full"


class ATMobileNetV3Large150dPlugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-mobilenetv3-large-150d-dbv4-full",
        display_name="AnimeTimm MobileNetV3 Large 150d",
        description="Danbooru v4-full tagger using the AnimeTimm MobileNetV3 Large 150d architecture.",
    )
    default_repo_id = "animetimm/mobilenetv3_large_150d.dbv4-full"


class ATMobileNetV4ConvAaLargePlugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-mobilenetv4-conv-aa-large-dbv4-full",
        display_name="AnimeTimm MobileNetV4 Conv AA Large",
        description="Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv AA Large architecture.",
    )
    default_repo_id = "animetimm/mobilenetv4_conv_aa_large.dbv4-full"


class ATMobileNetV4ConvSmallPlugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-mobilenetv4-conv-small-dbv4-full",
        display_name="AnimeTimm MobileNetV4 Conv Small",
        description="Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv Small architecture.",
    )
    default_repo_id = "animetimm/mobilenetv4_conv_small.dbv4-full"


class ATMobileNetV4ConvSmall050Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-mobilenetv4-conv-small-050-dbv4-full",
        display_name="AnimeTimm MobileNetV4 Conv Small 050",
        description="Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv Small 050 architecture.",
    )
    default_repo_id = "animetimm/mobilenetv4_conv_small_050.dbv4-full"


class ATResNet101Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-resnet101-dbv4-full",
        display_name="AnimeTimm ResNet101",
        description="Danbooru v4-full tagger using the AnimeTimm ResNet101 architecture.",
    )
    default_repo_id = "animetimm/resnet101.dbv4-full"


class ATResNet152Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-resnet152-dbv4-full",
        display_name="AnimeTimm ResNet152",
        description="Danbooru v4-full tagger using the AnimeTimm ResNet152 architecture.",
    )
    default_repo_id = "animetimm/resnet152.dbv4-full"


class ATResNet18Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-resnet18-dbv4-full",
        display_name="AnimeTimm ResNet18",
        description="Danbooru v4-full tagger using the AnimeTimm ResNet18 architecture.",
    )
    default_repo_id = "animetimm/resnet18.dbv4-full"


class ATResNet34Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-resnet34-dbv4-full",
        display_name="AnimeTimm ResNet34",
        description="Danbooru v4-full tagger using the AnimeTimm ResNet34 architecture.",
    )
    default_repo_id = "animetimm/resnet34.dbv4-full"


class ATResNet50Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-resnet50-dbv4-full",
        display_name="AnimeTimm ResNet50",
        description="Danbooru v4-full tagger using the AnimeTimm ResNet50 architecture.",
    )
    default_repo_id = "animetimm/resnet50.dbv4-full"


class ATSwinV2BaseWindow8256Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-swinv2-base-window8-256-dbv4-full",
        display_name="AnimeTimm SwinV2 Base Window8 256",
        description="Danbooru v4-full tagger using the AnimeTimm SwinV2 Base Window8 256 architecture.",
    )
    default_repo_id = "animetimm/swinv2_base_window8_256.dbv4-full"


class ATSwinV2BaseWindow8256Dbv4aPlugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-swinv2-base-window8-256-dbv4a-full",
        display_name="AnimeTimm SwinV2 Base Window8 256 (with artist tags)",
        description="Danbooru tagger using the AnimeTimm SwinV2 Base Window8 256 architecture. Trained with artist tags.",
    )
    default_repo_id = "animetimm/swinv2_base_window8_256.dbv4a-full"


class ATVitBasePatch16224Plugin(AnimeTimmBasePlugin):
    identity = ModelIdentity(
        model_id="at-vit-base-patch16-224-dbv4-full",
        display_name="AnimeTimm ViT Base Patch16 224",
        description="Danbooru v4-full tagger using the AnimeTimm ViT Base Patch16 224 architecture.",
    )
    default_repo_id = "animetimm/vit_base_patch16_224.dbv4-full"


# endregion Model Variants
