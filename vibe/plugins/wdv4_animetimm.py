from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.plugins.shared.tagger_shared import (
    build_entries_for_indices,
    load_tag_metadata,
    normalize_output_scores,
    preprocess_tagger_image,
)
from vibe.result_processors import CharacterIPMapping, CleanTags, TagLevelThresholds
from vibe.results import OutputType, TagEntry, TagResult
from vibe.tag_categories import DanbooruTagCategory

logger = logging.getLogger(__name__)


@dataclass
class AnimeTimmV4TagResult(TagResult):
    """AnimeTimm v4 output grouped by explicit category fields."""

    rating: list[TagEntry] = field(default_factory=list)
    general: list[TagEntry] = field(default_factory=list)
    character: list[TagEntry] = field(default_factory=list)
    artist: list[TagEntry] = field(default_factory=list)

    def categories(self) -> dict[str, list[TagEntry]]:
        return {
            "rating": self.rating,
            "general": self.general,
            "character": self.character,
            "artist": self.artist,
        }


class WDV4AnimeTimmBasePlugin(ModelPlugin):
    """Shared implementation for AnimeTimm dbv4-full taggers."""

    _abstract = True

    output_type = OutputType.TAGS
    supported_backends = [Backend.ONNX, Backend.PYTORCH]
    supported_processors = [CleanTags, CharacterIPMapping, TagLevelThresholds]

    required_files = [
        FileSpec(
            name="model.onnx",
            role=FileRole.WEIGHTS,
            backends=[Backend.ONNX],
        ),
        FileSpec(
            name="model.safetensors",
            role=FileRole.WEIGHTS,
            required=False,
            backends=[Backend.PYTORCH],
        ),
        FileSpec(
            name="config.json",
            role=FileRole.CONFIG,
            required=False,
            backends=[Backend.PYTORCH],
        ),
        FileSpec(
            name="selected_tags.csv",
            role=FileRole.TAG_LIST,
            backends=[],
        ),
    ]

    IMAGE_SIZE = 448
    INPUT_LAYOUT = "NCHW"
    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)
    FALLBACK_TIMM_MODEL_ARGS: dict[str, Any] = {}

    _raw_tag_names: list[str]
    _rating_indices: list[int]
    _general_indices: list[int]
    _character_indices: list[int]
    _artist_indices: list[int]
    _backend: Backend | None = None
    _backend_instance: Any | None = None
    _runtime_image_size: int | None = None
    _runtime_normalize_mean: tuple[float, float, float] | None = None
    _runtime_normalize_std: tuple[float, float, float] | None = None
    _source: str = ""
    _optional_missing_files: dict[str, str] = {}
    _config_issue_reason: str | None = None

    def configure(self, **kwargs: Any) -> None:
        self._backend = kwargs.get("backend")
        self._backend_instance = kwargs.get("backend_instance")
        self._source = str(kwargs.get("source", ""))
        optional_missing = kwargs.get("optional_missing_files")
        self._optional_missing_files = dict(optional_missing) if isinstance(optional_missing, dict) else {}

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        csv_path = file_map["selected_tags.csv"]
        logger.info("Loading AnimeTimm tag list from %s", csv_path)

        metadata = load_tag_metadata(csv_path)

        self._raw_tag_names = metadata.raw_tag_names
        self._rating_indices = metadata.indices_for(int(DanbooruTagCategory.RATING))
        self._general_indices = metadata.indices_for(int(DanbooruTagCategory.GENERAL))
        self._character_indices = metadata.indices_for(int(DanbooruTagCategory.CHARACTER))
        self._artist_indices = metadata.indices_for(int(DanbooruTagCategory.ARTIST))

        config = self._read_config_json(file_map.get("config.json"))
        self._config_issue_reason = self._resolve_config_issue_reason(file_map)
        self._log_config_fallback_once(config)
        self._configure_pytorch_preprocess_from_config(config)

        self._maybe_prepare_pytorch_model(file_map, config=config)

        logger.info(
            # todo: update log message to be more model specific maybe
            "Loaded AnimeTimm tags: total=%d general=%d artist=%d character=%d rating=%d",
            len(self._raw_tag_names),
            len(self._general_indices),
            len(self._artist_indices),
            len(self._character_indices),
            len(self._rating_indices),
        )

    def _maybe_prepare_pytorch_model(self, file_map: dict[str, Path], *, config: dict[str, Any] | None) -> None:
        if self._backend != Backend.PYTORCH:
            return

        backend = self._backend_instance
        if backend is None:
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            # Keep ONNX-only installations unaffected.
            return

        model_or_state = getattr(backend, "raw", None)
        if isinstance(model_or_state, nn.Module):
            return
        if not isinstance(model_or_state, dict):
            return

        architecture = self._resolve_timm_architecture(config)
        if not architecture:
            raise RuntimeError(
                "Could not resolve timm architecture for AnimeTimm PyTorch model reconstruction. "
                "Provide config.json or use a model with a recognizable default_hf_repo suffix."
            )

        model_args = self._resolve_timm_model_args(config)
        model_args["num_classes"] = len(self._raw_tag_names)

        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "timm is required to build AnimeTimm PyTorch architectures from .safetensors state dicts. "
                "Install it with your torch extra (e.g. pip install 'vibe[torch-cpu]')."
            ) from exc

        logger.info(
            "Building timm model for model_id=%s architecture=%s num_classes=%s",
            self.model_id,
            architecture,
            model_args["num_classes"],
        )
        model = self._create_timm_model(timm, architecture, model_args)

        missing, unexpected = model.load_state_dict(model_or_state, strict=False)
        if missing:
            logger.warning("timm load_state_dict missing keys for model_id=%s: %s", self.model_id, missing[:8])
        if unexpected:
            logger.warning(
                "timm load_state_dict unexpected keys for model_id=%s: %s",
                self.model_id,
                unexpected[:8],
            )

        backend._model = model

        apply_precision = getattr(backend, "_apply_precision_plan", None)
        if callable(apply_precision):
            apply_precision(torch)
        else:
            device = getattr(backend, "device", "cpu")
            model.to(device=device)
            model.eval()

    def _create_timm_model(self, timm_module: Any, architecture: str, model_args: dict[str, Any]) -> Any:
        try:
            return timm_module.create_model(architecture, pretrained=False, **model_args)
        except TypeError as exc:
            num_classes = int(model_args.get("num_classes", 0))
            logger.warning(
                "timm.create_model rejected model_args for model_id=%s architecture=%s (%s). "
                "Retrying with conservative fallback args.",
                self.model_id,
                architecture,
                exc,
            )
            return timm_module.create_model(architecture, pretrained=False, num_classes=num_classes)

    def _resolve_timm_architecture(self, config: dict[str, Any] | None) -> str | None:
        architecture = config.get("architecture") if config else None
        if isinstance(architecture, str) and architecture.strip():
            return architecture.strip()

        if config is not None:
            logger.debug("config.json missing usable architecture for model_id=%s", self.model_id)

        repo = self.default_hf_repo or ""
        suffix = repo.split("/", 1)[-1]
        if ".dbv" in suffix:
            fallback_arch = suffix.split(".dbv", 1)[0]
            logger.info("Using fallback timm architecture '%s' for model_id=%s", fallback_arch, self.model_id)
            return fallback_arch

        logger.debug("Unable to infer fallback architecture from default_hf_repo for model_id=%s.", self.model_id)
        return None

    def _resolve_timm_model_args(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config is not None:
            raw_args = config.get("model_args")
            if isinstance(raw_args, dict):
                return dict(raw_args)
            logger.debug("config.json missing usable model_args for model_id=%s", self.model_id)

        return dict(self.FALLBACK_TIMM_MODEL_ARGS)

    def _read_config_json(self, config_path: Path | None) -> dict[str, Any] | None:
        if config_path is None or not config_path.is_file():
            return None

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            if isinstance(parsed, dict):
                return parsed
            self._config_issue_reason = f"config.json at '{config_path}' is not a JSON object"
        except Exception as exc:
            self._config_issue_reason = f"failed to parse config.json at '{config_path}': {exc}"
        return None

    def _resolve_config_issue_reason(self, file_map: dict[str, Path]) -> str:
        if self._config_issue_reason:
            return self._config_issue_reason

        config_path = file_map.get("config.json")
        if config_path is not None and not config_path.is_file():
            return f"resolved config path does not exist: {config_path}"

        mapped_reason = self._optional_missing_files.get("config.json")
        if mapped_reason:
            return mapped_reason

        if self._source.startswith("hf:"):
            return (
                "config.json unavailable from HF resolution (check repo access, cache, auto-download, and auth token)"
            )

        if self._source.startswith("local:") or Path(self._source).exists():
            return "config.json missing from local model folder"

        return "config.json could not be resolved"

    def _log_config_fallback_once(self, config: dict[str, Any] | None) -> None:
        if self._backend != Backend.PYTORCH:
            return
        if config is not None:
            return
        logger.warning(
            "PyTorch config unavailable for model_id=%s (%s); using fallback preprocessing/model settings.",
            self.model_id,
            self._config_issue_reason or "unknown reason",
        )

    def _configure_pytorch_preprocess_from_config(self, config: dict[str, Any] | None) -> None:
        self._runtime_image_size = None
        self._runtime_normalize_mean = None
        self._runtime_normalize_std = None

        if self._backend != Backend.PYTORCH:
            return

        if config is None:
            logger.debug("Using hardcoded preprocess defaults for model_id=%s", self.model_id)
            return

        pretrained_cfg = config.get("pretrained_cfg")
        if not isinstance(pretrained_cfg, dict):
            logger.warning(
                "config.json missing 'pretrained_cfg' for model_id=%s; using preprocess defaults.", self.model_id
            )
            return

        self._runtime_image_size = self._extract_image_size(pretrained_cfg)
        if self._runtime_image_size is None:
            logger.warning(
                "config.json missing usable input size for model_id=%s; using IMAGE_SIZE=%s.",
                self.model_id,
                self.IMAGE_SIZE,
            )

        self._runtime_normalize_mean = self._extract_triplet(pretrained_cfg, "mean")
        self._runtime_normalize_std = self._extract_triplet(pretrained_cfg, "std")

        if self._runtime_normalize_mean is None or self._runtime_normalize_std is None:
            logger.warning(
                "config.json missing usable mean/std for model_id=%s; using normalize defaults.", self.model_id
            )
            self._runtime_normalize_mean = None
            self._runtime_normalize_std = None

    def _extract_image_size(self, pretrained_cfg: dict[str, Any]) -> int | None:
        for key in ("test_input_size", "input_size"):
            raw = pretrained_cfg.get(key)
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                continue
            try:
                height = int(raw[1])
                width = int(raw[2])
            except (TypeError, ValueError):
                continue
            if height > 0 and width > 0 and height == width:
                return height
        return None

    def _extract_triplet(self, source: dict[str, Any], key: str) -> tuple[float, float, float] | None:
        raw = source.get(key)
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        except (TypeError, ValueError):
            return None

    def preprocess(self, image: Any) -> np.ndarray:
        """Convert image to float32 NCHW tensor using timm-like normalization."""
        image_size = self.IMAGE_SIZE
        normalize_mean = self.NORMALIZE_MEAN
        normalize_std = self.NORMALIZE_STD

        if self._backend == Backend.PYTORCH:
            if self._runtime_image_size is not None:
                image_size = self._runtime_image_size
            if self._runtime_normalize_mean is not None and self._runtime_normalize_std is not None:
                normalize_mean = self._runtime_normalize_mean
                normalize_std = self._runtime_normalize_std

        return preprocess_tagger_image(
            image,
            image_size=image_size,
            input_layout=self.INPUT_LAYOUT,
            rgb_to_bgr=False,
            normalize_to_unit=True,
            mean=normalize_mean,
            std=normalize_std,
        )

    def postprocess(self, raw_output: Any) -> AnimeTimmV4TagResult:
        """Return full scored output grouped by AnimeTimm categories."""
        scores = normalize_output_scores(raw_output)

        usable_count = min(len(scores), len(self._raw_tag_names))
        if usable_count != len(self._raw_tag_names):
            logger.warning(
                "Score length mismatch: got %d scores for %d tags.",
                len(scores),
                len(self._raw_tag_names),
            )

        rating = self._entries_for_indices(self._rating_indices, scores, usable_count)
        general = self._entries_for_indices(self._general_indices, scores, usable_count)
        character = self._entries_for_indices(self._character_indices, scores, usable_count)
        artist = self._entries_for_indices(self._artist_indices, scores, usable_count)

        return AnimeTimmV4TagResult(
            rating=rating,
            general=general,
            character=character,
            artist=artist,
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


# region Model Variants


class WDV4ConvNextV2HugePlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ConvNeXtV2 Huge model trained on Danbooru v4-full tags."""

    model_id = "wdv4-convnextv2-huge-dbv4-full"
    aliases = [
        "convnextv2-huge-dbv4-full",
        "convnextv2_huge.dbv4-full",
        "animetimm-convnextv2-huge",
        "wdv4-convnextv2-huge",
    ]
    IMAGE_SIZE = 512
    display_name = "AnimeTimm ConvNeXtV2 Huge (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm ConvNeXtV2 Huge architecture."
    default_hf_repo = "animetimm/convnextv2_huge.dbv4-full"


class WDV4CaformerB36FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm CaFormer B36 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-caformer-b36-dbv4-full"
    aliases = [
        "caformer-b36-dbv4-full",
        "caformer_b36.dbv4-full",
        "animetimm-caformer-b36",
        "wdv4-caformer-b36",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm CaFormer B36 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm CaFormer B36 architecture."
    default_hf_repo = "animetimm/caformer_b36.dbv4-full"


class WDV4CaformerM36FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm CaFormer M36 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-caformer-m36-dbv4-full"
    aliases = [
        "caformer-m36-dbv4-full",
        "caformer_m36.dbv4-full",
        "animetimm-caformer-m36",
        "wdv4-caformer-m36",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm CaFormer M36 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm CaFormer M36 architecture."
    default_hf_repo = "animetimm/caformer_m36.dbv4-full"


class WDV4CaformerS36FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm CaFormer S36 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-caformer-s36-dbv4-full"
    aliases = [
        "caformer-s36-dbv4-full",
        "caformer_s36.dbv4-full",
        "animetimm-caformer-s36",
        "wdv4-caformer-s36",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm CaFormer S36 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm CaFormer S36 architecture."
    default_hf_repo = "animetimm/caformer_s36.dbv4-full"


class WDV4CaformerS18FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm CaFormer S18 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-caformer-s18-dbv4-full"
    aliases = [
        "caformer-s18-dbv4-full",
        "caformer_s18.dbv4-full",
        "animetimm-caformer-s18",
        "wdv4-caformer-s18",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm CaFormer S18 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm CaFormer S18 architecture."
    default_hf_repo = "animetimm/caformer_s18.dbv4-full"


class WDV4ConvNextBaseFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ConvNeXt Base model trained on Danbooru v4-full tags."""

    model_id = "wdv4-convnext-base-dbv4-full"
    aliases = [
        "convnext-base-dbv4-full",
        "convnext_base.dbv4-full",
        "animetimm-convnext-base",
        "wdv4-convnext-base",
    ]
    IMAGE_SIZE = 448
    display_name = "AnimeTimm ConvNeXt Base (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm ConvNeXt Base architecture."
    default_hf_repo = "animetimm/convnext_base.dbv4-full"


class WDV4Eva02LargePatch14448FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm Eva02 Large Patch14 448 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-eva02-large-patch14-448-dbv4-full"
    aliases = [
        "eva02-large-patch14-448-dbv4-full",
        "eva02_large_patch14_448.dbv4-full",
        "animetimm-eva02-large-patch14-448",
        "wdv4-eva02-large-patch14-448",
    ]
    IMAGE_SIZE = 448
    display_name = "AnimeTimm Eva02 Large Patch14 448 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm Eva02 Large Patch14 448 architecture."
    default_hf_repo = "animetimm/eva02_large_patch14_448.dbv4-full"


class WDV4MobileNetV3Large100FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV3 Large 100 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv3-large-100-dbv4-full"
    aliases = [
        "mobilenetv3-large-100-dbv4-full",
        "mobilenetv3_large_100.dbv4-full",
        "animetimm-mobilenetv3-large-100",
        "wdv4-mobilenetv3-large-100",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm MobileNetV3 Large 100 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV3 Large 100 architecture."
    default_hf_repo = "animetimm/mobilenetv3_large_100.dbv4-full"


class WDV4MobileNetV3Large150dFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV3 Large 150d model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv3-large-150d-dbv4-full"
    aliases = [
        "mobilenetv3-large-150d-dbv4-full",
        "mobilenetv3_large_150d.dbv4-full",
        "animetimm-mobilenetv3-large-150d",
        "wdv4-mobilenetv3-large-150d",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm MobileNetV3 Large 150d (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV3 Large 150d architecture."
    default_hf_repo = "animetimm/mobilenetv3_large_150d.dbv4-full"


class WDV4MobileNetV4ConvAaLargeFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV4 Conv AA Large model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv4-conv-aa-large-dbv4-full"
    aliases = [
        "mobilenetv4-conv-aa-large-dbv4-full",
        "mobilenetv4_conv_aa_large.dbv4-full",
        "animetimm-mobilenetv4-conv-aa-large",
        "wdv4-mobilenetv4-conv-aa-large",
    ]
    IMAGE_SIZE = 448
    display_name = "AnimeTimm MobileNetV4 Conv AA Large (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv AA Large architecture."
    default_hf_repo = "animetimm/mobilenetv4_conv_aa_large.dbv4-full"


class WDV4MobileNetV4ConvSmallFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV4 Conv Small model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv4-conv-small-dbv4-full"
    aliases = [
        "mobilenetv4-conv-small-dbv4-full",
        "mobilenetv4_conv_small.dbv4-full",
        "animetimm-mobilenetv4-conv-small",
        "wdv4-mobilenetv4-conv-small",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm MobileNetV4 Conv Small (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv Small architecture."
    default_hf_repo = "animetimm/mobilenetv4_conv_small.dbv4-full"


class WDV4MobileNetV4ConvSmall050FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm MobileNetV4 Conv Small 050 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-mobilenetv4-conv-small-050-dbv4-full"
    aliases = [
        "mobilenetv4-conv-small-050-dbv4-full",
        "mobilenetv4_conv_small_050.dbv4-full",
        "animetimm-mobilenetv4-conv-small-050",
        "wdv4-mobilenetv4-conv-small-050",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm MobileNetV4 Conv Small 050 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm MobileNetV4 Conv Small 050 architecture."
    default_hf_repo = "animetimm/mobilenetv4_conv_small_050.dbv4-full"


class WDV4ResNet101FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet101 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet101-dbv4-full"
    aliases = [
        "resnet101-dbv4-full",
        "resnet101.dbv4-full",
        "animetimm-resnet101",
        "wdv4-resnet101",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm ResNet101 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet101 architecture."
    default_hf_repo = "animetimm/resnet101.dbv4-full"


class WDV4ResNet152FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet152 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet152-dbv4-full"
    aliases = [
        "resnet152-dbv4-full",
        "resnet152.dbv4-full",
        "animetimm-resnet152",
        "wdv4-resnet152",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm ResNet152 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet152 architecture."
    default_hf_repo = "animetimm/resnet152.dbv4-full"


class WDV4ResNet18FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet18 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet18-dbv4-full"
    aliases = [
        "resnet18-dbv4-full",
        "resnet18.dbv4-full",
        "animetimm-resnet18",
        "wdv4-resnet18",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm ResNet18 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet18 architecture."
    default_hf_repo = "animetimm/resnet18.dbv4-full"


class WDV4ResNet34FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet34 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet34-dbv4-full"
    aliases = [
        "resnet34-dbv4-full",
        "resnet34.dbv4-full",
        "animetimm-resnet34",
        "wdv4-resnet34",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm ResNet34 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet34 architecture."
    default_hf_repo = "animetimm/resnet34.dbv4-full"


class WDV4ResNet50FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ResNet50 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-resnet50-dbv4-full"
    aliases = [
        "resnet50-dbv4-full",
        "resnet50.dbv4-full",
        "animetimm-resnet50",
        "wdv4-resnet50",
    ]
    IMAGE_SIZE = 384
    display_name = "AnimeTimm ResNet50 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm ResNet50 architecture."
    default_hf_repo = "animetimm/resnet50.dbv4-full"


class WDV4SwinV2BaseWindow8256FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm SwinV2 Base Window8 256 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-swinv2-base-window8-256-dbv4-full"
    aliases = [
        "swinv2-base-window8-256-dbv4-full",
        "swinv2_base_window8_256.dbv4-full",
        "animetimm-swinv2-base-window8-256",
        "wdv4-swinv2-base-window8-256",
    ]
    IMAGE_SIZE = 448
    display_name = "AnimeTimm SwinV2 Base Window8 256 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm SwinV2 Base Window8 256 architecture."
    default_hf_repo = "animetimm/swinv2_base_window8_256.dbv4-full"


class WDV4SwinV2BaseWindow8256Dbv4aFullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm SwinV2 Base Window8 256 model trained on Danbooru v4a-full tags."""

    model_id = "wdv4-swinv2-base-window8-256-dbv4a-full"
    aliases = [
        "swinv2-base-window8-256-dbv4a-full",
        "swinv2_base_window8_256.dbv4a-full",
        "animetimm-swinv2-base-window8-256-dbv4a",
        "wdv4-swinv2-base-window8-256-dbv4a",
    ]
    IMAGE_SIZE = 448
    display_name = "AnimeTimm SwinV2 Base Window8 256 (dbv4a-full)"
    description = "Danbooru v4a-full tagger using the AnimeTimm SwinV2 Base Window8 256 architecture."
    default_hf_repo = "animetimm/swinv2_base_window8_256.dbv4a-full"


class WDV4VitBasePatch16224FullPlugin(WDV4AnimeTimmBasePlugin):
    """AnimeTimm ViT Base Patch16 224 model trained on Danbooru v4-full tags."""

    model_id = "wdv4-vit-base-patch16-224-dbv4-full"
    aliases = [
        "vit-base-patch16-224-dbv4-full",
        "vit_base_patch16_224.dbv4-full",
        "animetimm-vit-base-patch16-224",
        "wdv4-vit-base-patch16-224",
    ]
    IMAGE_SIZE = 448
    display_name = "AnimeTimm ViT Base Patch16 224 (dbv4-full)"
    description = "Danbooru v4-full tagger using the AnimeTimm ViT Base Patch16 224 architecture."
    default_hf_repo = "animetimm/vit_base_patch16_224.dbv4-full"


# endregion Model Variants
