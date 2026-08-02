"""Reusable timm pipeline mixin handling config parsing, preprocessing, and runtime building."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from vibe.backends.base import ArtifactMap, Backend, ExecutionRequest, RuntimeExecutor
from vibe.backends.runtime.onnx import ONNXBackend
from vibe.backends.runtime.pytorch import PyTorchBackend

logger = logging.getLogger(__name__)


class TimmPipelineMixin:
    """Reusable timm config, preprocess, and runtime builder helpers."""

    model_id: str
    default_repo_id: str
    FALLBACK_TIMM_MODEL_ARGS: ClassVar[dict[str, Any]] = {}

    # Class-level fallback overrides when config.json lacks mean/std
    FALLBACK_MEAN: tuple[float, float, float] | None = None
    FALLBACK_STD: tuple[float, float, float] | None = None

    _runtime_preprocess_steps: list[dict[str, Any]]
    _runtime_timm_transform: Any | None = None
    _active_backend: Backend | None = None

    # region Runtime Builder

    def build_runtime(self, artifacts: ArtifactMap, request: ExecutionRequest) -> RuntimeExecutor:
        """Build an ONNX or PyTorch runtime executor for a timm model."""
        self._active_backend = request.backend

        if request.backend == Backend.ONNX:
            onnx_path = artifacts.get("model_onnx")
            backend = ONNXBackend()
            backend.load(onnx_path, request)
            return backend

        if request.backend == Backend.PYTORCH:
            config_path = artifacts.get_optional("config")
            config = self.read_timm_config_json(config_path) if config_path else None

            weights_path = artifacts.get("model_pt")
            num_classes = getattr(self, "_num_classes", None)

            model = self.build_timm_pytorch_model(
                weights_path=weights_path,
                config=config,
                num_classes=num_classes,
            )
            backend = PyTorchBackend()
            backend.load(model, request)
            return backend

        raise ValueError(f"Unsupported backend '{request.backend}' for timm pipeline.")

    def build_timm_pytorch_model(
        self,
        *,
        weights_path: Path,
        config: dict[str, Any] | None,
        num_classes: int | None = None,
    ) -> Any:
        """Construct a PyTorch nn.Module from timm architecture and load state dict."""
        architecture = self.resolve_timm_architecture(config)
        if not architecture:
            raise RuntimeError(
                f"Could not resolve timm architecture for model '{self.model_id}'. "
                "Provide config.json with an architecture/model_type field."
            )

        model_args = self.resolve_timm_model_args(config)
        if num_classes is not None:
            model_args["num_classes"] = int(num_classes)

        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required to build PyTorch models.") from exc

        logger.info("Loading PyTorch weights from %s", weights_path)
        if weights_path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(weights_path, device="cpu")
        else:
            import torch

            state_dict = torch.load(weights_path, map_location="cpu")

        model = self.create_timm_model(timm, architecture, model_args)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("timm load_state_dict missing keys for model_id=%s: %s", self.model_id, missing[:8])
        if unexpected:
            logger.warning("timm load_state_dict unexpected keys for model_id=%s: %s", self.model_id, unexpected[:8])

        return model

    # endregion Runtime Builder

    # region Config Parsing

    def read_timm_config_json(self, config_path: Path) -> dict[str, Any]:
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError(f"config.json at '{config_path}' is not a JSON object")
        except Exception as exc:
            raise RuntimeError(f"Failed to parse config.json at '{config_path}': {exc}") from exc

    def read_timm_preprocess_json(self, preprocess_path: Path) -> list[dict[str, Any]]:
        try:
            with preprocess_path.open("r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            if not isinstance(parsed, dict):
                raise RuntimeError(f"preprocess.json at '{preprocess_path}' is not a JSON object")  # noqa: TRY004

            raw_steps = parsed.get("test")
            if not isinstance(raw_steps, list):
                raise RuntimeError(f"preprocess.json at '{preprocess_path}' missing required 'test' list")  # noqa: TRY004

            steps = [dict(item) for item in raw_steps if isinstance(item, dict)]
            if not steps:
                raise RuntimeError(f"preprocess.json at '{preprocess_path}' has no usable test steps")

            logger.info("Using preprocess.json inference pipeline (test) for model_id=%s", self.model_id)
            return steps
        except Exception as exc:
            raise RuntimeError(f"Failed to parse preprocess.json at '{preprocess_path}': {exc}") from exc

    def resolve_timm_preprocess_steps(
        self,
        config: dict[str, Any],
        preprocess_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        if preprocess_path is not None:
            return self.read_timm_preprocess_json(preprocess_path)
        return self.build_timm_preprocess_steps_from_config(config)

    def build_timm_preprocess_steps_from_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        cfg = config.get("pretrained_cfg")
        if not isinstance(cfg, dict):
            cfg = config.get("pretrained_cfg_overlay")
        if not isinstance(cfg, dict):
            cfg = config

        input_size = cfg.get("input_size") or cfg.get("test_input_size")
        crop_pct = _float_or_none(cfg.get("crop_pct"))

        # Resolve mean/std: 1) config.json -> 2) class attribute -> 3) ImageNet default with warning
        mean = _triplet_or_none(cfg.get("mean")) or self.FALLBACK_MEAN
        std = _triplet_or_none(cfg.get("std")) or self.FALLBACK_STD

        if mean is None or std is None:
            logger.warning(
                "Model '%s' config.json lacks mean/std and no class FALLBACK_MEAN/STD defined. "
                "Defaulting to standard ImageNet normalization.",
                self.model_id,
            )
            mean = mean or (0.485, 0.456, 0.406)
            std = std or (0.229, 0.224, 0.225)

        interpolation = str(cfg.get("interpolation", "bicubic"))

        image_size = _image_size_from_input_size(input_size)
        if image_size is None:
            raise RuntimeError(f"Could not resolve timm preprocess size from config.json for model '{self.model_id}'.")

        resize_size = image_size
        if crop_pct and 0 < crop_pct < 1:
            resize_size = round(image_size / crop_pct)

        steps: list[dict[str, Any]] = [
            {"type": "resize", "size": resize_size, "interpolation": interpolation},
        ]
        if resize_size != image_size:
            steps.append({"type": "center_crop", "size": [image_size, image_size]})
        steps.extend(
            [
                {"type": "maybe_to_tensor"},
                {"type": "normalize", "mean": list(mean), "std": list(std)},
            ]
        )
        logger.info("Using config.json timm preprocess pipeline for model_id=%s", self.model_id)
        return steps

    # endregion Config Parsing

    # region Native timm

    def prepare_timm_runtime_preprocess(
        self,
        config: dict[str, Any],
        preprocess_path: Path | None = None,
        *,
        prefer_timm: bool = False,
    ) -> None:
        self._runtime_timm_transform = None
        if prefer_timm and self._prepare_native_timm_transform(config):
            return
        self._runtime_preprocess_steps = self.resolve_timm_preprocess_steps(config, preprocess_path)

    def _prepare_native_timm_transform(self, config: dict[str, Any]) -> bool:
        try:
            from timm.data import create_transform, resolve_data_config
        except ImportError:
            logger.info(
                "timm is not available for model_id=%s; using manual timm config parser for preprocessing.",
                self.model_id,
            )
            return False

        try:
            cfg = config.get("pretrained_cfg")
            if not isinstance(cfg, dict):
                cfg = config.get("pretrained_cfg_overlay")
            if not isinstance(cfg, dict):
                cfg = config
            data_config = resolve_data_config(cfg)
            self._runtime_timm_transform = create_transform(**data_config)
            logger.info("Using native timm preprocessing for model_id=%s", self.model_id)
            return True
        except Exception as exc:
            logger.warning(
                "Native timm preprocessing setup failed for model_id=%s; using manual config parser instead: %s",
                self.model_id,
                exc,
            )
            return False

    # endregion Native timm

    # region Model helpers

    def create_timm_model(self, timm_module: Any, architecture: str, model_args: dict[str, Any]) -> Any:
        try:
            return timm_module.create_model(architecture, pretrained=False, **model_args)
        except TypeError as exc:
            num_classes = model_args.get("num_classes")
            logger.warning(
                "timm.create_model rejected model_args for model_id=%s architecture=%s (%s). "
                "Retrying with conservative fallback args.",
                self.model_id,
                architecture,
                exc,
            )
            fallback_args = {}
            if num_classes is not None:
                fallback_args["num_classes"] = int(num_classes)
            return timm_module.create_model(architecture, pretrained=False, **fallback_args)

    def resolve_timm_architecture(self, config: dict[str, Any] | None) -> str | None:
        if config:
            for key in ("architecture", "model_type", "model_name", "arch"):
                value = config.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        repo = getattr(self, "default_repo_id", None) or ""
        suffix = repo.split("/", 1)[-1]
        if ".dbv" in suffix:
            fallback_arch = suffix.split(".dbv", 1)[0]
            logger.info("Using fallback timm architecture '%s' for model_id=%s", fallback_arch, self.model_id)
            return fallback_arch
        return None

    def resolve_timm_model_args(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config is not None and isinstance(config.get("model_args"), dict):
            args = dict(config["model_args"])
        else:
            args = dict(self.FALLBACK_TIMM_MODEL_ARGS)

        for key in ("pretrained_cfg", "pretrained_cfg_overlay"):
            value = config.get(key) if config else None
            if isinstance(value, dict):
                args.setdefault(key, value)
        return args

    # endregion Model helpers

    # region Preprocess

    def preprocess(self, image: Any) -> np.ndarray:
        if getattr(self, "_runtime_timm_transform", None) is not None:
            return self.preprocess_with_native_timm(image, self._runtime_timm_transform)
        return self.preprocess_with_timm_steps(image, getattr(self, "_runtime_preprocess_steps", []))

    def preprocess_with_native_timm(self, image: Any, transform: Any) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = self._to_rgb_with_background(image)
        tensor = transform(image)
        try:
            import torch

            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
        except ImportError:
            pass
        arr = np.asarray(tensor, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        return arr.astype(np.float32, copy=False)

    def preprocess_with_timm_steps(self, image: Any, steps: list[dict[str, Any]]) -> np.ndarray:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = self._to_rgb_with_background(image)

        normalize_mean: tuple[float, float, float] | None = None
        normalize_std: tuple[float, float, float] | None = None

        for step in steps:
            step_type = str(step.get("type", "")).strip().lower()
            if step_type == "pad_to_size":
                size = step.get("size")
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    image = self._pad_to_size(
                        image,
                        target_h=int(size[0]),
                        target_w=int(size[1]),
                        interpolation=str(step.get("interpolation", "bilinear")),
                        background_color=step.get("background_color", "white"),
                    )
            elif step_type == "resize":
                image = self._resize_like_torchvision(
                    image,
                    size=step.get("size"),
                    interpolation=str(step.get("interpolation", "bilinear")),
                )
            elif step_type == "center_crop":
                image = self._center_crop(image, size=step.get("size"))
            elif step_type == "normalize":
                normalize_mean = _triplet_or_none(step.get("mean"))
                normalize_std = _triplet_or_none(step.get("std"))

        arr = np.asarray(image, dtype=np.float32) / 255.0
        if normalize_mean is not None and normalize_std is not None:
            mean_arr = np.asarray(normalize_mean, dtype=np.float32).reshape(1, 1, 3)
            std_arr = np.asarray(normalize_std, dtype=np.float32).reshape(1, 1, 3)
            arr = (arr - mean_arr) / std_arr

        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0).astype(np.float32, copy=False)

    # endregion Preprocess

    # region Image Helpers

    def _to_rgb_with_background(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            return background
        if image.mode == "P" and "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[3])
            return background
        return image.convert("RGB")

    def _pad_to_size(
        self,
        image: Image.Image,
        *,
        target_h: int,
        target_w: int,
        interpolation: str,
        background_color: Any,
    ) -> Image.Image:
        source_w, source_h = image.size
        if source_w <= 0 or source_h <= 0 or target_w <= 0 or target_h <= 0:
            return image
        ratio = min(target_w / source_w, target_h / source_h)
        resized_w = max(1, round(source_w * ratio))
        resized_h = max(1, round(source_h * ratio))
        resized = image.resize((resized_w, resized_h), self._pil_interpolation(interpolation))
        result = Image.new("RGB", (target_w, target_h), self._resolve_background_color(background_color))
        result.paste(resized, ((target_w - resized_w) // 2, (target_h - resized_h) // 2))
        return result

    def _resize_like_torchvision(self, image: Image.Image, *, size: Any, interpolation: str) -> Image.Image:
        resample = self._pil_interpolation(interpolation)
        source_w, source_h = image.size
        if isinstance(size, int):
            if source_w <= 0 or source_h <= 0:
                return image
            if source_w < source_h:
                return image.resize((size, max(1, round((source_h / source_w) * size))), resample)
            return image.resize((max(1, round((source_w / source_h) * size)), size), resample)
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return image.resize((max(1, int(size[1])), max(1, int(size[0]))), resample)
        return image

    def _center_crop(self, image: Image.Image, *, size: Any) -> Image.Image:
        if isinstance(size, int):
            crop_h = size
            crop_w = size
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            crop_h = int(size[0])
            crop_w = int(size[1])
        else:
            return image
        source_w, source_h = image.size
        crop_w = min(max(1, crop_w), source_w)
        crop_h = min(max(1, crop_h), source_h)
        left = max(0, (source_w - crop_w) // 2)
        top = max(0, (source_h - crop_h) // 2)
        return image.crop((left, top, left + crop_w, top + crop_h))

    def _pil_interpolation(self, name: str) -> int:
        normalized = str(name).strip().lower()
        if normalized == "nearest":
            return Image.Resampling.NEAREST
        if normalized == "bicubic":
            return Image.Resampling.BICUBIC
        if normalized == "lanczos":
            return Image.Resampling.LANCZOS
        return Image.Resampling.BILINEAR

    def _resolve_background_color(self, value: Any) -> tuple[int, int, int]:
        if isinstance(value, str):
            if value.strip().lower() == "black":
                return (0, 0, 0)
            return (255, 255, 255)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return (int(value[0]), int(value[1]), int(value[2]))
        return (255, 255, 255)

    # endregion Image Helpers


# region Helper Functions


def _triplet_or_none(raw: Any) -> tuple[float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    try:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None


def _float_or_none(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _image_size_from_input_size(raw: Any) -> int | None:
    if isinstance(raw, int) and raw > 0:
        return raw
    if isinstance(raw, (list, tuple)) and raw:
        values = [int(value) for value in raw if isinstance(value, int) and value > 0]
        if len(values) >= 2:
            return values[-1]
    return None


# endregion Helper Functions
