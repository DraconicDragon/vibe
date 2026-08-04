"""Centralized programmatic configuration for vibe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ONNXConfig:
    providers: list[str] | None = None


@dataclass
class PyTorchConfig:
    cudnn_enabled: bool = True


@dataclass
class PluginConfig:
    """Generic key-value store for plugin-specific settings."""

    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.extras[key] = value


class VibeConfig:
    """Global configuration state holding defaults for inference sessions."""

    def __init__(self) -> None:
        self.onnx = ONNXConfig()
        self.pytorch = PyTorchConfig()
        self.plugins = PluginConfig()


config = VibeConfig()
