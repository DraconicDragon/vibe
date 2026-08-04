"""Centralized programmatic configuration for vibe."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ONNXConfig:
    providers: list[str] | None = None


@dataclass
class PyTorchConfig:
    cudnn_enabled: bool = True


@dataclass
class JTPHydraConfig:
    seqlen: int = 1024


@dataclass
class PluginConfig:
    jtp_hydra: JTPHydraConfig = field(default_factory=JTPHydraConfig)


class VibeConfig:
    """Global configuration state holding defaults for inference sessions."""

    def __init__(self) -> None:
        self.onnx = ONNXConfig()
        self.pytorch = PyTorchConfig()
        self.plugins = PluginConfig()


config = VibeConfig()
