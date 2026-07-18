"""
ModelPlugin — core abstraction and metadata definitions.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe.results import ModelResult, OutputType

if TYPE_CHECKING:
    from vibe.result_processors import ResultProcessor


class FileRole(str, Enum):
    WEIGHTS = "weights"
    TAG_LIST = "tag_list"
    MAPPING = "mapping"
    CONFIG = "config"


class Backend(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"


@dataclass(frozen=True)
class ModelDescriptor:
    """Consolidated metadata description for consumption by external UIs and APIs."""

    model_id: str
    display_name: str
    family_name: str
    description: str
    output_type: OutputType
    supported_backends: tuple[Backend, ...]
    supported_processors: tuple[str, ...]
    variants: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ArtifactSpec:
    """A logical file required by the model. Identity is driven by 'id', not 'name'."""

    id: str
    name: str  # Default download/lookup filename
    role: FileRole
    required: bool = True
    repo_id: str | None = None
    hf_subdir: str | None = None


@dataclass(frozen=True)
class ModelVariant:
    """Groups artifacts required for a specific backend and execution environment."""

    backend: Backend
    artifacts: tuple[ArtifactSpec, ...]
    repo_id: str | None = None
    hf_subdir: str | None = None


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    display_name: str
    family_name: str = ""
    description: str = ""


@dataclass(frozen=True)
class ModelCapabilities:
    output_type: OutputType = OutputType.TAGS
    supported_processors: tuple[type["ResultProcessor"], ...] = ()


class ArtifactMap:
    """A strictly ID-keyed mapping of resolved paths. Plugins never index by filename."""

    def __init__(self, paths_by_id: dict[str, Path], optional_missing: dict[str, str] | None = None):
        self._paths = paths_by_id
        self._optional_missing = optional_missing or {}

    def get(self, artifact_id: str) -> Path:
        if artifact_id not in self._paths:
            raise KeyError(f"Artifact '{artifact_id}' was not resolved. Available: {list(self._paths)}")
        return self._paths[artifact_id]

    def get_optional(self, artifact_id: str) -> Path | None:
        return self._paths.get(artifact_id)

    def as_path_dict(self) -> dict[str, Path]:
        return dict(self._paths)


class ModelPlugin(ABC):
    """
    Abstract base class for all vibe model plugins.
    Concrete subclasses MUST define `identity`, `capabilities`, and `variants`.
    """

    identity: ModelIdentity
    capabilities: ModelCapabilities
    variants: tuple[ModelVariant, ...]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Automatically skip validation for abstract classes
        if not inspect.isabstract(cls):
            if not hasattr(cls, "identity") or not cls.identity.model_id:
                raise ValueError(f"Concrete plugin {cls.__name__} must define 'identity' with a valid model_id.")
            if not hasattr(cls, "variants") or not cls.variants:
                raise ValueError(f"Concrete plugin {cls.__name__} must define at least one ModelVariant.")

            from vibe.registry import model_registry

            try:
                model_registry.register(cls)
            except ValueError as exc:
                import warnings

                warnings.warn(str(exc), stacklevel=2)

    def configure(self, **kwargs: Any) -> None:
        """Optional per-session configuration hook."""
        pass

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Load tag lists, mappings, etc., using strict artifact IDs."""
        pass

    @abstractmethod
    def preprocess(self, image: Any) -> Any:
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any) -> ModelResult:
        pass

    @classmethod
    def describe(cls) -> ModelDescriptor:
        """Assembles a structured descriptor of the model plugin's metadata."""
        family = getattr(cls, "family_name", "") or cls.identity.family_name or cls.identity.display_name

        return ModelDescriptor(
            model_id=cls.identity.model_id,
            display_name=cls.identity.display_name,
            family_name=family,
            description=cls.identity.description,
            output_type=cls.capabilities.output_type,
            supported_backends=tuple(v.backend for v in cls.variants),
            supported_processors=tuple(p.__name__ for p in cls.capabilities.supported_processors),
            variants=tuple(
                {
                    "backend": v.backend.value,
                    "repo_id": v.repo_id or getattr(cls, "default_repo_id", None),
                    "hf_subdir": v.hf_subdir,
                    "artifacts": [
                        {
                            "id": a.id,
                            "name": a.name,
                            "role": a.role.value,
                            "required": a.required,
                        }
                        for a in v.artifacts
                    ],
                }
                for v in cls.variants
            ),
        )
