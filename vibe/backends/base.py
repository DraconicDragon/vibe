"""
ModelPlugin — core abstraction and metadata definitions.
"""

from __future__ import annotations

import dataclasses
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

from vibe.results import ModelResult, OutputType

if TYPE_CHECKING:
    from vibe.result_transforms import ResultTransform


class FileRole(str, Enum):
    WEIGHTS = "weights"
    TAG_LIST = "tag_list"
    MAPPING = "mapping"
    CONFIG = "config"


class Backend(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"


@dataclass(frozen=True)
class ExecutionRequest:
    """Resolved execution settings passed to a plugin's runtime builder."""

    backend: Backend
    device: str
    precision: str
    onnx_providers: tuple[str, ...] | None = None


class RuntimeExecutor(Protocol):
    """The small contract the inference layer needs from a loaded runtime."""

    def run(self, inputs: Any) -> Any: ...

    def close(self) -> None: ...

    def supports_true_batching(self) -> bool: ...


@dataclass(frozen=True)
class ModelDescriptor:
    """Consolidated metadata description for consumption by external UIs and APIs."""

    model_id: str
    display_name: str
    family_name: str
    description: str
    output_type: OutputType
    output_categories: tuple[str, ...]
    output_extras: dict[str, str]
    entry_extras: dict[str, str]
    supported_backends: tuple[Backend, ...]
    supported_transforms: tuple[str, ...]
    recommended_transforms: dict[str, dict[str, Any]]  # transform_id -> dict of recommended values
    default_repo_id: str
    variants: tuple[ModelVariant, ...]


@dataclass(frozen=True)
class ArtifactSpec:
    """A logical file required by the model. Identity is driven by 'id', not 'name'."""

    id: str
    name: str  # Default download/lookup filename
    role: FileRole
    required: bool = True
    repo_id: str | None = None
    hf_subdir: str | None = None

    def resolve(self, fallback_repo_id: str) -> ArtifactSpec:
        """Return a resolved copy of this artifact with fallback repo_ids populated."""
        return dataclasses.replace(self, repo_id=self.repo_id or fallback_repo_id)


@dataclass(frozen=True)
class ModelVariant:
    """Groups artifacts required for a specific backend and execution environment."""

    backend: Backend
    artifacts: tuple[ArtifactSpec, ...]
    repo_id: str | None = None
    hf_subdir: str | None = None

    def resolve(self, fallback_repo_id: str) -> ModelVariant:
        """Return a resolved copy of this variant and its children with cascading fallbacks."""
        v_repo = self.repo_id or fallback_repo_id
        return dataclasses.replace(self, repo_id=v_repo, artifacts=tuple(art.resolve(v_repo) for art in self.artifacts))


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    display_name: str
    description: str = ""


@dataclass(frozen=True)
class ModelCapabilities:
    output_type: OutputType
    output_categories: tuple[str, ...] = ()
    # Top-level extras
    output_extras: dict[str, str] = field(default_factory=dict)
    # Per-entry extras
    entry_extras: dict[str, str] = field(default_factory=dict)
    transforms: tuple[type[ResultTransform] | ResultTransform, ...] = ()

    def with_transforms(self, *overrides: type[ResultTransform] | ResultTransform) -> Self:
        """Return a copy with specified transforms added or replaced by their transform_id."""
        override_map = {o.transform_id: o for o in overrides}
        seen_overrides = set()
        new_transforms = []

        for t in self.transforms:
            tid = getattr(t, "transform_id", None)
            if tid in override_map:
                new_transforms.append(override_map[tid])
                seen_overrides.add(tid)
            else:
                new_transforms.append(t)

        for tid, o in override_map.items():
            if tid not in seen_overrides:
                new_transforms.append(o)

        return dataclasses.replace(self, transforms=tuple(new_transforms))


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

    @property
    def optional_missing(self) -> dict[str, str]:
        return self._optional_missing


class ModelPlugin(ABC):
    """
    Abstract base class for all vibe model plugins.
    Concrete subclasses MUST define `identity`, `capabilities`, and `variants`.
    """

    family_name: str
    identity: ModelIdentity
    capabilities: ModelCapabilities
    default_repo_id: str
    variants: tuple[ModelVariant, ...]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Check if this specific subclass declared an identity, signaling intent to be concrete
        is_concrete_intent = "identity" in cls.__dict__

        if inspect.isabstract(cls):
            if is_concrete_intent:
                missing = sorted(cls.__abstractmethods__)
                raise TypeError(
                    f"Plugin class '{cls.__name__}' declared an identity but is missing concrete "
                    f"implementations for abstract methods: {missing}"
                )
            return  # Quietly skip intermediate abstract base classes (e.g. WDTaggerBasePlugin)

        if not getattr(cls, "family_name", None):
            raise ValueError(f"Concrete plugin {cls.__name__} must inherit or define a 'family_name' string.")
        if not hasattr(cls, "identity") or not cls.identity.model_id:
            raise ValueError(f"Concrete plugin {cls.__name__} must define 'identity' with a valid model_id.")
        if not getattr(cls, "default_repo_id", None):
            raise ValueError(f"Concrete plugin {cls.__name__} must define a valid 'default_repo_id' string.")
        if not hasattr(cls, "variants") or not cls.variants:
            raise ValueError(f"Concrete plugin {cls.__name__} must define at least one ModelVariant.")

        from vibe.registry import model_registry

        try:
            model_registry.register(cls)
        except ValueError as exc:
            import warnings

            warnings.warn(str(exc), stacklevel=2)

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Initialize plugin-local metadata from resolved artifacts.

        This hook must not configure or mutate a runtime executor. It remains
        separate from `build_runtime` so every session has its own plugin
        state even when a completed runtime is shared from the pool.
        """

    def build_runtime(self, artifacts: ArtifactMap, request: ExecutionRequest) -> RuntimeExecutor:
        """Build a fully initialized runtime for this model and request.

        This is intentionally not abstract during the metadata migration:
        making it abstract would make old concrete plugins unimportable before
        they can be migrated.
        """
        raise NotImplementedError(
            f"Plugin '{self.identity.model_id}' has not migrated to the build_runtime() contract."
        )

    @abstractmethod
    def preprocess(self, image: Any) -> Any:
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any) -> ModelResult:
        pass

    @classmethod
    def describe(cls) -> ModelDescriptor:
        """Assembles a structured descriptor of the model plugin's metadata."""
        resolved_variants = tuple(v.resolve(cls.default_repo_id) for v in cls.variants)

        supported_ids = []
        recommended_configs = {}

        from vibe.result_transforms import ResultTransform

        for t in cls.capabilities.transforms:
            if isinstance(t, type) and issubclass(t, ResultTransform):
                supported_ids.append(t.transform_id)
            elif isinstance(t, ResultTransform):
                supported_ids.append(t.transform_id)
                recommended_configs[t.transform_id] = t.to_config_dict()

        supported_ids = list(dict.fromkeys(supported_ids))

        return ModelDescriptor(
            model_id=cls.identity.model_id,
            display_name=cls.identity.display_name,
            family_name=cls.family_name,
            description=cls.identity.description,
            output_type=cls.capabilities.output_type,
            output_categories=cls.capabilities.output_categories,
            output_extras=cls.capabilities.output_extras,
            entry_extras=cls.capabilities.entry_extras,
            supported_backends=tuple(v.backend for v in cls.variants),
            supported_transforms=tuple(supported_ids),
            recommended_transforms=recommended_configs,
            default_repo_id=cls.default_repo_id,
            variants=resolved_variants,
        )
