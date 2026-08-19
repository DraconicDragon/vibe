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
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

from vibe.features import FeatureSpec, InferenceRequest
from vibe.precision import PrecisionRequest
from vibe.results import ModelResult, OutputType

if TYPE_CHECKING:
    from vibe.result_transforms import PluginData


class FileRole(str, Enum):
    WEIGHTS = "weights"
    TAG_LIST = "tag_list"
    MAPPING = "mapping"
    CONFIG = "config"


class Backend(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"


class HardwareIntent(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    ACCELERATOR = "accelerator"


@dataclass(frozen=True)
class ExecutionPreference:
    """Universal hardware intent, replacing framework-specific device strings."""

    intent: HardwareIntent
    ordinal: int | None = None
    hint: str | None = None  # Preserves specific framework hints like "mps" or "rocm"

    @classmethod
    def parse(cls, value: str | None) -> ExecutionPreference:
        if not value:
            return cls(HardwareIntent.AUTO)

        val = str(value).strip().lower()
        if val in ("auto", ""):
            return cls(HardwareIntent.AUTO)
        if val == "cpu":
            return cls(HardwareIntent.CPU)

        # Parse legacy/framework strings (e.g. "cuda:0", "xpu:0", "gpu:1", "mps", "rocm", "openvino")
        parts = val.split(":", 1)
        base = parts[0]
        ordinal = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None

        hint = base if base not in ("gpu", "cuda") else None
        return cls(HardwareIntent.ACCELERATOR, ordinal=ordinal, hint=hint)


@dataclass(frozen=True)
class ExecutionPlan:
    """The factory's resolved choices for execution (intent)."""

    backend: Backend
    preference: ExecutionPreference
    precision: PrecisionRequest
    variant_id: str | None = None
    onnx_providers: tuple[str, ...] | None = None
    hf_token: str | None = None


class RuntimeExecutor(Protocol):
    """The small contract the inference layer needs from a loaded runtime."""

    def run(self, inputs: Any) -> Any: ...
    def close(self) -> None: ...
    def supports_true_batching(self) -> bool: ...
    def execution_info(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ArtifactSpec:
    """A logical file required by the model."""

    id: str
    name: str  # Default download/lookup filename
    role: FileRole
    required: bool = True
    repo_id: str | None = None
    hf_subdir: str | None = None

    def resolve(self, fallback_repo_id: str, fallback_hf_subdir: str | None = None) -> ArtifactSpec:
        """Return a resolved copy of this artifact with fallback repo_ids and hf_subdirs populated."""
        return dataclasses.replace(
            self,
            repo_id=self.repo_id or fallback_repo_id,
            hf_subdir=self.hf_subdir or fallback_hf_subdir,
        )


@dataclass(frozen=True)
class ModelVariant:
    """Groups artifacts required for a specific backend and execution environment."""

    backend: Backend
    artifacts: tuple[ArtifactSpec, ...]
    variant_id: str | None = None
    description: str = ""
    repo_id: str | None = None
    hf_subdir: str | None = None

    def resolve(self, fallback_repo_id: str) -> ModelVariant:
        """Return a resolved copy of this variant and its children with cascading fallbacks."""
        v_repo = self.repo_id or fallback_repo_id
        v_subdir = self.hf_subdir
        return dataclasses.replace(
            self,
            repo_id=v_repo,
            artifacts=tuple(art.resolve(v_repo, v_subdir) for art in self.artifacts),
        )


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    display_name: str
    description: str


@dataclass(frozen=True)
class ModelCapabilities:
    output_type: OutputType
    output_categories: tuple[str, ...] = ()
    # Top-level extras
    output_extras: dict[str, str] = field(default_factory=dict)
    # Per-entry extras
    entry_extras: dict[str, str] = field(default_factory=dict)
    features: tuple[FeatureSpec, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(feature, FeatureSpec) for feature in self.features):
            raise TypeError("ModelCapabilities.features must contain only FeatureSpec instances.")

        output_categories = tuple(
            str(category.value if isinstance(category, Enum) else category) for category in self.output_categories
        )
        object.__setattr__(self, "output_categories", output_categories)
        object.__setattr__(
            self,
            "features",
            tuple(dataclasses.replace(feature, _output_categories=output_categories) for feature in self.features),
        )

        feature_ids = [feature.id for feature in self.features]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError(f"ModelCapabilities contains duplicate feature IDs: {feature_ids}")

        config_types = [feature.config_type for feature in self.features]
        if len(config_types) != len(set(config_types)):
            raise ValueError("ModelCapabilities cannot bind one configuration type to multiple features.")

    def with_features(self, *overrides: FeatureSpec) -> Self:
        """Return a copy with specified feature specs added or replaced."""
        override_ids = [override.id for override in overrides]
        if len(override_ids) != len(set(override_ids)):
            raise ValueError(f"Duplicate feature IDs in overrides: {override_ids}")
        override_map = {o.id: o for o in overrides}
        seen = set()
        new_features = []

        for f in self.features:
            if f.id in override_map:
                new_features.append(override_map[f.id])
                seen.add(f.id)
            else:
                new_features.append(f)

        for o in overrides:
            if o.id not in seen:
                new_features.append(o)
                seen.add(o.id)

        return dataclasses.replace(self, features=tuple(new_features))


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
    features: tuple[dict[str, Any], ...]
    default_repo_id: str | None
    variants: tuple[ModelVariant, ...]


class ArtifactMap:
    """Strictly ID-keyed mapping of resolved file paths."""

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
    def cache_key(self) -> tuple[tuple[str, str], ...]:
        """Return a deterministic, immutable identity for the resolved artifacts."""
        return tuple(sorted((artifact_id, str(path.resolve())) for artifact_id, path in self._paths.items()))

    @property
    def optional_missing(self) -> dict[str, str]:
        return self._optional_missing


class ModelPlugin(ABC):
    """Abstract base class for all vibe model plugins."""

    family_name: str
    identity: ModelIdentity
    capabilities: ModelCapabilities
    default_repo_id: str
    variants: tuple[ModelVariant, ...]
    custom_only: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Skip abstract classes (e.g. ModelPlugin itself)
        if inspect.isabstract(cls):
            return

        # Intermediate base classes (e.g. WDTaggerBasePlugin) do not define an identity.
        # Only register classes that possess a fully declared ModelIdentity.
        identity = getattr(cls, "identity", None)
        if identity is None:
            return

        # Validate required class-level metadata on concrete models
        if not getattr(cls, "family_name", None):
            raise ValueError(f"Concrete plugin '{cls.__name__}' must inherit or define a 'family_name' string.")
        if not getattr(cls, "default_repo_id", None) and not getattr(cls, "custom_only", False):
            raise ValueError(f"Concrete plugin '{cls.__name__}' must define a valid 'default_repo_id' string.")

        variants = getattr(cls, "variants", None)
        if not variants:
            raise ValueError(f"Concrete plugin '{cls.__name__}' must define at least one ModelVariant.")

        seen_variant_ids: set[str] = set()
        backend_counts: dict[Backend, int] = {}

        for v in variants:
            backend_counts[v.backend] = backend_counts.get(v.backend, 0) + 1

        for v in variants:
            if v.variant_id:
                if v.variant_id in seen_variant_ids:
                    raise ValueError(f"Concrete plugin '{cls.__name__}' defines duplicate variant_id '{v.variant_id}'.")
                seen_variant_ids.add(v.variant_id)

            # If a backend has >1 variant, ALL variants for that backend MUST declare a unique variant_id
            if backend_counts[v.backend] > 1 and not v.variant_id:
                raise ValueError(
                    f"Concrete plugin '{cls.__name__}' defines multiple variants for backend '{v.backend.value}'. "
                    f"Every variant for backend '{v.backend.value}' MUST declare a unique 'variant_id'."
                )

        from vibe.registry import model_registry

        try:
            model_registry.register(cls)
        except ValueError as exc:
            import warnings

            warnings.warn(str(exc), stacklevel=2)

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Initialize plugin-local static metadata from resolved artifacts."""

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        """Build a fully initialized runtime for this model and execution plan."""
        raise NotImplementedError(
            f"Plugin '{self.identity.model_id}' has not implemented the build_runtime() contract."
        )

    def provide_transform_data(self) -> tuple[PluginData, ...]:
        """Optional hook: return static data precomputed by this plugin for transforms."""
        return ()

    def collate_batch(self, samples: list[Any]) -> Any:
        """Collate a list of preprocessed samples into a batch tensor."""
        if not samples:
            raise ValueError("Cannot collate an empty batch.")

        first = samples[0]

        import numpy as np

        if isinstance(first, np.ndarray):
            return np.concatenate(samples, axis=0)

        try:
            import torch

            if isinstance(first, torch.Tensor):
                return torch.cat(samples, dim=0)
        except ImportError:
            pass

        raise TypeError(
            f"Default collate_batch unsupported for type {type(first).__name__}. "
            f"Plugin {self.identity.model_id} must override this method."
        )

    def split_batch(self, batched_output: Any, expected_size: int) -> list[Any]:
        """Split a batched raw output back into a list of per-sample outputs."""
        if expected_size == 1:
            return [batched_output]

        # Handle nested dictionaries
        if isinstance(batched_output, dict):
            keys = list(batched_output.keys())
            split_vals = {k: self.split_batch(v, expected_size) for k, v in batched_output.items()}
            return [{k: split_vals[k][i] for k in keys} for i in range(expected_size)]

        # Handle nested tuples/lists
        if isinstance(batched_output, (tuple, list)):
            split_vals = [self.split_batch(v, expected_size) for v in batched_output]
            if isinstance(batched_output, tuple):
                results: list[Any] = []
                for i in range(expected_size):
                    values = [v[i] for v in split_vals]
                    # NamedTuple subclasses require positional construction;
                    # plain tuples take one iterable argument.
                    if hasattr(batched_output, "_fields"):
                        results.append(type(batched_output)(*values))
                    else:
                        results.append(tuple(values))
                return results
            return [type(batched_output)(v[i] for v in split_vals) for i in range(expected_size)]

        shape = getattr(batched_output, "shape", None)
        ndim = getattr(batched_output, "ndim", None)

        if ndim == 0:
            return [batched_output for _ in range(expected_size)]

        if shape is not None and len(shape) > 0 and shape[0] == expected_size:
            return [batched_output[i : i + 1] for i in range(expected_size)]

        import numpy as np

        try:
            arr = np.asarray(batched_output)
        except Exception as exc:
            raise TypeError(f"Default split_batch expected array-like, got {type(batched_output).__name__}") from exc

        if arr.ndim == 0:
            return [arr for _ in range(expected_size)]
        if arr.shape[0] == expected_size:
            return [arr[i : i + 1] for i in range(expected_size)]

        raise ValueError(f"Batch dimension mismatch: expected {expected_size}, got shape {arr.shape}.")

    @abstractmethod
    def preprocess(self, image: Any, request: InferenceRequest | None = None) -> Any:
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any) -> ModelResult:
        pass

    @classmethod
    def describe(cls) -> ModelDescriptor:
        """Assembles a structured descriptor of the model plugin's metadata."""
        resolved_variants = tuple(v.resolve(cls.default_repo_id) for v in cls.variants)

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
            features=tuple(f.to_dict() for f in cls.capabilities.features),
            default_repo_id=None if cls.custom_only and not cls.default_repo_id else cls.default_repo_id,
            variants=resolved_variants,
        )
