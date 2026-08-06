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

from vibe.precision import PrecisionRequest
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


class RuntimeExecutor(Protocol):
    """The small contract the inference layer needs from a loaded runtime."""

    def run(self, inputs: Any) -> Any: ...
    def close(self) -> None: ...
    def supports_true_batching(self) -> bool: ...
    def execution_info(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PluginOptionSpec:
    key: str
    type: type[int | float | str | bool]
    default: int | float | str | bool
    display_name: str
    description: str
    choices: tuple[Any, ...] | None = None
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "type": self.type.__name__,
            "default": self.default,
            "display_name": self.display_name,
            "description": self.description,
            "choices": self.choices,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "step": self.step,
        }


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
    variant_id: str | None = None
    description: str = ""
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
    options: tuple[PluginOptionSpec, ...] = ()

    def with_transforms(self, *overrides: type[ResultTransform] | ResultTransform) -> Self:
        """Return a copy with specified transforms added or replaced by their transform_id."""
        override_map = {o.transform_id: o for o in overrides}
        seen_overrides = set()
        new_transforms = []

        for t in self.transforms:
            tid = t.transform_id
            if tid in override_map:
                new_transforms.append(override_map[tid])
                seen_overrides.add(tid)
            else:
                new_transforms.append(t)

        for tid, o in override_map.items():
            if tid not in seen_overrides:
                new_transforms.append(o)

        return dataclasses.replace(self, transforms=tuple(new_transforms))


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
    options: tuple[dict[str, Any], ...]
    default_repo_id: str
    variants: tuple[ModelVariant, ...]


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

        # Skip abstract classes (e.g. ModelPlugin itself)
        if inspect.isabstract(cls):
            return

        # Intermediate base classes (e.g. WDTaggerBasePlugin) do not define an identity.
        # Only register classes that possess a fully declared ModelIdentity.
        identity = getattr(cls, "identity", None)
        if identity is None or not getattr(identity, "model_id", None):
            return

        # Validate required metadata on concrete models
        if not getattr(cls, "family_name", None):
            raise ValueError(f"Concrete plugin '{cls.__name__}' must inherit or define a 'family_name' string.")
        if not getattr(cls, "default_repo_id", None):
            raise ValueError(f"Concrete plugin '{cls.__name__}' must define a valid 'default_repo_id' string.")

        variants = getattr(cls, "variants", None)
        if not variants:
            raise ValueError(f"Concrete plugin '{cls.__name__}' must define at least one ModelVariant.")

        # region Variant Validation
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
        # endregion Variant Validation

        from vibe.registry import model_registry

        try:
            model_registry.register(cls)
        except ValueError as exc:
            import warnings

            warnings.warn(str(exc), stacklevel=2)

    def get_option(self, key: str) -> Any:
        """Fetch an option value from global vibe.config.plugins, falling back to declared default."""
        from vibe.config import config

        declared_specs = {spec.key: spec for spec in self.capabilities.options}

        if key not in declared_specs:
            available = list(declared_specs.keys())
            raise KeyError(
                f"Plugin '{self.identity.model_id}' attempted to access undeclared option '{key}'. "
                f"Available options for this model: {available if available else 'None'}"
            )

        spec = declared_specs[key]
        return config.plugins.get(key, spec.default)

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Initialize plugin-local metadata from resolved artifacts.

        This hook must not configure or mutate a runtime executor. It remains
        separate from `build_runtime` so every session has its own plugin
        state even when a completed runtime is shared from the pool.
        """

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        """Build a fully initialized runtime for this model and execution plan.

        This is intentionally not abstract during the metadata migration:
        making it abstract would make old concrete plugins unimportable before
        they can be migrated.
        """
        raise NotImplementedError(
            f"Plugin '{self.identity.model_id}' has not migrated to the build_runtime() contract."
        )

    def collate_batch(self, samples: list[Any]) -> Any:
        """Collate a list of preprocessed samples into a batch tensor.

        The default implementation handles standard NumPy arrays and PyTorch tensors
        that already have a batch dimension (e.g. shape (1, C, H, W)).
        """
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
        """Split a batched raw output back into a list of per-sample outputs.

        The default implementation safely slices numpy arrays, torch tensors,
        and traverses nested dictionaries/tuples recursively.
        """
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
            options=tuple(opt.to_dict() for opt in cls.capabilities.options),
            default_repo_id=cls.default_repo_id,
            variants=resolved_variants,
        )
