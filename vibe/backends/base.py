"""
ModelPlugin — core abstraction and metadata definitions.
"""

from __future__ import annotations

import dataclasses
import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol

from vibe.metadata import ModelDescriptor, ModelIdentity, ModelProfile
from vibe.precision import PrecisionRequest
from vibe.results import ModelResult
from vibe.settings import InferenceRequest, SettingGroupSpec


class FileRole(StrEnum):
    """Semantic role of a required or optional model file."""

    WEIGHTS = "weights"
    TAG_LIST = "tag_list"
    MAPPING = "mapping"
    CONFIG = "config"


class Backend(StrEnum):
    """Supported execution frameworks."""

    PYTORCH = "pytorch"
    ONNX = "onnx"


class HardwareIntent(StrEnum):
    """High-level target compute device classes."""

    AUTO = "auto"
    CPU = "cpu"
    ACCELERATOR = "accelerator"


@dataclass(frozen=True, kw_only=True)
class ExecutionPreference:
    """Universal hardware intent, replacing framework-specific device strings."""

    intent: HardwareIntent
    ordinal: int | None = None
    hint: str | None = None  # Preserves specific framework hints like "cuda", "mps", or "rocm"

    @classmethod
    def parse(cls, value: str | None) -> ExecutionPreference:
        if not value:
            return cls(intent=HardwareIntent.AUTO)

        val = str(value).strip().lower()
        if val in ("auto", ""):
            return cls(intent=HardwareIntent.AUTO)

        parts = val.split(":", 1)
        base = parts[0]
        ordinal = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None

        # (e.g. "cpu" or "cpu:0")
        if base == "cpu":
            return cls(intent=HardwareIntent.CPU, ordinal=ordinal)

        # (e.g. "auto:0" or ":0")
        if base in ("auto", ""):
            return cls(intent=HardwareIntent.AUTO, ordinal=ordinal)

        # Only drop generic descriptors ("gpu", "accelerator");
        # preserve concrete hardware families ("cuda", "mps", "xpu", "rocm", etc.)
        hint = base if base not in ("gpu", "accelerator") else None
        return cls(intent=HardwareIntent.ACCELERATOR, ordinal=ordinal, hint=hint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "ordinal": self.ordinal,
            "hint": self.hint,
        }


@dataclass(frozen=True, kw_only=True)
class ExecutionPlan:
    """The factory's resolved choices for execution (intent)."""

    backend: Backend
    preference: ExecutionPreference
    precision: PrecisionRequest
    variant_id: str | None = None
    onnx_providers: tuple[str, ...] | None = None
    hf_token: str | None = field(default=None, compare=False, hash=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "variant_id": self.variant_id,
            "preference": self.preference.to_dict(),
            "precision": self.precision.to_dict(),
            "onnx_providers": list(self.onnx_providers) if self.onnx_providers is not None else None,
        }


class RuntimeExecutor(Protocol):
    """The minimal runtime contract needed by the inference session."""

    def run(self, inputs: Any) -> Any: ...
    def close(self) -> None: ...
    def supports_true_batching(self) -> bool: ...
    def execution_info(self) -> dict[str, Any]: ...


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
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


class ArtifactMap:
    """Strictly ID-keyed mapping of resolved file paths with dictionary-like ergonomics."""

    def __init__(
        self,
        paths_by_id: Mapping[str, Path],
        optional_missing: Mapping[str, str] | None = None,
        specs_by_id: Mapping[str, ArtifactSpec] | None = None,
    ) -> None:
        self._paths = dict(paths_by_id)
        self._optional_missing = dict(optional_missing) if optional_missing else {}
        self._specs = dict(specs_by_id) if specs_by_id else {}
        # Pre-compute cache key once at construction time
        self._cache_key = tuple(sorted((art_id, str(p.resolve())) for art_id, p in self._paths.items()))

    def get(self, artifact_id: str) -> Path:
        """Retrieve a required artifact's path, raising KeyError if missing."""
        if artifact_id not in self._paths:
            raise KeyError(f"Artifact '{artifact_id}' was not resolved. Available: {list(self._paths.keys())}")
        return self._paths[artifact_id]

    def get_optional(self, artifact_id: str) -> Path | None:
        """Retrieve an optional artifact's path, returning None if missing."""
        return self._paths.get(artifact_id)

    def get_spec(self, artifact_id: str) -> ArtifactSpec | None:
        """Retrieve the ArtifactSpec metadata for an artifact ID, if available."""
        return self._specs.get(artifact_id)

    def by_role(self, role: FileRole | str) -> dict[str, Path]:
        """Return a mapping of artifact ID -> Path for all resolved artifacts matching a specific role."""
        target_role = role.value if isinstance(role, FileRole) else str(role)
        return {
            art_id: path
            for art_id, path in self._paths.items()
            if self._specs.get(art_id) and self._specs[art_id].role.value == target_role
        }

    def to_dict(self) -> dict[str, str]:
        """Return a stringified path mapping suitable for serialization."""
        return {art_id: str(path) for art_id, path in self._paths.items()}

    def as_path_dict(self) -> dict[str, Path]:
        """Return a shallow copy of the internal path mapping."""
        return dict(self._paths)

    @property
    def specs(self) -> dict[str, ArtifactSpec]:
        """Mapping of artifact ID to its declared ArtifactSpec."""
        return dict(self._specs)

    @property
    def optional_missing(self) -> dict[str, str]:
        """Reasons why optional artifacts were not resolved."""
        return dict(self._optional_missing)

    @property
    def cache_key(self) -> tuple[tuple[str, str], ...]:
        """Deterministic, immutable identity for the resolved artifacts."""
        return self._cache_key

    def __getitem__(self, artifact_id: str) -> Path:
        return self.get(artifact_id)

    def __contains__(self, artifact_id: object) -> bool:
        return artifact_id in self._paths

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def items(self) -> Iterator[tuple[str, Path]]:
        return iter(self._paths.items())

    def keys(self) -> Iterator[str]:
        return iter(self._paths.keys())

    def values(self) -> Iterator[Path]:
        return iter(self._paths.values())

    def __repr__(self) -> str:
        return f"ArtifactMap({self._paths!r}, optional_missing={self._optional_missing!r})"


class ModelPlugin(ABC):
    """Abstract base class for all vibe model plugins."""

    family_name: str
    identity: ModelIdentity
    profile: ModelProfile
    settings: tuple[SettingGroupSpec, ...] = ()
    implements: tuple[type, ...] = ()  # Declarative contract of provided domain protocols
    default_repo_id: str
    variants: tuple[ModelVariant, ...]
    active_backend: Backend | None = None
    custom_only: ClassVar[bool] = False

    def bind_execution_state(self, plan: ExecutionPlan) -> None:
        """Bind the execution context for this plugin instance for the current session."""
        self.active_backend = plan.backend

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Skip abstract classes (e.g. ModelPlugin itself)
        if inspect.isabstract(cls):
            return

        # Intermediate base classes (e.g. WDTaggerBasePlugin) do not define an identity.
        # Subclasses also should not automatically re-register under their parent's identity.
        # Only register classes that explicitly declare their own ModelIdentity.
        identity = cls.__dict__.get("identity", None)
        if identity is None:
            return

        # Validate required class-level metadata on concrete models
        if not getattr(cls, "family_name", None):
            raise ValueError(f"Concrete plugin '{cls.__name__}' must inherit or define a 'family_name' string.")
        if not getattr(cls, "profile", None):
            raise ValueError(f"Concrete plugin '{cls.__name__}' must declare a 'profile: ModelProfile'.")
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

        try:
            import vibe.registry as reg

            reg.model_registry.register(cls)
        except (ImportError, AttributeError, ValueError) as exc:
            import warnings

            warnings.warn(str(exc), stacklevel=2)

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        """Initialize plugin-local static metadata from resolved artifacts."""

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        """Build a fully initialized runtime for this model and execution plan."""
        raise NotImplementedError(
            f"Plugin '{self.identity.model_id}' has not implemented the build_runtime() contract."
        )

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

        # TODO: Add support for dataclass splitting (e.g., Hugging Face ModelOutput).
        # Future models like SigLIP 2 or DINOv2 return dataclasses instead of raw arrays.
        # Implementation needs `dataclasses.is_dataclass(batched_output)` to extract fields,
        # recursively call split_batch on each field, and reconstruct using `type(batched_output)(**...)`.

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
        """Preprocess an image input using optional per-inference settings."""

    @abstractmethod
    def postprocess(self, raw_output: Any) -> ModelResult:
        """Transform raw framework outputs into a standard ModelResult."""

    @classmethod
    def describe(cls) -> ModelDescriptor:
        """Assembles a structured descriptor of the model plugin's metadata."""
        resolved_variants = tuple(v.resolve(cls.default_repo_id) for v in cls.variants)

        return ModelDescriptor(
            schema_version="1.0.0",
            identity=cls.identity,
            family_name=cls.family_name,
            model_types=cls.profile.model_types,
            capabilities=tuple(p.__name__ for p in getattr(cls, "implements", ())),
            input=cls.profile.input_spec,
            output=cls.profile.output_spec,
            settings=cls.settings,
            consumer_settings=cls.profile.consumer_settings,
            default_repo_id=None if cls.custom_only and not cls.default_repo_id else cls.default_repo_id,
            variants=resolved_variants,
        )
