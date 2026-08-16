"""
vibe — vision transformer inference backend.

# todo: change

Quick start
-----------
    import vibe

    # Load a registered model (downloads from HF automatically)
    session = vibe.load("wd-eva02-large-v3")
    result = session.infer(image).first()
    print([entry.tag for entry in result.tags["general"][:5]])

    # Batch processing: infer returns InferenceResult with multiple items
    results = session.infer([image1, image2, image3])
    for item in results:
        print(f"Input {item.index}: {[entry.tag for entry in item.result.tags['general'][:3]]}")

    # Use a local folder instead of HF
    session = vibe.load("wd-eva02-large-v3", source="local:/path/to/folder")

    # Custom: arbitrary source with a chosen plugin
    session = vibe.load_custom(
        source="hf:SmilingWolf/wd-eva02-large-tagger-v3-updated", # doesn't exist, just example
        plugin="WDEva02Plugin",
    )

    # Inspect available models
    vibe.list_models()
    vibe.describe("wd-eva02-large-v3")
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Any

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    ExecutionPlan,
    ExecutionPreference,
    FileRole,
    HardwareIntent,
    ModelCapabilities,
    ModelDescriptor,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
    PluginOptionSpec,
)
from vibe.exceptions import InferenceCancelled, RegistryError, SessionError, TransformError
from vibe.hardware import list_available_devices
from vibe.hf_downloader import (
    get_auto_download_default,
    set_auto_download_default,
)
from vibe.image_loading import ImageChunk, iter_load_images
from vibe.loader import ModelAvailability, VariantAvailability, inspect_variant_artifacts
from vibe.memory_stats import (
    InferenceMemoryRecord,
    MemorySnapshot,
    MemoryTrackerStats,
)
from vibe.precision import PrecisionPolicy, PrecisionRequest, ResolvedPrecisionPlan, parse_precision
from vibe.registry import model_registry, transform_registry
from vibe.result_transforms import (
    CharacterIPMapping,
    CleanTags,
    ResultTransform,
    ScoreThresholds,
    TagLevelThresholds,
    TransformInfo,
    TransformOptionSpec,
    transform_meta,
)
from vibe.results import (
    InferenceResult,
    InferenceResultItem,
    ModelResult,
    MultiScoreResult,
    OutputType,
    ScoreResult,
    TagEntry,
    TagResult,
    is_multi_score_result,
    is_score_result,
    is_tag_result,
)
from vibe.session import ModelSession
from vibe.session_factory import build_session

logger = logging.getLogger(__name__)

try:
    __version__ = _package_version("vibe")
except PackageNotFoundError:
    logger.warning("Package version not found. Are you running from source without installing? Defaulting to None.")
    __version__ = None

__author__ = "Drac"
__license__ = "MIT"


# region API


def _load_internal(
    plugin_cls: type[ModelPlugin],
    source: str | None,
    source_map: Mapping[str, str] | None,
    backend: str | Backend | None,
    variant: str | None,
    device: str,
    precision: str | PrecisionRequest,
    hf_revision: str | None,
    hf_cache_dir: str | None,
    onnx_providers: list[str] | None,
    hf_token: str | None,
    options: Mapping[str, Any] | None,
    auto_download: bool | None,
    file_name_map: Mapping[str, str] | None,
    memory_tracking: bool,
    is_custom: bool,
) -> ModelSession:
    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)
    precision_request = parse_precision(precision)
    resolved_source = _resolve_source(source, plugin_cls)

    if is_custom:
        logger.info("Loading custom plugin '%s' from '%s'", plugin_cls.__name__, resolved_source)
        logger.debug(
            "Load custom options source=%s backend=%s variant=%s device=%s auto_download=%s memory_tracking=%s",
            resolved_source,
            backend.value if isinstance(backend, Backend) else backend or "auto",
            variant or "(default)",
            device,
            effective_auto_download,
            memory_tracking,
        )
    else:
        logger.info("Loading model '%s' from '%s'", plugin_cls.identity.model_id, resolved_source)
        logger.debug(
            "Load options plugin=%s source=%s backend=%s variant=%s device=%s auto_download=%s memory_tracking=%s",
            plugin_cls.__name__,
            resolved_source,
            backend.value if isinstance(backend, Backend) else backend or "auto",
            variant or "(default)",
            device,
            effective_auto_download,
            memory_tracking,
        )

    logger.debug("Load precision request=%s", precision_request)

    return build_session(
        plugin_cls=plugin_cls,
        source=resolved_source,
        source_map=source_map,
        backend=backend,
        variant=variant,
        device=device,
        precision=precision_request,
        onnx_providers=onnx_providers,
        hf_token=hf_token,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        auto_download=effective_auto_download,
        file_name_map=file_name_map,
        options=options,
        memory_tracking=memory_tracking,
    )


def load(
    model: str,
    *,
    source: str | None = None,
    source_map: Mapping[str, str] | None = None,
    backend: str | Backend | None = None,
    variant: str | None = None,
    device: str = "auto",
    precision: str | PrecisionRequest = "auto",
    onnx_providers: list[str] | None = None,
    hf_token: str | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    options: Mapping[str, Any] | None = None,
    auto_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    memory_tracking: bool = False,
) -> ModelSession:
    """
    Load a model and return a ready-to-use ModelSession.

    Args:
        model:          Model ID (e.g. "wd-eva02-large-v3").
                        Run vibe.list_models() to see all options.
                        # todo: check if doc gen will set optional by default through type hints or if i should put it in docstring explicitly, or maybe just in general
        source:         Optional. Where to load files from. String options:
                          - None (default): use the plugin's default HF repo.
                          - Prefix forms (strict mode):
                              "local:/path/to/folder"
                              "hf:owner/repo-name"
                          - Unprefixed (auto mode):
                              first tries local folder when it exists,
                              then tries HF repo/cache/download.
        source_map:     Optional mapping of repo_id -> source string used
                        to override the source for specific FileSpec entries.
                        Any FileSpec whose repo_id matches a key here uses
                        the mapped source. All others use `source`.
        backend:        "pytorch" or "onnx". None = auto-detect.
        device:         Logical device selector. For ONNX it guides provider
            auto-selection (e.g. "cpu", "gpu", "gpu:1", "cuda:0").
            Default "auto". 'cuda' and 'gpu' are interchangeable.
        precision:      Runtime precision selector. Supported values:
                        "auto" (default): Backend/model will dictate weight & compute precision.
                        - PyTorch: "fp32", "fp16", "bf16"
                        - ONNX: "ov", "int8_ov"
                        Note: ONNX precision is usually based on model weight precision outside of ov/openvino.
        hf_revision:    HF repo revision (branch/tag/commit). Only used when
                        source is None (default HF repo) or source is "hf:...".
        hf_cache_dir:   Override HF cache directory.
        onnx_providers: Override ONNX execution providers.
        auto_download: Per-session download policy. None uses global default.
                   False uses only local/cached files; no downloads.
        file_name_map:
            Optional filename remapping for file resolution across
            local folders, HF repos, and HF cache paths.
            Keys are plugin-declared filenames (e.g. "model.onnx"),
            values are source filenames to use instead
            (e.g. "wdeva02.onnx").
        memory_tracking: Enable per-call memory telemetry inside this session.

    Returns:
        ModelSession ready for .infer(image).

    Raises:
        RegistryError:  If the model name is not recognised.
        SessionError:   If loading fails (missing files, bad backend, etc.).
    """
    model_registry.ensure_discovered()

    plugin_cls = model_registry.get(model)
    return _load_internal(
        plugin_cls=plugin_cls,
        source=source,
        source_map=source_map,
        backend=backend,
        variant=variant,
        device=device,
        precision=precision,
        onnx_providers=onnx_providers,
        hf_token=hf_token,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        options=options,
        auto_download=auto_download,
        file_name_map=file_name_map,
        memory_tracking=memory_tracking,
        is_custom=False,
    )


def load_custom(
    *,
    source: str | None = None,
    source_map: Mapping[str, str] | None = None,
    plugin: str,
    backend: str | Backend | None = None,
    variant: str | None = None,
    device: str = "auto",
    precision: str | PrecisionRequest = "auto",
    onnx_providers: list[str] | None = None,
    hf_token: str | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    options: Mapping[str, Any] | None = None,
    auto_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    memory_tracking: bool = False,
) -> ModelSession:
    """
    Load a model by specifying the plugin class explicitly.

    This is the power-user path: point at any source and tell vibe
    which plugin's inference code to use. Useful when:
      - A new model was released that isn't registered yet.
      - You want to use a fine-tune with the same architecture as a known plugin.

    Args:
        source:         Where to load files from. Same rules as load():
                          - None (default): use the plugin's default HF repo.
                          - Prefix forms (strict mode):
                              "local:/path/to/folder"
                              "hf:owner/repo-name"
                          - Unprefixed (auto mode):
                              first tries local folder when it exists,
                              then tries HF repo/cache/download.
        source_map:     Optional mapping of repo_id -> source string used
                        to override the source for specific FileSpec entries.
                        Any FileSpec whose repo_id matches a key here uses
                        the mapped source. All others use `source`.
        plugin:         Plugin class name (e.g. "WDEva02Plugin").
                        Run vibe.list_plugin_classes() to see all options.
        backend, device, precision, hf_revision, hf_cache_dir, onnx_providers:
                        Same as load().
        file_name_map:
                Same as load(). Maps plugin file names to source file
                names across local/HF/HF-cache resolution.

    Example:
        session = vibe.load_custom(
            source="hf:SmilingWolf/wd-eva02-large-tagger-v3-updated",
            plugin="WDEva02Plugin",
        )
    """
    model_registry.ensure_discovered()

    plugin_cls = model_registry.get_by_class_name(plugin)
    return _load_internal(
        plugin_cls=plugin_cls,
        source=source,
        source_map=source_map,
        backend=backend,
        variant=variant,
        device=device,
        precision=precision,
        onnx_providers=onnx_providers,
        hf_token=hf_token,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        options=options,
        auto_download=auto_download,
        file_name_map=file_name_map,
        memory_tracking=memory_tracking,
        is_custom=True,
    )


def list_models() -> list[str]:
    """Return a sorted list of all registered model IDs."""
    model_registry.ensure_discovered()

    return model_registry.list_model_ids()


def list_plugin_classes() -> list[str]:
    """Return the class names of all registered plugins (for load_custom)."""
    model_registry.ensure_discovered()

    return model_registry.list_plugin_classes()


def describe(model: str) -> ModelDescriptor:
    """Return typed model metadata for a model ID."""
    model_registry.ensure_discovered()

    return model_registry.get(model).describe()


def describe_all() -> list[ModelDescriptor]:
    """Return typed metadata objects for all registered models."""
    model_registry.ensure_discovered()

    return model_registry.list_all()


def check_availability(
    model: str,
    *,
    source: str | None = None,
    variant: str | None = None,
    source_map: Mapping[str, str] | None = None,
    file_name_map: Mapping[str, str] | None = None,
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    hf_token: str | None = None,
) -> ModelAvailability:
    """
    Check if a model's required files are already present on disk or in HF cache without downloading.

    Args:
        model:         Model ID (e.g. "wd-eva02-large-v3").
        source:        Source string (e.g. "local:/path", "hf:owner/repo", or None for default repo).
        variant:       Optional variant ID filter.
        source_map:    Optional per-artifact source overrides.
        file_name_map: Optional filename remappings.
        hf_revision:   HF repo revision.
        hf_cache_dir:  Override HF cache directory.
        hf_token:      HF access token.

    Returns:
        ModelAvailability summary detailing presence/absence of all artifacts per variant.
    """
    model_registry.ensure_discovered()

    plugin_cls = model_registry.get(model)
    resolved_source = _resolve_source(source, plugin_cls)

    variants_to_check = plugin_cls.variants
    if variant is not None:
        matched = [v for v in plugin_cls.variants if v.variant_id == variant]
        if not matched:
            available = [v.variant_id for v in plugin_cls.variants if v.variant_id]
            raise RegistryError(f"Model '{model}' has no variant '{variant}'. Available variants: {available}")
        variants_to_check = tuple(matched)

    variant_statuses: list[VariantAvailability] = []

    for v in variants_to_check:
        resolved_v = v.resolve(plugin_cls.default_repo_id)
        artifact_statuses = inspect_variant_artifacts(
            source=resolved_source,
            variant=resolved_v,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            file_name_map=file_name_map,
            source_map=source_map,
            token=hf_token,
        )
        variant_ok = all(art.is_available for art in artifact_statuses if art.required)
        variant_statuses.append(
            VariantAvailability(
                variant_id=v.variant_id,
                backend=v.backend,
                is_available=variant_ok,
                artifacts=artifact_statuses,
            )
        )

    model_ok = any(v.is_available for v in variant_statuses)

    return ModelAvailability(
        model_id=plugin_cls.identity.model_id,
        is_available=model_ok,
        variants=variant_statuses,
    )


# endregion API


# region Helpers


def _resolve_source(
    source: str | None,
    plugin_cls: type[ModelPlugin],
) -> str:
    if source is None:
        return f"hf:{plugin_cls.default_repo_id}"

    normalized = source.strip()
    if not normalized:
        raise SessionError("Source cannot be empty.")
    return normalized


# endregion Helpers


# region Utils
# todo: move out in future


def list_transforms() -> list[TransformInfo]:
    """Return metadata in form of TransformInfo for all available transforms in the library."""
    # Triggers evaluation of any pending class decorators
    model_registry.ensure_discovered()
    return transform_registry.list_all()


def get_transform(transform_id: str) -> type[ResultTransform]:
    """Return a result transform class by its string ID."""
    model_registry.ensure_discovered()
    return transform_registry.get(transform_id)


# endregion Utils

# region Public re-Exports
# for users doing from vibe import ...

__all__ = [
    "ArtifactAvailability",
    "ArtifactMap",
    "ArtifactSpec",
    "Backend",
    "CharacterIPMapping",
    "CleanTags",
    "ExecutionPlan",
    "ExecutionPreference",
    "FileRole",
    "HardwareIntent",
    "ImageChunk",
    "InferenceCancelled",
    "InferenceMemoryRecord",
    "InferenceResult",
    "InferenceResultItem",
    "MemorySnapshot",
    "MemoryTrackerStats",
    "ModelAvailability",
    "ModelCapabilities",
    "ModelDescriptor",
    "ModelIdentity",
    "ModelPlugin",
    "ModelResult",
    "ModelSession",
    "ModelVariant",
    "MultiScoreResult",
    "OutputType",
    "PluginOptionSpec",
    "PrecisionPolicy",
    "PrecisionRequest",
    "RegistryError",
    "ResolvedPrecisionPlan",
    "ResultTransform",
    "ScoreResult",
    "ScoreThresholds",
    "SessionError",
    "TagEntry",
    "TagLevelThresholds",
    "TagResult",
    "TransformError",
    "TransformInfo",
    "TransformOptionSpec",
    "VariantAvailability",
    "__author__",
    "__license__",
    "__version__",
    "check_availability",
    "describe",
    "describe_all",
    "get_auto_download_default",
    "get_transform",
    "is_multi_score_result",
    "is_score_result",
    "is_tag_result",
    "iter_load_images",
    "list_available_devices",
    "list_models",
    "list_plugin_classes",
    "list_transforms",
    "load",
    "load_custom",
    "model_registry",
    "parse_precision",
    "set_auto_download_default",
    "transform_meta",
    "transform_registry",
]

# endregion Public re-Exports
