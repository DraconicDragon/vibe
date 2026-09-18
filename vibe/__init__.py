"""
vibe — vision transformer inference backend.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    ExecutionPlan,
    ExecutionPreference,
    FileRole,
    HardwareIntent,
    ModelPlugin,
    ModelVariant,
)
from vibe.contracts import (
    CalibrationProvider,
    LabelCatalogProvider,
    ScorerView,
    TaggerView,
    ThresholdProvider,
)
from vibe.exceptions import (
    InferenceCancelled,
    PluginContractError,
    RegistryError,
    SessionCapabilityError,
    SessionError,
)
from vibe.hardware import list_available_devices
from vibe.hf_downloader import (
    get_auto_download_default,
    set_auto_download_default,
)
from vibe.image_loading import ImageChunk, iter_load_images, iter_load_normalized
from vibe.loader import (
    ArtifactAvailability,
    ModelAvailability,
    VariantAvailability,
    inspect_variant_artifacts,
)
from vibe.memory_stats import (
    InferenceMemoryRecord,
    MemorySnapshot,
    MemoryTrackerStats,
)
from vibe.metadata import (
    CalibrationTable,
    ConsumerSettingSpec,
    InputSpec,
    LabelCatalog,
    LabelInfo,
    LabelSource,
    Modality,
    ModelDescriptor,
    ModelIdentity,
    ModelProfile,
    OutputKind,
    OutputSpec,
    ScoreSemantics,
    StandardConsumerSettingId,
    TagFilterRecommendation,
    ThresholdTable,
)
from vibe.model_profiles import (
    build_multi_scorer_profile,
    build_scorer_profile,
    build_tagger_profile,
)
from vibe.precision import PrecisionPolicy, PrecisionRequest, ResolvedPrecisionPlan, parse_precision
from vibe.registry import model_registry
from vibe.results import (
    InferenceResult,
    InferenceResultItem,
    ModelResult,
    MultiScoreResult,
    ScoreResult,
    TagEntry,
    TagResult,
    is_multi_score_result,
    is_score_result,
    is_tag_result,
)
from vibe.session import ModelSession
from vibe.session_factory import build_session
from vibe.settings import (
    InferenceRequest,
    OptionScope,
    SettingGroupSpec,
    compile_settings,
    serialize_value,
)
from vibe.tag_categories import (
    DANBOORU_CATEGORY_LABELS,
    E621_CATEGORY_LABELS,
    DanbooruTagCategory,
    E621TagCategory,
    TagCategory,
)

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
    auto_download: bool | None,
    file_name_map: Mapping[str, str] | None,
    memory_tracking: bool,
    is_custom: bool,
) -> ModelSession:
    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)
    precision_request = parse_precision(precision)

    if source is not None and not source.strip():
        raise SessionError("Source cannot be empty.")

    log_source = source or f"(default: {plugin_cls.default_repo_id})"
    if is_custom:
        logger.info("Loading custom plugin '%s' from '%s'", plugin_cls.__name__, log_source)
        logger.debug(
            "Load custom options source=%s backend=%s variant=%s device=%s auto_download=%s memory_tracking=%s",
            log_source,
            backend.value if isinstance(backend, Backend) else backend or "auto",
            variant or "(default)",
            device,
            effective_auto_download,
            memory_tracking,
        )
    else:
        logger.info("Loading model '%s' from '%s'", plugin_cls.identity.model_id, log_source)
        logger.debug(
            "Load options plugin=%s source=%s backend=%s variant=%s device=%s auto_download=%s memory_tracking=%s",
            plugin_cls.__name__,
            log_source,
            backend.value if isinstance(backend, Backend) else backend or "auto",
            variant or "(default)",
            device,
            effective_auto_download,
            memory_tracking,
        )

    logger.debug("Load precision request=%s", precision_request)

    return build_session(
        plugin_cls=plugin_cls,
        source=source,
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
    auto_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    memory_tracking: bool = False,
) -> ModelSession:
    """Load a model and return a ready-to-use ModelSession."""
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
    auto_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    memory_tracking: bool = False,
) -> ModelSession:
    """Load a model by specifying the plugin class explicitly."""
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
    """Check if a model's required files are already present on disk or in HF cache without downloading."""
    model_registry.ensure_discovered()

    plugin_cls = model_registry.get(model)

    variants_to_check = plugin_cls.variants
    if variant is not None:
        matched = [v for v in plugin_cls.variants if v.variant_id == variant]
        if not matched:
            available = [v.variant_id for v in plugin_cls.variants if v.variant_id]
            raise RegistryError(f"Model '{model}' has no variant '{variant}'. Available variants: {available}")
        variants_to_check = tuple(matched)

    variant_statuses: list[VariantAvailability] = []

    for v in variants_to_check:
        if source is not None and source.strip():
            effective_v_source = source.strip()
        else:
            default_repo = v.repo_id or plugin_cls.default_repo_id
            effective_v_source = f"hf:{default_repo}"

        artifact_statuses = inspect_variant_artifacts(
            source=effective_v_source,
            variant=v,
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

    return ModelAvailability(
        model_id=plugin_cls.identity.model_id,
        is_available=any(v.is_available for v in variant_statuses),
        variants=variant_statuses,
    )


# endregion API


__all__ = [
    "DANBOORU_CATEGORY_LABELS",
    "E621_CATEGORY_LABELS",
    "ArtifactAvailability",
    "ArtifactMap",
    "ArtifactSpec",
    "Backend",
    "CalibrationProvider",
    "CalibrationTable",
    "ConsumerSettingSpec",
    "DanbooruTagCategory",
    "E621TagCategory",
    "ExecutionPlan",
    "ExecutionPreference",
    "FileRole",
    "HardwareIntent",
    "ImageChunk",
    "InferenceCancelled",
    "InferenceMemoryRecord",
    "InferenceRequest",
    "InferenceResult",
    "InferenceResultItem",
    "InputSpec",
    "LabelCatalog",
    "LabelCatalogProvider",
    "LabelInfo",
    "LabelSource",
    "MemorySnapshot",
    "MemoryTrackerStats",
    "Modality",
    "ModelAvailability",
    "ModelDescriptor",
    "ModelIdentity",
    "ModelPlugin",
    "ModelProfile",
    "ModelResult",
    "ModelSession",
    "ModelVariant",
    "MultiScoreResult",
    "OptionScope",
    "OutputKind",
    "OutputSpec",
    "PluginContractError",
    "PrecisionPolicy",
    "PrecisionRequest",
    "RegistryError",
    "ResolvedPrecisionPlan",
    "ScoreResult",
    "ScoreSemantics",
    "ScorerView",
    "SessionCapabilityError",
    "SessionError",
    "SettingGroupSpec",
    "StandardConsumerSettingId",
    "TagCategory",
    "TagEntry",
    "TagFilterRecommendation",
    "TagResult",
    "TaggerView",
    "ThresholdProvider",
    "ThresholdTable",
    "VariantAvailability",
    "__author__",
    "__license__",
    "__version__",
    "build_multi_scorer_profile",
    "build_scorer_profile",
    "build_tagger_profile",
    "check_availability",
    "compile_settings",
    "describe",
    "describe_all",
    "get_auto_download_default",
    "is_multi_score_result",
    "is_score_result",
    "is_tag_result",
    "iter_load_images",
    "iter_load_normalized",
    "list_available_devices",
    "list_models",
    "list_plugin_classes",
    "load",
    "load_custom",
    "model_registry",
    "parse_precision",
    "serialize_value",
    "set_auto_download_default",
]
