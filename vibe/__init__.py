"""
vibe — vision transformer inference backend.

# todo: change

Quick start
-----------
    import vibe

    # Load a registered model (downloads from HF automatically)
    session = vibe.load("wd-eva02-v3")
    result = session.infer(image).first()
    print([entry.tag for entry in result.tags["general"][:5]])

    # Batch processing: infer returns InferenceResult with multiple items
    results = session.infer([image1, image2, image3])
    for item in results:
        print(f"Input {item.index}: {[entry.tag for entry in item.result.tags['general'][:3]]}")

    # Use a local folder instead of HF
    session = vibe.load("wd-eva02-v3", source="local:/path/to/folder")

    # Custom: arbitrary source with a chosen plugin
    session = vibe.load_custom(
        source="hf:SmilingWolf/wd-eva02-large-tagger-v3-updated", # doesn't exist, just example
        plugin="WDEva02Plugin",
    )

    # Inspect available models
    vibe.list_models()
    vibe.describe("wd-eva02-v3")
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Mapping

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin, ModelPluginInfo
from vibe.devices import list_available_devices
from vibe.hf_downloader import (
    get_auto_download_default,
    set_auto_download_default,
)
from vibe.image_loading import ImageChunk, iter_load_images
from vibe.memory_stats import (
    InferenceMemoryRecord,
    MemorySnapshot,
    MemoryTrackerStats,
)
from vibe.precision import normalize_precision_string
from vibe.registry import ModelRegistry, RegistryError, _make_auto_register_hook
from vibe.result_processors import (
    CharacterIPMapping,
    CleanTags,
    MultiScoreToScore,
    NormalizedScore,
    ProcessorInfo,
    ResultProcessor,
    TagLevelThresholds,
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
from vibe.session import ModelSession, SessionError
from vibe.session_factory import build_session

logger = logging.getLogger(__name__)

try:
    __version__ = _package_version("vibe")
except PackageNotFoundError:
    logger.warning("Package version not found. Are you running from source without installing? Defaulting to None.")
    __version__ = None

__author__ = "Drac"
__license__ = "MIT"

# region Global Registry

model_registry: ModelRegistry = ModelRegistry()

# Wire up auto-registration: whenever a ModelPlugin subclass is defined
# (i.e. when a plugin module is imported), it registers itself.
_auto_register = _make_auto_register_hook(model_registry)
_original_init_subclass = ModelPlugin.__init_subclass__.__func__


def _patched_init_subclass(cls, **kwargs):
    _original_init_subclass(cls, **kwargs)
    _auto_register(cls)


ModelPlugin.__init_subclass__ = classmethod(_patched_init_subclass)  # type: ignore[assignment]

# Discover and register all built-in plugins
model_registry.discover_all()

# endregion Global Registry


# region API


def load(
    model: str,
    *,
    source: str | None = None,
    source_map: Mapping[str, str] | None = None,
    backend: str | Backend | None = None,
    device: str = "auto",
    precision: str = "auto",
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    onnx_providers: list[str] | None = None,
    auto_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    memory_tracking: bool = True,
) -> ModelSession:
    """
    Load a model and return a ready-to-use ModelSession.

    Args:
        model:          Model ID or alias (e.g. "wd-eva02-v3").
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
    plugin_cls = model_registry.get(model)
    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)
    normalized_precision = normalize_precision_string(precision)

    resolved_source = _resolve_source(
        source,
        plugin_cls,
    )

    logger.info("Loading model '%s' from '%s'", model, resolved_source)
    logger.debug(
        "Load options plugin=%s source=%s backend=%s device=%s auto_download=%s memory_tracking=%s",
        plugin_cls.__name__,
        resolved_source,
        backend.value if isinstance(backend, Backend) else backend or "auto",
        device,
        effective_auto_download,
        memory_tracking,
    )
    logger.debug("Load precision request=%s normalized=%s", precision, normalized_precision)

    return build_session(
        plugin_cls=plugin_cls,
        source=resolved_source,
        source_map=source_map,
        backend=backend,
        device=device,
        precision=normalized_precision,
        onnx_providers=onnx_providers,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        auto_download=effective_auto_download,
        file_name_map=file_name_map,
        memory_tracking=memory_tracking,
    )


def load_custom(
    *,
    source: str | None = None,
    source_map: Mapping[str, str] | None = None,
    plugin: str,
    backend: str | Backend | None = None,
    device: str = "auto",
    precision: str = "auto",
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    onnx_providers: list[str] | None = None,
    auto_download: bool | None = None,
    file_name_map: Mapping[str, str] | None = None,
    memory_tracking: bool = True,
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
    plugin_cls = model_registry.get_by_class_name(plugin)
    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)
    normalized_precision = normalize_precision_string(precision)
    resolved_source = _resolve_source(
        source,
        plugin_cls,
    )

    logger.info("Loading custom plugin '%s' from '%s'", plugin_cls.__name__, resolved_source)
    logger.debug(
        "Load custom options source=%s backend=%s device=%s auto_download=%s memory_tracking=%s",
        resolved_source,
        backend.value if isinstance(backend, Backend) else backend or "auto",
        device,
        effective_auto_download,
        memory_tracking,
    )
    logger.debug("Load custom precision request=%s normalized=%s", precision, normalized_precision)

    return build_session(
        plugin_cls=plugin_cls,
        source=resolved_source,
        source_map=source_map,
        backend=backend,
        device=device,
        precision=normalized_precision,
        onnx_providers=onnx_providers,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        auto_download=effective_auto_download,
        file_name_map=file_name_map,
        memory_tracking=memory_tracking,
    )


def list_models() -> list[str]:
    """Return a sorted list of all registered model IDs."""
    return model_registry.list_model_ids()


def list_plugin_classes() -> list[str]:
    """Return the class names of all registered plugins (for load_custom)."""
    return model_registry.list_plugin_classes()


def describe(model: str) -> ModelPluginInfo:
    """Return typed model metadata for a model ID or alias."""
    return model_registry.get(model).describe()


def describe_all() -> list[ModelPluginInfo]:
    """Return typed metadata objects for all registered models."""
    return model_registry.list_all()


# endregion API


# region Helpers


def _resolve_source(
    source: str | None,
    plugin_cls: type[ModelPlugin],
) -> str:
    if source is None:
        if plugin_cls.default_hf_repo is None:
            raise SessionError(
                f"Model '{plugin_cls.model_id}' has no default HF repo. "
                f"Provide a source explicitly: "
                f"vibe.load('{plugin_cls.model_id}', source=...)"
            )
        return f"hf:{plugin_cls.default_hf_repo}"

    normalized = source.strip()
    if not normalized:
        raise SessionError("Source cannot be empty.")
    return normalized


# endregion Helpers


# region Utils
# todo: move out in future


def list_processors() -> list[ProcessorInfo]:
    "Return metadata in form of ProcessorInfo for all available result processor classes in the library."
    import vibe.result_processors as rp

    return [
        cls.describe()
        for cls in vars(rp).values()
        if isinstance(cls, type)
        and issubclass(cls, rp.ResultProcessor)
        and cls is not rp.ResultProcessor
        and hasattr(cls, "_processor_info")  # only classes that completed __init_subclass__
    ]


def get_processor(name: str) -> type[ResultProcessor]:
    """Return a result processor class by its class name in form of a string."""
    import vibe.result_processors as rp

    cls = getattr(rp, name, None)
    if cls is None or not (isinstance(cls, type) and issubclass(cls, rp.ResultProcessor)):
        available = [
            c.__name__
            for c in vars(rp).values()
            if isinstance(c, type) and issubclass(c, rp.ResultProcessor) and c is not rp.ResultProcessor
        ]
        raise RegistryError(f"No processor named '{name}'. Available: {available}")
    return cls


# endregion Utils

# region Public re-Exports
# for users doing from vibe import ...

__all__ = [
    # Core objects
    "ModelSession",
    "ModelPlugin",
    "FileSpec",
    "FileRole",
    "Backend",
    "ModelPluginInfo",
    "MemorySnapshot",
    "InferenceMemoryRecord",
    "MemoryTrackerStats",
    # Results
    "TagResult",
    "TagEntry",
    "ScoreResult",
    "MultiScoreResult",
    "OutputType",
    "ModelResult",
    "InferenceResultItem",
    "InferenceResult",
    "is_tag_result",
    "is_score_result",
    "is_multi_score_result",
    "ImageChunk",
    "iter_load_images",
    # Processors
    "ResultProcessor",
    "CleanTags",
    "CharacterIPMapping",
    "TagLevelThresholds",
    "MultiScoreToScore",
    "NormalizedScore",
    # Registry
    "model_registry",
    "RegistryError",
    "SessionError",
    # API
    "__version__",
    "__author__",
    "__license__",
    "load",
    "load_custom",
    "list_models",
    "list_available_devices",
    "list_plugin_classes",
    "describe",
    "describe_all",
    "set_auto_download_default",
    "get_auto_download_default",
    "normalize_precision_string",
]


# endregion Public re-Exports
