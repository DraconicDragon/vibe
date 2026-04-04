"""
vibe — vision transformer inference backend.

Quick start
-----------
    import vibe

    # Load a registered model (downloads from HF automatically)
    session = vibe.load("wd-eva02-large")
    result = session.infer(image).first()
    print(result.general[:5])

    # Batch processing: infer returns InferenceResult with multiple items
    results = session.infer([image1, image2, image3])
    for item in results:
        print(f"Input {item.index}: {item.result.general[:3]}")

    # Use a local folder instead of HF
    session = vibe.load("wd-eva02-large", source="local:/path/to/folder")

    # Custom: arbitrary source with a chosen plugin
    session = vibe.load_custom(
        source="hf:SmilingWolf/wd-eva02-large-tagger-v3-updated",
        plugin="WDEva02Plugin",
    )

    # Inspect available models
    vibe.list_models()
    vibe.describe("wd-eva02-large")
"""

from __future__ import annotations

from typing import Any

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.devices import list_available_devices
from vibe.hf_downloader import (
    get_auto_download_default,
    set_auto_download_default,
)
from vibe.memory_stats import (
    InferenceMemoryRecord,
    MemorySnapshot,
    MemoryTrackerStats,
)
from vibe.registry import ModelRegistry, RegistryError, _make_auto_register_hook
from vibe.result_processors import CharacterIPMapping, CleanTags, ResultProcessor
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

# region Global Registry

registry: ModelRegistry = ModelRegistry()

# Wire up auto-registration: whenever a ModelPlugin subclass is defined
# (i.e. when a plugin module is imported), it registers itself.
_auto_register = _make_auto_register_hook(registry)
_original_init_subclass = ModelPlugin.__init_subclass__.__func__


def _patched_init_subclass(cls, **kwargs):
    _original_init_subclass(cls, **kwargs)
    _auto_register(cls)


ModelPlugin.__init_subclass__ = classmethod(_patched_init_subclass)  # type: ignore[assignment]

# Discover and register all built-in plugins
registry.discover_all()

# endregion Global Registry


# region API


def load(
    model: str,
    *,
    source: str | None = None,
    backend: str | Backend | None = None,
    device: str = "cpu",
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    onnx_providers: list[str] | None = None,
    auto_download: bool | None = None,
    memory_tracking: bool = True,
) -> ModelSession:
    """
    Load a model and return a ready-to-use ModelSession.

    Args:
        model:          Model ID or alias (e.g. "wd-eva02-large", "eva02").
                        Run vibe.list_models() to see all options.
                        # todo: check if doc gen will set optional by default through type hints or if i should put it in docstring explicitly, or maybe just in general
        source:         Optional. Where to load files from. String options:
                          - None (default): use the plugin's default HF repo.
                          - Prefix forms (strict mode):
                              "local:/path/to/folder"
                              "hf:owner/repo-name"
                              "hf_cache:/path/to/snapshot"
                          - Unprefixed (auto mode):
                              first tries local folder when it exists,
                              then tries HF repo/cache/download.
        backend:        "pytorch" or "onnx". None = auto-detect.
        device:         Logical device selector. For ONNX it guides provider
            auto-selection (e.g. "cpu", "gpu", "gpu:1", "cuda:0").
                Default "cpu". 'cuda' and 'gpu' are interchangeable.
        hf_revision:    HF repo revision (branch/tag/commit). Only used when
                        source is None (default HF repo) or source is "hf:...".
        hf_cache_dir:   Override HF cache directory.
        onnx_providers: Override ONNX execution providers.
        auto_download: Per-session download policy. None uses global default.
                   False uses only local/cached files; no downloads.
        memory_tracking: Enable per-call memory telemetry inside this session.

    Returns:
        ModelSession ready for .infer(image).

    Raises:
        RegistryError:  If the model name is not recognised.
        SessionError:   If loading fails (missing files, bad backend, etc.).
    """
    plugin_cls = registry.get(model)
    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)

    resolved_source = _resolve_source(
        source,
        plugin_cls,
    )

    return build_session(
        plugin_cls=plugin_cls,
        source=resolved_source,
        backend=backend,
        device=device,
        onnx_providers=onnx_providers,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        auto_download=effective_auto_download,
        memory_tracking=memory_tracking,
    )


def load_custom(
    *,
    source: str | None = None,
    plugin: str,
    backend: str | Backend | None = None,
    device: str = "cpu",
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    onnx_providers: list[str] | None = None,
    auto_download: bool | None = None,
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
                              "hf_cache:/path/to/snapshot"
                          - Unprefixed (auto mode):
                              first tries local folder when it exists,
                              then tries HF repo/cache/download.
        plugin:         Plugin class name (e.g. "WDEva02Plugin").
                        Run vibe.list_plugin_classes() to see all options.
        backend, device, hf_revision, hf_cache_dir, onnx_providers:
                        Same as load().

    Example:
        session = vibe.load_custom(
            source="hf:SmilingWolf/wd-eva02-large-tagger-v3-updated",
            plugin="WDEva02Plugin",
        )
    """
    plugin_cls = registry.get_by_class_name(plugin)
    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)
    resolved_source = _resolve_source(
        source,
        plugin_cls,
    )

    return build_session(
        plugin_cls=plugin_cls,
        source=resolved_source,
        backend=backend,
        device=device,
        onnx_providers=onnx_providers,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        auto_download=effective_auto_download,
        memory_tracking=memory_tracking,
    )


def list_models() -> list[str]:
    """Return a sorted list of all registered model IDs."""
    return registry.list_model_ids()


def list_plugin_classes() -> list[str]:
    """Return the class names of all registered plugins (for load_custom)."""
    return registry.list_plugin_classes()


def describe(model: str) -> dict[str, Any]:
    """Return a full description dict for a model ID or alias."""
    return registry.get(model).to_dict()


def describe_all() -> list[dict[str, Any]]:
    """Return description dicts for all registered models."""
    return registry.list_all()


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


# region Public re-Exports
# for users doing from vibe import ...

__all__ = [
    # Core objects
    "ModelSession",
    "ModelPlugin",
    "FileSpec",
    "FileRole",
    "Backend",
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
    # Processors
    "ResultProcessor",
    "CleanTags",
    "CharacterIPMapping",
    # Registry
    "registry",
    "RegistryError",
    "SessionError",
    # API
    "load",
    "load_custom",
    "list_models",
    "list_available_devices",
    "list_plugin_classes",
    "describe",
    "describe_all",
    "set_auto_download_default",
    "get_auto_download_default",
]


# endregion Public re-Exports
