"""
autotagger — modular image tagging library.

Quick start
-----------
    import autotagger

    # Load a registered model (downloads from HF automatically)
    session = autotagger.load("wd-eva02-large")
    result = session.infer(image)
    print(result.tag_names())

    # Use a local folder instead of HF
    session = autotagger.load("wd-eva02-large", source="local:/path/to/folder")

    # Advanced: arbitrary HF repo with a chosen plugin
    session = autotagger.load_advanced(
        hf_repo="SmilingWolf/wd-eva02-large-tagger-v3-updated",
        plugin="WDEva02Plugin",
    )

    # Inspect available models
    autotagger.list_models()
    autotagger.describe("wd-eva02-large")
"""

from __future__ import annotations

from typing import Any

from autotagger.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from autotagger.devices import list_available_devices
from autotagger.hf_downloader import (
    get_auto_download_default,
    set_auto_download_default,
)
from autotagger.loader import ModelSource
from autotagger.memory_stats import (
    InferenceMemoryRecord,
    MemorySnapshot,
    MemoryTrackerStats,
)
from autotagger.params import EMPTY_SCHEMA, ParamDef, ParamSchema
from autotagger.registry import ModelRegistry, RegistryError, _make_auto_register_hook
from autotagger.results import (
    InferenceResult,
    MultiScoreResult,
    OutputType,
    ScoreResult,
    TagEntry,
    TagResult,
    is_multi_score_result,
    is_score_result,
    is_tag_result,
)
from autotagger.session import ModelSession, SessionError, build_session

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

# Import optional typed helpers after discovery so plugin auto-registration
# has already happened.
from autotagger.plugins.wd_tagger import WDTaggerParams, wd_tagger_params  # noqa: E402

# endregion Global Registry


# region API


def load(
    model: str,
    *,
    source: str | ModelSource | None = None,
    backend: str | Backend | None = None,
    device: str = "cpu",
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    onnx_providers: list[str] | None = None,
    auto_download: bool | None = None,
    character_mapping_path: str | None = None,
    memory_tracking: bool = True,
) -> ModelSession:
    """
    Load a model and return a ready-to-use ModelSession.

    Args:
        model:          Model ID or alias (e.g. "wd-eva02-large", "eva02").
                        Run autotagger.list_models() to see all options.
        source:         Where to load files from. Options:
                          - None (default): use the plugin's default HF repo.
                          - A ModelSource object.
                          - A string shorthand:
                              "local:/path/to/folder"
                              "hf:owner/repo-name"
                              "hf_cache:/path/to/snapshot"
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
        character_mapping_path: Optional explicit character-IP mapping file.
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
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        auto_download=effective_auto_download,
    )

    return build_session(
        plugin_cls=plugin_cls,
        source=resolved_source,
        backend=backend,
        device=device,
        onnx_providers=onnx_providers,
        auto_download=effective_auto_download,
        character_mapping_path=character_mapping_path,
        memory_tracking=memory_tracking,
    )


def load_advanced(
    *,
    hf_repo: str | None = None,
    local_folder: str | None = None,
    plugin: str,
    backend: str | Backend | None = None,
    device: str = "cpu",
    hf_revision: str | None = None,
    hf_cache_dir: str | None = None,
    onnx_providers: list[str] | None = None,
    auto_download: bool | None = None,
    character_mapping_path: str | None = None,
    memory_tracking: bool = True,
) -> ModelSession:
    """
    Load an arbitrary model by specifying the plugin class to use explicitly.

    This is the power-user path: point at any HF repo or local folder and
    tell autotagger which plugin's inference code to use. Useful when:
      - A new model was released that isn't registered yet.
      - You want to use a fine-tune with the same architecture as a known plugin.

    Exactly one of hf_repo or local_folder must be provided.

    Args:
        hf_repo:        HuggingFace repo ID.
        local_folder:   Path to a local folder.
        plugin:         Plugin class name (e.g. "WDEva02Plugin").
                        Run autotagger.list_plugin_classes() to see options.
        backend, device, hf_revision, hf_cache_dir, onnx_providers:
                        Same as load().

    Example:
        session = autotagger.load_advanced(
            hf_repo="SmilingWolf/wd-eva02-large-tagger-v3-updated",
            plugin="WDEva02Plugin",
        )
    """
    if hf_repo and local_folder:
        raise ValueError("Provide either hf_repo or local_folder, not both.")
    if not hf_repo and not local_folder:
        raise ValueError("Provide one of: hf_repo, local_folder.")

    plugin_cls = registry.get_by_class_name(plugin)
    effective_auto_download = get_auto_download_default() if auto_download is None else bool(auto_download)

    if hf_repo:
        source = ModelSource.hf(
            hf_repo,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            allow_download=effective_auto_download,
        )
    else:
        if local_folder is None:
            raise ValueError("Provide local_folder when hf_repo is not set.")
        source = ModelSource.local(local_folder)

    return build_session(
        plugin_cls=plugin_cls,
        source=source,
        backend=backend,
        device=device,
        onnx_providers=onnx_providers,
        auto_download=effective_auto_download,
        character_mapping_path=character_mapping_path,
        memory_tracking=memory_tracking,
    )


def list_models() -> list[str]:
    """Return a sorted list of all registered model IDs."""
    return registry.list_model_ids()


def list_plugin_classes() -> list[str]:
    """Return the class names of all registered plugins (for load_advanced)."""
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
    source: str | ModelSource | None,
    plugin_cls: type[ModelPlugin],
    hf_revision: str | None,
    hf_cache_dir: str | None,
    auto_download: bool,
) -> ModelSource:
    if source is None:
        # Use plugin's default HF repo
        if plugin_cls.default_hf_repo is None:
            raise SessionError(
                f"Model '{plugin_cls.model_id}' has no default HF repo. "
                f"Provide a source explicitly: "
                f"autotagger.load('{plugin_cls.model_id}', source=...)"
            )
        return ModelSource.hf(
            plugin_cls.default_hf_repo,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            allow_download=auto_download,
        )

    if isinstance(source, ModelSource):
        return source

    # String shorthand parsing
    if isinstance(source, str):
        if source.startswith("hf:"):
            repo = source[3:]
            return ModelSource.hf(
                repo,
                revision=hf_revision,
                cache_dir=hf_cache_dir,
                allow_download=auto_download,
            )
        elif source.startswith("local:"):
            folder = source[6:]
            return ModelSource.local(folder)
        elif source.startswith("hf_cache:"):
            path = source[9:]
            return ModelSource.hf_cache(path)
        else:
            # Assume it's a local path if it looks like one
            from pathlib import Path

            if Path(source).exists():
                return ModelSource.local(source)
            raise SessionError(
                f"Cannot parse source string '{source}'. Use 'hf:owner/repo', 'local:/path', or 'hf_cache:/path'."
            )

    raise SessionError(f"Invalid source type: {type(source)}")


# endregion Helpers


# region Public re-Exports
# for users doing from autotagger import ...

__all__ = [
    # Core objects
    "ModelSession",
    "ModelPlugin",
    "ModelSource",
    "FileSpec",
    "FileRole",
    "Backend",
    # Params
    "ParamDef",
    "ParamSchema",
    "EMPTY_SCHEMA",
    "WDTaggerParams",
    "wd_tagger_params",
    "MemorySnapshot",
    "InferenceMemoryRecord",
    "MemoryTrackerStats",
    # Results
    "TagResult",
    "TagEntry",
    "ScoreResult",
    "MultiScoreResult",
    "OutputType",
    "InferenceResult",
    "is_tag_result",
    "is_score_result",
    "is_multi_score_result",
    # Registry
    "registry",
    "RegistryError",
    "SessionError",
    # API
    "load",
    "load_advanced",
    "list_models",
    "list_available_devices",
    "list_plugin_classes",
    "describe",
    "describe_all",
    "set_auto_download_default",
    "get_auto_download_default",
]


# endregion Public re-Exports
