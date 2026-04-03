"""
ModelPlugin — the abstract base class every plugin must implement.

A plugin is responsible for:
  - declaring what files it needs (required_files)
  - preprocessing an image into a tensor/array
  - postprocessing raw model output into a typed result
  - optionally loading ancillary files (tag lists, mappings, etc.)

The inference engine (session.py) handles the actual forward pass —
plugins don't call the model directly.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autotagger.results import ModelResult, OutputType

if TYPE_CHECKING:
    from autotagger.result_processors import ResultProcessor


# region File Spec


class FileRole(str, Enum):
    WEIGHTS = "weights"  # model weights (.pt, .safetensors, .onnx, …)
    TAG_LIST = "tag_list"  # tag list/CSV
    MAPPING = "mapping"  # e.g. character→copyright mapping JSON
    CONFIG = "config"  # model config JSON


class Backend(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"


@dataclass
class FileSpec:
    """
    Declares one file the plugin needs to operate.

    Attributes:
        name:       The filename as it appears in the HF repo / local folder.
        role:       What kind of file this is.
        required:   If False the plugin can run without it (feature degrades).
        backends:   Which backends need this file. Empty list = all backends.
                    Use this to declare .onnx files only for Backend.ONNX, etc.
    """

    name: str
    role: FileRole
    required: bool = True
    backends: list[Backend] = field(default_factory=list)

    def needed_for(self, backend: Backend) -> bool:
        if not self.backends:
            return True
        return backend in self.backends

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        d["backends"] = [b.value for b in self.backends]
        return d


# endregion File Spec


# region ModelPlugin API


class ModelPlugin(abc.ABC):
    """
    Abstract base class for all autotagger model plugins.

    Subclasses must set class-level attributes and implement the abstract
    methods. The registry uses the class attributes for discovery and the
    loader uses required_files to know what to fetch.

    Class-level attributes to set in every subclass
    ------------------------------------------------
    model_id : str
        Canonical identifier, e.g. "wd-eva02-large-tagger".
    aliases : list[str]
        Alternative names users can pass, e.g. ["wd-eva02", "eva02-tagger"].
    output_type : OutputType
        What kind of result this plugin produces.
    required_files : list[FileSpec]
        Files the plugin needs. Declared at class level so the loader can
        inspect them before instantiating the plugin.
    default_hf_repo : str | None
        The HuggingFace repo ID this plugin uses out-of-the-box.
        None means there is no canonical upstream repo.
    supported_backends : list[Backend]
        Which inference backends this plugin supports.
    supported_processors : list[type[ResultProcessor]]
        Optional result processors this plugin is designed to work with.
    display_name : str
        Human-readable name for GUIs and listings.
    description : str
        One-line description.
    """

    # --- Subclasses must override these ---
    model_id: str = ""
    aliases: list[str] = []
    output_type: OutputType = OutputType.TAGS
    required_files: list[FileSpec] = []
    default_hf_repo: str | None = None
    supported_backends: list[Backend] = [Backend.PYTORCH, Backend.ONNX]
    supported_processors: list[type["ResultProcessor"]] = []
    display_name: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip validation on intermediate base classes (mixins, helpers)
        if not getattr(cls, "_abstract", False) and cls.model_id:
            if not cls.required_files and cls.output_type == OutputType.TAGS:
                import warnings

                warnings.warn(
                    f"Plugin '{cls.model_id}' has no required_files declared.",
                    stacklevel=2,
                )

    # --- Called by the loader after files are resolved ---

    def configure(self, **kwargs: Any) -> None:
        """
        Optional per-session plugin configuration hook.

        The session builder calls this before load_ancillary(). Plugins can use
        it for options such as download policy overrides, explicit mapping
        paths, or other runtime behavior flags.
        """

    def load_ancillary(self, file_map: dict[str, Path]) -> None:
        """
        Load tag lists, mappings, configs, etc. from resolved file paths.

        Called once after the plugin is instantiated and before inference.
        file_map keys are FileSpec.name values; values are resolved Paths.

        Default implementation does nothing — override in plugins that need it.
        """

    # --- Implement these in every concrete plugin ---

    @abc.abstractmethod
    def preprocess(self, image: Any) -> Any:
        """
        Transform a PIL Image (or numpy array) into a model-ready tensor/array.

        Returns whatever the chosen backend expects:
          - PyTorch: torch.Tensor with shape (1, C, H, W)
          - ONNX: numpy ndarray with shape (1, C, H, W)

        The inference engine passes this directly to the model.
        """

    @abc.abstractmethod
    def postprocess(
        self,
        raw_output: Any,
    ) -> ModelResult:
        """
        Convert raw model output into a typed result.

        raw_output is whatever the model/backend returns (torch.Tensor,
        numpy ndarray, etc.).
        """

    # --- Optional hooks ---

    def get_input_name(self) -> str | None:
        """
        For ONNX models: return the name of the input node.
        Return None to let the runtime auto-detect it.
        """
        return None

    # --- Introspection helpers ---

    @classmethod
    def all_names(cls) -> list[str]:
        """model_id + all aliases."""
        return [cls.model_id] + list(cls.aliases)

    @classmethod
    def files_for_backend(cls, backend: Backend) -> list[FileSpec]:
        """Subset of required_files that are needed for a given backend."""
        return [f for f in cls.required_files if f.needed_for(backend)]

    @classmethod
    def required_file_names(cls, backend: Backend) -> list[str]:
        return [f.name for f in cls.files_for_backend(backend) if f.required]

    @classmethod
    def to_dict(cls) -> dict[str, Any]:
        return {
            "model_id": cls.model_id,
            "aliases": cls.aliases,
            "display_name": cls.display_name,
            "description": cls.description,
            "output_type": cls.output_type.value,
            "default_hf_repo": cls.default_hf_repo,
            "supported_backends": [b.value for b in cls.supported_backends],
            "supported_processors": [processor.__name__ for processor in cls.supported_processors],
            "required_files": [f.to_dict() for f in cls.required_files],
        }


# endregion ModelPlugin API
