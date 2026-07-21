"""
ModelRegistry — central index of all known plugins.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import pkgutil
import threading
import warnings
from typing import TYPE_CHECKING

from vibe.result_transforms import ResultTransform, TransformInfo

if TYPE_CHECKING:
    from vibe.backends.base import ModelDescriptor, ModelPlugin


class RegistryError(Exception):
    """Raised when a plugin lookup fails."""


class ModelRegistry:
    """
    Central index for registering and looking up ModelPlugin classes.
    """

    def __init__(self) -> None:
        # model_id → plugin class
        self._plugins: dict[str, type[ModelPlugin]] = {}
        self._discovered = False
        self._discover_lock = threading.Lock()

    def ensure_discovered(self) -> None:
        if self._discovered:
            return

        with self._discover_lock:
            if self._discovered:
                return

            self.discover_all()
            self._discovered = True

    def register(self, plugin_cls: type[ModelPlugin]) -> None:
        """
        Register a plugin class.

        Concrete classes are indexed using their class-level identity.model_id.
        """
        if inspect.isabstract(plugin_cls):
            return

        identity = getattr(plugin_cls, "identity", None)
        if not identity or not identity.model_id:
            raise ValueError(f"Cannot register plugin {plugin_cls.__name__}: identity is missing or model_id is empty.")

        mid = identity.model_id

        if mid in self._plugins:
            existing = self._plugins[mid]
            if existing is plugin_cls:
                return  # already registered
            raise ValueError(
                f"model_id '{mid}' is already registered by {existing.__name__}. "
                f"Cannot register {plugin_cls.__name__} with the same ID."
            )

        self._plugins[mid] = plugin_cls

    def unregister(self, model_id: str) -> None:
        """Remove a plugin from the registry index."""
        self._plugins.pop(model_id, None)

    def get(self, name: str) -> type[ModelPlugin]:
        """Resolve a model ID string to its registered plugin class."""
        if name in self._plugins:
            return self._plugins[name]

        suggestions = self._suggest(name)
        msg = f"No plugin found for '{name}'."
        if suggestions:
            msg += f" Did you mean: {suggestions}?"
        else:
            msg += f" Known models: {self.list_model_ids()}"
        raise RegistryError(msg)

    def get_by_class_name(self, class_name: str) -> type[ModelPlugin]:
        """Look up a plugin strictly by its Python class name (e.g. 'WDEva02Plugin')."""
        for cls in self._plugins.values():
            if cls.__name__ == class_name:
                return cls
        raise RegistryError(f"No plugin class named '{class_name}'. Known classes: {self.list_plugin_classes()}")

    def is_known(self, name: str) -> bool:
        """Return True if the model ID string is registered."""
        return name in self._plugins

    def list_model_ids(self) -> list[str]:
        """Return a sorted list of all registered model IDs."""
        return sorted(self._plugins.keys())

    def list_all(self) -> list[ModelDescriptor]:
        """Return structured metadata descriptions for all registered plugins."""
        return [cls.describe() for cls in self._plugins.values()]

    def list_plugin_classes(self) -> list[str]:
        """Return plugin class names ordered by model ID."""
        names: list[str] = []
        for model_id in self.list_model_ids():
            names.append(self._plugins[model_id].__name__)
        return names

    def discover_builtins(self) -> None:
        """Import all built-in modules to trigger auto-registration."""
        import vibe.plugins as plugins_pkg

        for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
            module_name = f"vibe.plugins.{module_info.name}"
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                warnings.warn(
                    f"Failed to import plugin module '{module_name}': {exc}",
                    stacklevel=2,
                )

    def discover_entry_points(self) -> None:
        """Load external third-party plugins declared via pyproject.toml entry points."""
        try:
            eps = importlib.metadata.entry_points(group="vibe.plugins")
        except Exception:
            return

        for ep in eps:
            try:
                plugin_cls = ep.load()
                self.register(plugin_cls)
            except Exception as exc:
                warnings.warn(
                    f"Failed to load entry point plugin '{ep.name}': {exc}",
                    stacklevel=2,
                )

    def discover_all(self) -> None:
        """Run all plugin discovery mechanisms."""
        self.discover_builtins()
        self.discover_entry_points()

    def _suggest(self, name: str, max_suggestions: int = 3) -> list[str]:
        name_lower = name.lower()
        all_names = list(self._plugins.keys())
        return [n for n in all_names if name_lower in n.lower() or n.lower() in name_lower][:max_suggestions]


class TransformRegistry:
    """Central index for registering and looking up ResultTransform classes."""

    def __init__(self) -> None:
        self._transforms: dict[str, type["ResultTransform"]] = {}

    def register(self, transform_cls: type["ResultTransform"]) -> None:
        tid = getattr(transform_cls, "transform_id", None)
        if not tid:
            return
        self._transforms[tid] = transform_cls

    def get(self, transform_id: str) -> type["ResultTransform"]:
        if transform_id not in self._transforms:
            raise RegistryError(f"No transform found for '{transform_id}'. Known: {list(self._transforms)}")
        return self._transforms[transform_id]

    def list_all(self) -> list["TransformInfo"]:
        return [cls.describe() for cls in self._transforms.values()]


model_registry = ModelRegistry()
transform_registry = TransformRegistry()
