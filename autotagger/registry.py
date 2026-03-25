"""
ModelRegistry — central index of all known plugins.

Plugins are registered in two ways:
  1. Auto-discovery: all modules in autotagger/plugins/ are imported at
     startup, and any ModelPlugin subclass with a non-empty model_id is
     registered automatically.
  2. Third-party entry points: packages can ship plugins by declaring an
     entry point in their pyproject.toml under the group "autotagger.plugins".

After discovery, the registry resolves names/aliases to plugin classes and
supports override — useful when a user wants to run an arbitrary HF repo
through an existing plugin's inference code.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import warnings
from typing import TYPE_CHECKING, Any

from autotagger.plugin_validation import validate_plugin_declaration

if TYPE_CHECKING:
    from autotagger.backends.base import ModelPlugin


class RegistryError(Exception):
    """Raised when a plugin lookup fails."""


class ModelRegistry:
    """
    Singleton-style registry (one instance is created in __init__.py).
    You can also instantiate your own for testing.
    """

    def __init__(self) -> None:
        # model_id → plugin class
        self._plugins: dict[str, type[ModelPlugin]] = {}
        # alias → model_id (for quick resolution)
        self._aliases: dict[str, str] = {}

    # region Registration

    def register(self, plugin_cls: type[ModelPlugin]) -> None:
        """
        Register a plugin class.

        Raises if model_id is empty or already registered by a *different* class.
        Re-registering the same class is a no-op.
        """
        mid = plugin_cls.model_id
        if not mid:
            raise ValueError(f"Cannot register plugin {plugin_cls.__name__}: model_id is empty.")

        if mid in self._plugins:
            existing = self._plugins[mid]
            if existing is plugin_cls:
                return  # already registered, fine
            raise ValueError(
                f"model_id '{mid}' is already registered by {existing.__name__}. "
                f"Cannot register {plugin_cls.__name__} with the same id."
            )

        for warning_message in validate_plugin_declaration(plugin_cls):
            warnings.warn(warning_message, stacklevel=2)

        self._plugins[mid] = plugin_cls

        # Index all aliases → model_id
        for alias in plugin_cls.aliases:
            if alias in self._aliases and self._aliases[alias] != mid:
                warnings.warn(
                    f"Alias '{alias}' from plugin '{mid}' conflicts with "
                    f"existing alias pointing to '{self._aliases[alias]}'. "
                    f"The new plugin wins.",
                    stacklevel=2,
                )
            self._aliases[alias] = mid

    def unregister(self, model_id: str) -> None:
        """Remove a plugin (mostly useful in tests)."""
        plugin = self._plugins.pop(model_id, None)
        if plugin:
            for alias in plugin.aliases:
                self._aliases.pop(alias, None)


# endregion Registration


# region Lookup

    def get(self, name: str) -> type[ModelPlugin]:
        """
        Resolve a name (model_id or alias) to a plugin class.

        Raises RegistryError with helpful message if not found.
        """
        # Direct model_id match
        if name in self._plugins:
            return self._plugins[name]

        # Alias match
        if name in self._aliases:
            return self._plugins[self._aliases[name]]

        # Friendly error
        suggestions = self._suggest(name)
        msg = f"No plugin found for '{name}'."
        if suggestions:
            msg += f" Did you mean: {suggestions}?"
        else:
            msg += f" Known models: {self.list_model_ids()}"
        raise RegistryError(msg)

    def get_by_class_name(self, class_name: str) -> type[ModelPlugin]:
        """
        Look up a plugin by its Python class name.

        This is the advanced-user override path: the user knows the class name
        of the inference code they want to use with an arbitrary HF repo.
        """
        for cls in self._plugins.values():
            if cls.__name__ == class_name:
                return cls
        raise RegistryError(
            f"No plugin class named '{class_name}'. " f"Known classes: {[c.__name__ for c in self._plugins.values()]}"
        )

    def is_known(self, name: str) -> bool:
        """Return True if name resolves to a registered plugin."""
        return name in self._plugins or name in self._aliases


# endregion Lookup


# region List

    def list_model_ids(self) -> list[str]:
        return sorted(self._plugins.keys())

    def list_all(self) -> list[dict[str, Any]]:
        """Return a list of dicts describing every registered plugin."""
        return [cls.to_dict() for cls in self._plugins.values()]

    def list_plugin_classes(self) -> list[str]:
        """Return plugin class names in model-id order."""
        names: list[str] = []
        for model_id in self.list_model_ids():
            names.append(self._plugins[model_id].__name__)
        return names

    def __len__(self) -> int:
        return len(self._plugins)

    def __repr__(self) -> str:
        return f"ModelRegistry({self.list_model_ids()})"


# endregion List


# region Discovery

    def discover_builtins(self) -> None:
        """
        Import every module in autotagger/plugins/ so their ModelPlugin
        subclasses get defined and trigger auto-registration.
        """
        import autotagger.plugins as plugins_pkg

        for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
            module_name = f"autotagger.plugins.{module_info.name}"
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                warnings.warn(
                    f"Failed to import plugin module '{module_name}': {exc}",
                    stacklevel=2,
                )

    def discover_entry_points(self) -> None:
        """
        Load plugins declared via Python entry points.

        Third-party packages add plugins by declaring in pyproject.toml:

            [project.entry-points."autotagger.plugins"]
            my_plugin = "my_package.my_module:MyPlugin"

        Each entry point value should be a ModelPlugin subclass.
        """
        try:
            eps = importlib.metadata.entry_points(group="autotagger.plugins")
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
        """Run all discovery mechanisms. Called once at library init."""
        self.discover_builtins()
        self.discover_entry_points()


# endregion Discovery


# region Internal Helpers

    def _suggest(self, name: str, max_suggestions: int = 3) -> list[str]:
        """Very basic fuzzy suggestion — find names that share a substring."""
        name_lower = name.lower()
        all_names = list(self._plugins.keys()) + list(self._aliases.keys())
        return [n for n in all_names if name_lower in n.lower() or n.lower() in name_lower][:max_suggestions]


# endregion Internal Helpers


# region Auto Registration


# When a ModelPlugin subclass is defined (i.e. when its module is imported),
# we want it to register itself automatically — no manual register() call
# needed in plugin code.
#
# We do this by monkeypatching __init_subclass__ on ModelPlugin after the
# registry is created. See autotagger/__init__.py where this is wired up.


def _make_auto_register_hook(registry: ModelRegistry):
    """
    Returns a function that registers a plugin class when called.
    Meant to be called from ModelPlugin.__init_subclass__.
    """

    def auto_register(cls: type[ModelPlugin]) -> None:
        # Skip abstract base classes and mixins (no model_id)
        if cls.model_id:
            try:
                registry.register(cls)
            except ValueError as exc:
                warnings.warn(str(exc), stacklevel=3)

    return auto_register


# endregion Auto Registration
