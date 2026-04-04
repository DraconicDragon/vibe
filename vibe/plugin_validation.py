"""Validation helpers for plugin declarations."""

from __future__ import annotations

from vibe.backends.base import FileRole, ModelPlugin


def validate_plugin_declaration(plugin_cls: type[ModelPlugin]) -> list[str]:
    """Return non-fatal warnings for suspicious plugin declarations."""
    warnings: list[str] = []

    supported = set(plugin_cls.supported_backends)
    weights_specs = [spec for spec in plugin_cls.required_files if spec.role == FileRole.WEIGHTS]

    for backend in supported:
        if not any(spec.needed_for(backend) for spec in weights_specs):
            warnings.append(
                f"Plugin '{plugin_cls.model_id}' supports backend '{backend.value}' "
                "but declares no weights file for that backend."
            )

    for spec in weights_specs:
        if not spec.backends:
            continue
        for backend in spec.backends:
            if backend not in supported:
                warnings.append(
                    f"Plugin '{plugin_cls.model_id}' declares weights file '{spec.name}' for backend "
                    f"'{backend.value}' but does not include it in supported_backends."
                )

    return warnings
