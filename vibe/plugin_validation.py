"""Validation helpers for plugin declarations."""

from __future__ import annotations

from collections import defaultdict

from vibe.backends.base import FileRole, ModelPlugin


def validate_plugin_declaration(plugin_cls: type[ModelPlugin]) -> list[str]:
    """Return non-fatal warnings for suspicious plugin declarations."""
    warnings: list[str] = []

    file_specs_by_name: dict[str, list] = defaultdict(list)
    for spec in plugin_cls.required_files:
        file_specs_by_name[spec.name].append(spec)

    for filename, specs in file_specs_by_name.items():
        if len(specs) < 2:
            continue

        keys = [spec.key or spec.name for spec in specs]
        if len(set(keys)) == len(specs):
            continue

        duplicate_repo_ids = {spec.repo_id for spec in specs}
        if len(duplicate_repo_ids) == len(specs):
            default_repo = plugin_cls.default_hf_repo
            if default_repo is None or all(repo_id != default_repo for repo_id in duplicate_repo_ids):
                continue

        warnings.append(
            f"Plugin '{plugin_cls.model_id}' declares duplicate file spec '{filename}'. "
            "Duplicate file names are only allowed when each duplicate points at a repo_id "
            "outside the model variant's default_hf_repo."
        )

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
