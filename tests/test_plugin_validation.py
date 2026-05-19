from __future__ import annotations

from dataclasses import dataclass, field

from vibe.backends.base import Backend, FileRole, FileSpec, ModelPlugin
from vibe.plugin_validation import validate_plugin_declaration
from vibe.results import OutputType, TagEntry, TagResult


class _BaseTestPlugin(ModelPlugin):
    model_id = ""
    aliases: list[str] = []
    output_type = OutputType.TAGS
    required_files: list[FileSpec] = []
    default_hf_repo = None
    supported_backends: list[Backend] = []
    display_name = "test"
    description = "test"

    def preprocess(self, image):
        return image

    def postprocess(self, raw_output):
        del raw_output
        return TagResult()


class _MissingWeightsPlugin(_BaseTestPlugin):
    pass


class _MismatchedWeightsBackendPlugin(_BaseTestPlugin):
    pass


class _DuplicateNameWithKeyPlugin(_BaseTestPlugin):
    pass


def test_validate_plugin_declaration_warns_when_supported_backend_has_no_weights() -> None:
    _MissingWeightsPlugin.model_id = "missing-weights"
    _MissingWeightsPlugin.supported_backends = [Backend.PYTORCH]
    _MissingWeightsPlugin.required_files = []

    warnings = validate_plugin_declaration(_MissingWeightsPlugin)

    assert warnings
    assert "supports backend 'pytorch'" in warnings[0]


def test_validate_plugin_declaration_warns_when_weights_backend_not_supported() -> None:
    _MismatchedWeightsBackendPlugin.model_id = "mismatched-backend"
    _MismatchedWeightsBackendPlugin.supported_backends = [Backend.ONNX]
    _MismatchedWeightsBackendPlugin.required_files = [
        FileSpec(
            name="model.safetensors",
            role=FileRole.WEIGHTS,
            required=True,
            backends=[Backend.PYTORCH],
        )
    ]

    warnings = validate_plugin_declaration(_MismatchedWeightsBackendPlugin)

    assert warnings
    assert any("does not include it in supported_backends" in message for message in warnings)


def test_validate_plugin_declaration_allows_duplicate_names_with_unique_keys() -> None:
    _DuplicateNameWithKeyPlugin.model_id = "duplicate-keys-ok"
    _DuplicateNameWithKeyPlugin.supported_backends = [Backend.PYTORCH]
    _DuplicateNameWithKeyPlugin.required_files = [
        FileSpec(
            name="model.safetensors",
            key="mlp_weights",
            role=FileRole.WEIGHTS,
            required=True,
            backends=[Backend.PYTORCH],
            repo_id="repo/a",
        ),
        FileSpec(
            name="model.safetensors",
            key="clip_weights",
            role=FileRole.WEIGHTS,
            required=True,
            backends=[Backend.PYTORCH],
            repo_id="repo/b",
        ),
    ]

    warnings = validate_plugin_declaration(_DuplicateNameWithKeyPlugin)

    assert not warnings
