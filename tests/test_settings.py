from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from vibe.exceptions import SessionError
from vibe.settings import (
    InferenceRequest,
    OptionScope,
    SettingGroupSpec,
    compile_settings,
)


class AlphaConfig(BaseModel):
    """Configuration A."""

    steps: int = Field(default=20, ge=1, le=100)
    mode: str = Field(default="fast")


class BetaConfig(BaseModel):
    """Configuration B."""

    scale: float = Field(default=1.0, ge=0.1, le=5.0)


class ForeignConfig(BaseModel):
    """Config that does not belong to the target model."""

    unsupported: bool = True


@pytest.fixture
def supported_specs() -> list[SettingGroupSpec]:
    return [
        SettingGroupSpec.from_model(AlphaConfig, id="alpha", scope=OptionScope.INFER),
        SettingGroupSpec.from_model(BetaConfig, id="beta", scope=OptionScope.INFER),
    ]


def test_setting_group_spec_schema_generation() -> None:
    spec = SettingGroupSpec.from_model(AlphaConfig, id="alpha", display_name="Alpha Settings")
    schema = spec.json_schema()

    assert spec.id == "alpha"
    assert spec.display_name == "Alpha Settings"
    assert schema["type"] == "object"
    assert "steps" in schema["properties"]
    assert schema["properties"]["steps"]["default"] == 20
    assert schema["properties"]["steps"]["minimum"] == 1
    assert schema["properties"]["steps"]["maximum"] == 100


def test_compile_settings_empty_inputs(supported_specs: list[SettingGroupSpec]) -> None:
    req = compile_settings(None, supported_specs)
    assert isinstance(req, InferenceRequest)
    assert req.configs == ()
    assert req.get(AlphaConfig) is None


def test_compile_settings_single_model_instance(supported_specs: list[SettingGroupSpec]) -> None:
    cfg = AlphaConfig(steps=50, mode="slow")
    req = compile_settings(cfg, supported_specs)

    resolved = req.get(AlphaConfig)
    assert resolved is not None
    assert resolved.steps == 50
    assert resolved.mode == "slow"


def test_compile_settings_sequence_of_model_instances(supported_specs: list[SettingGroupSpec]) -> None:
    req = compile_settings([AlphaConfig(steps=30), BetaConfig(scale=2.5)], supported_specs)

    alpha = req.get(AlphaConfig)
    beta = req.get(BetaConfig)
    assert alpha is not None and alpha.steps == 30
    assert beta is not None and beta.scale == 2.5


def test_compile_settings_grouped_mapping(supported_specs: list[SettingGroupSpec]) -> None:
    payload = {
        "alpha": {"steps": 40},
        "beta": {"scale": 3.0},
    }
    req = compile_settings(payload, supported_specs)

    alpha = req.get(AlphaConfig)
    beta = req.get(BetaConfig)
    assert alpha is not None and alpha.steps == 40
    assert beta is not None and beta.scale == 3.0


def test_compile_settings_rejects_foreign_inference_request(
    supported_specs: list[SettingGroupSpec],
) -> None:
    foreign_req = InferenceRequest(configs=(ForeignConfig(),))

    with pytest.raises(SessionError, match="unsupported config type 'ForeignConfig'"):
        compile_settings(foreign_req, supported_specs)


def test_compile_settings_enforces_allowed_scopes() -> None:
    spec_load = SettingGroupSpec.from_model(AlphaConfig, id="alpha", scope=OptionScope.LOAD)

    with pytest.raises(SessionError, match="has scope 'load', but only \\['infer'\\] are permitted"):
        compile_settings(
            AlphaConfig(steps=10),
            [spec_load],
            allowed_scopes={OptionScope.INFER},
        )


def test_compile_settings_rejects_out_of_bounds_validation(
    supported_specs: list[SettingGroupSpec],
) -> None:
    # steps must be <= 100
    with pytest.raises(SessionError, match="Settings validation failed for 'alpha'"):
        compile_settings({"alpha": {"steps": 999}}, supported_specs)
