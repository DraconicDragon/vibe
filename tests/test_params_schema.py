from __future__ import annotations

import pytest

from autotagger.params import ParamDef, ParamSchema


def test_schema_fills_defaults_and_coerces_types() -> None:
    schema = ParamSchema(
        [
            ParamDef("threshold", float, default=0.5, range=(0.0, 1.0)),
            ParamDef("include_all", bool, default=False),
        ]
    )

    validated = schema.validate({"threshold": "0.25", "include_all": "true"})

    assert validated == {"threshold": 0.25, "include_all": True}


def test_schema_rejects_out_of_range_values() -> None:
    schema = ParamSchema([ParamDef("threshold", float, default=0.5, range=(0.0, 1.0))])

    with pytest.raises(ValueError) as excinfo:
        schema.validate({"threshold": 1.2})

    assert "outside allowed range" in str(excinfo.value)


def test_schema_unknown_key_includes_suggestion() -> None:
    schema = ParamSchema(
        [
            ParamDef("general_threshold", float, default=0.35),
            ParamDef("character_threshold", float, default=0.85),
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        schema.validate({"general_thresold": 0.3})

    message = str(excinfo.value)
    assert "Unknown parameter(s)" in message
    assert "general_threshold" in message
