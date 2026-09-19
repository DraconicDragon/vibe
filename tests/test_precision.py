from __future__ import annotations

import pytest

from vibe.precision import (
    PrecisionPolicy,
    PrecisionRequest,
    ResolvedPrecisionPlan,
    parse_precision,
)


def test_parse_precision_defaults_and_aliases() -> None:
    # Auto / default policy preserves weights and enables auto compute
    req_none = parse_precision(None)
    assert req_none.weight == PrecisionPolicy.PRESERVE
    assert req_none.compute == PrecisionPolicy.AUTO

    req_auto = parse_precision("auto")
    assert req_auto.weight == PrecisionPolicy.PRESERVE
    assert req_auto.compute == PrecisionPolicy.AUTO


def test_parse_precision_explicit_policies() -> None:
    fp16 = parse_precision("fp16")
    assert fp16.weight == PrecisionPolicy.FP16
    assert fp16.compute == PrecisionPolicy.FP16

    bf16 = parse_precision("bf16")
    assert bf16.weight == PrecisionPolicy.BF16
    assert bf16.compute == PrecisionPolicy.BF16

    fp32 = parse_precision("fp32")
    assert fp32.weight == PrecisionPolicy.FP32
    assert fp32.compute == PrecisionPolicy.FP32


def test_parse_precision_idempotent() -> None:
    req = PrecisionRequest(weight=PrecisionPolicy.FP16, compute=PrecisionPolicy.FP32)
    assert parse_precision(req) is req


def test_parse_precision_rejects_invalid_strings() -> None:
    with pytest.raises(ValueError, match="Unsupported precision 'invalid'"):
        parse_precision("invalid")


def test_precision_to_dict_serialization() -> None:
    req = PrecisionRequest(weight=PrecisionPolicy.FP16, compute=PrecisionPolicy.AUTO)
    data = req.to_dict()
    assert data["weight"] == "fp16"
    assert data["compute"] == "auto"
    assert data["fallback_allowed"] is True

    plan = ResolvedPrecisionPlan(
        weight_dtype="float16",
        compute_dtype="float16",
        autocast_enabled=True,
    )
    plan_data = plan.to_dict()
    assert plan_data["weight_dtype"] == "float16"
    assert plan_data["autocast_enabled"] is True
