"""
Granular capability protocols, contract verification, and trusted session views.
"""

from __future__ import annotations

import inspect
import os
import types
from dataclasses import dataclass
from typing import Any, Protocol, Union, get_args, get_origin, get_type_hints, runtime_checkable

from vibe.exceptions import PluginContractError
from vibe.metadata import CalibrationTable, LabelCatalog, TagFilterRecommendation, ThresholdTable

# region Introspection Helper


def _get_protocol_required_attrs(proto: type[Any]) -> set[str]:
    """
    Extract required attribute names from a Protocol.
    Uses inspect.get_annotations to avoid false positives from inherited methods or properties.
    """
    all_annotations: dict[str, Any] = {}
    for cls in reversed(proto.__mro__):
        all_annotations.update(inspect.get_annotations(cls, eval_str=False))

    return {name for name in all_annotations if not name.startswith("_")}


def _is_matching_type(actual_value: Any, expected_type: Any) -> bool:
    """
    Check if actual_value satisfies expected_type annotation, correctly handling Unions and None.
    """
    if expected_type is None:
        return True

    origin = get_origin(expected_type)

    # Handle Union / Optional types (e.g. ThresholdTable | None)
    if origin is types.UnionType or origin is Union:
        args = get_args(expected_type)
        if actual_value is None:
            return type(None) in args
        valid_types = tuple(get_origin(arg) or arg for arg in args if arg is not type(None))
        return isinstance(actual_value, valid_types)

    # For non-union types, None is never valid
    if actual_value is None:
        return False

    check_type = origin if origin is not None else expected_type
    try:
        return isinstance(actual_value, check_type)
    except TypeError:
        return True


# endregion


# region Granular Capability Protocols


@runtime_checkable
class LabelCatalogProvider(Protocol):
    """REQUIRED: Core vocabulary. Non-negotiable for taggers and categorical scorers."""

    catalog: LabelCatalog


@runtime_checkable
class ThresholdProvider(Protocol):
    """OPTIONAL: Calibrated per-label decision thresholds. Absence is valid."""

    thresholds: ThresholdTable


@runtime_checkable
class CalibrationProvider(Protocol):
    """OPTIONAL: Score percentile/distribution calibration tables."""

    calibration: CalibrationTable


# Master set of all framework-known capability protocols for bidirectional auditing
ALL_CAPABILITY_PROTOCOLS = {
    LabelCatalogProvider,
    ThresholdProvider,
    CalibrationProvider,
}

# endregion


# region Verification


def verify_plugin_contract(plugin: Any) -> None:
    """
    Perform a bidirectional audit of a plugin's implemented attributes against its declared contract.
    """
    declared = set(getattr(plugin, "implements", ()))
    model_name = getattr(getattr(plugin, "identity", None), "model_id", type(plugin).__name__)

    # CHECK 1: All declared protocols are fully satisfied (nag if missing OR wrong type)
    for proto in declared:
        required_attrs = _get_protocol_required_attrs(proto)
        try:
            hints = get_type_hints(proto, include_extras=True)
        except Exception:
            hints = {}

        errors = []
        for attr in required_attrs:
            if not hasattr(plugin, attr):
                errors.append(f"missing '{attr}'")
            else:
                expected_type = hints.get(attr)
                actual_value = getattr(plugin, attr)

                if not _is_matching_type(actual_value, expected_type):
                    type_name = type(actual_value).__name__ if actual_value is not None else "None"
                    exp_name = getattr(expected_type, "__name__", str(expected_type))
                    errors.append(f"'{attr}' has type {type_name}, expected {exp_name}")

        if errors:
            raise PluginContractError(
                f"Plugin '{model_name}' declares {proto.__name__} but has contract violations: {'; '.join(errors)}"
            )

    # CHECK 2: No undeclared capability protocols are present (nag if extra/zombie)
    satisfied_but_undeclared = set()
    for proto in ALL_CAPABILITY_PROTOCOLS:
        if proto not in declared:
            required_attrs = _get_protocol_required_attrs(proto)
            try:
                proto_hints = get_type_hints(proto, include_extras=True)
            except Exception:
                proto_hints = {}

            # Must exist AND match the protocol's expected type
            if required_attrs and all(
                hasattr(plugin, attr) and _is_matching_type(getattr(plugin, attr), proto_hints.get(attr))
                for attr in required_attrs
            ):
                satisfied_but_undeclared.add(proto)

    if satisfied_but_undeclared:
        proto_names = [p.__name__ for p in satisfied_but_undeclared]
        raise PluginContractError(
            f"Plugin '{model_name}' satisfies protocols {proto_names} "
            f"but did not declare them in `implements`. "
            f"This indicates stale/zombie attributes or a missing contract declaration."
        )


# endregion


# region Trusted Capability Views


@dataclass(frozen=True, slots=True)
class TaggerView:
    catalog: LabelCatalog
    thresholds: ThresholdTable | None
    recommendation: TagFilterRecommendation | None

    def __post_init__(self) -> None:
        # DEV-ONLY SAFETY NET: Catches contract/view mismatches during migration
        if os.environ.get("VIBE_STRICT_VIEWS", "1").lower() in ("1", "true", "yes"):
            if not isinstance(self.catalog, LabelCatalog):
                raise AssertionError(
                    f"TaggerView constructed with invalid catalog type: {type(self.catalog).__name__}. "
                    f"This indicates a verify_plugin_contract bug or migration oversight."
                )
            if self.thresholds is not None and not isinstance(self.thresholds, ThresholdTable):
                raise AssertionError(
                    f"TaggerView constructed with invalid thresholds type: {type(self.thresholds).__name__}."
                )
            if self.recommendation is not None and not isinstance(self.recommendation, TagFilterRecommendation):
                raise AssertionError(
                    f"TaggerView constructed with invalid recommendation type: {type(self.recommendation).__name__}."
                )


@dataclass(frozen=True, slots=True)
class ScorerView:
    catalog: LabelCatalog | None
    calibration: CalibrationTable | None

    def __post_init__(self) -> None:
        # DEV-ONLY SAFETY NET
        if os.environ.get("VIBE_STRICT_VIEWS", "1").lower() in ("1", "true", "yes"):
            if self.catalog is not None and not isinstance(self.catalog, LabelCatalog):
                raise AssertionError(
                    f"ScorerView constructed with invalid catalog type: {type(self.catalog).__name__}."
                )
            if self.calibration is not None and not isinstance(self.calibration, CalibrationTable):
                raise AssertionError(
                    f"ScorerView constructed with invalid calibration type: {type(self.calibration).__name__}."
                )


# endregion
