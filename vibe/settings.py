"""
Unified settings, configuration schema, and option specification system using Pydantic v2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from vibe.exceptions import SessionError

T = TypeVar("T", bound=BaseModel)


class OptionScope(StrEnum):
    """Execution lifecycle boundary where an option is evaluated."""

    LOAD = "load"
    SESSION = "session"
    INFER = "infer"


@dataclass(frozen=True, kw_only=True)
class SettingGroupSpec:
    """Describes a configurable setting group for a model backed by a Pydantic model."""

    id: str
    display_name: str
    description: str
    model_type: type[BaseModel]
    recommended: BaseModel | None = None
    scope: OptionScope = OptionScope.INFER

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("SettingGroupSpec.id must be a non-empty string.")
        if not (isinstance(self.model_type, type) and issubclass(self.model_type, BaseModel)):
            raise TypeError(f"Setting group '{self.id}' model_type must be a subclass of pydantic.BaseModel.")
        if self.recommended is not None and not isinstance(self.recommended, self.model_type):
            raise TypeError(
                f"Setting group '{self.id}' recommended config must be an instance of '{self.model_type.__name__}'."
            )

    @classmethod
    def from_model(
        cls,
        model_cls: type[BaseModel],
        *,
        id: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        recommended: BaseModel | None = None,
        scope: OptionScope = OptionScope.INFER,
    ) -> SettingGroupSpec:
        """Construct a SettingGroupSpec directly from a Pydantic BaseModel class."""
        if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
            raise TypeError(f"Setting group model '{model_cls}' must be a subclass of pydantic.BaseModel.")

        setting_id = id or model_cls.__name__.lower().removesuffix("settings")
        disp_name = display_name or getattr(model_cls, "title", setting_id.replace("_", " ").title())
        desc = description or (model_cls.__doc__ or "").strip()

        return cls(
            id=setting_id,
            display_name=disp_name,
            description=desc,
            model_type=model_cls,
            recommended=recommended,
            scope=scope,
        )

    def json_schema(self) -> dict[str, Any]:
        """Return the standard JSON Schema for this setting model."""
        return self.model_type.model_json_schema()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "scope": self.scope.value,
            "schema": self.json_schema(),
            "recommended": self.recommended.model_dump(mode="json") if self.recommended else None,
        }


@dataclass(frozen=True)
class InferenceRequest:
    """Immutable, typed per-call configuration bundle passed to preprocessing and runtime."""

    configs: tuple[BaseModel, ...] = ()

    def get(self, config_type: type[T]) -> T | None:
        """Retrieve a typed configuration instance by Pydantic class."""
        for cfg in self.configs:
            if isinstance(cfg, config_type):
                return cfg
        return None


def serialize_value(val: Any) -> Any:
    """Helper to serialize values and models to JSON-safe primitives."""
    if val is None:
        return None
    if isinstance(val, BaseModel):
        return val.model_dump(mode="json")
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return serialize_value(val.to_dict())
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, (list, tuple, set)):
        return [serialize_value(item) for item in val]
    if isinstance(val, Mapping):
        return {str(k): serialize_value(v) for k, v in val.items()}
    return str(val)


def compile_settings(
    settings_input: Mapping[str, Mapping[str, Any]]
    | Mapping[str, Any]
    | Sequence[BaseModel]
    | BaseModel
    | InferenceRequest
    | None,
    supported_settings: Sequence[SettingGroupSpec],
    *,
    allowed_scopes: set[OptionScope] | None = None,
) -> InferenceRequest:
    """
    Compile typed setting instances or dictionary payloads against supported SettingGroupSpecs.
    Optionally enforces allowed lifecycle scopes (e.g. rejecting LOAD-time options during INFER).
    """
    supported_by_id = {s.id: s for s in supported_settings}
    supported_by_type = {s.model_type: s for s in supported_settings}
    supported_types = set(supported_by_type.keys())

    def _check_scope(spec: SettingGroupSpec) -> None:
        if allowed_scopes is not None and spec.scope not in allowed_scopes:
            allowed_names = [s.value for s in allowed_scopes]
            raise SessionError(
                f"Setting group '{spec.id}' has scope '{spec.scope.value}', but only {allowed_names} "
                "are permitted in this context."
            )

    # 1. Foreign InferenceRequest validation
    if isinstance(settings_input, InferenceRequest):
        for cfg in settings_input.configs:
            if type(cfg) not in supported_types:
                raise SessionError(
                    f"InferenceRequest contains unsupported config type '{type(cfg).__name__}'. "
                    f"Supported types for this model: {[t.__name__ for t in supported_types]}"
                )
            spec = supported_by_type[type(cfg)]
            _check_scope(spec)
        return settings_input

    # 2. Allow passing a single BaseModel directly
    if isinstance(settings_input, BaseModel):
        settings_input = (settings_input,)

    if not settings_input or not supported_settings:
        return InferenceRequest()

    compiled: list[BaseModel] = []
    seen_types: set[type[BaseModel]] = set()

    # 3. Sequence of pre-instantiated Pydantic models
    if isinstance(settings_input, Sequence) and not isinstance(settings_input, (str, bytes)):
        for item in settings_input:
            if not isinstance(item, BaseModel):
                raise SessionError(
                    f"Expected Pydantic BaseModel instance in settings sequence, got {type(item).__name__}."
                )
            if type(item) not in supported_types:
                raise SessionError(
                    f"Setting model '{type(item).__name__}' is not supported by this model. "
                    f"Supported: {[t.__name__ for t in supported_types]}"
                )
            spec = supported_by_type[type(item)]
            _check_scope(spec)

            if type(item) in seen_types:
                raise SessionError(f"Duplicate setting model '{type(item).__name__}' provided.")
            seen_types.add(type(item))
            compiled.append(item)

    # 4. Mapping of setting payloads (nested by group ID or flat for single model)
    elif isinstance(settings_input, Mapping):
        is_grouped = any(str(k) in supported_by_id for k in settings_input)
        has_nested_mappings = any(isinstance(v, Mapping) for v in settings_input.values())

        if is_grouped:
            for raw_group_id, option_dict in settings_input.items():
                group_id = str(raw_group_id)
                if group_id not in supported_by_id:
                    raise SessionError(
                        f"Unknown setting group '{group_id}' requested. Supported: {list(supported_by_id.keys())}"
                    )
                spec = supported_by_id[group_id]
                _check_scope(spec)

                if spec.model_type in seen_types:
                    raise SessionError(f"Duplicate setting group '{group_id}' provided.")
                seen_types.add(spec.model_type)

                payload = dict(option_dict) if isinstance(option_dict, Mapping) else {}
                try:
                    instantiated = spec.model_type.model_validate(payload)
                except ValidationError as exc:
                    raise SessionError(f"Settings validation failed for '{group_id}': {exc}") from exc
                compiled.append(instantiated)
        elif len(supported_settings) == 1:
            spec = supported_settings[0]
            # Guard against misspelled group IDs when passing nested dictionaries:
            # If the payload has nested mappings but no key matches a field of the single model,
            # the user intended a grouped dictionary (e.g. {'jtp_hydra_typo': {'seqlen': 512}}).
            if has_nested_mappings and not any(str(k) in spec.model_type.model_fields for k in settings_input):
                raise SessionError(
                    f"Unknown setting group or option '{next(iter(settings_input.keys()))}'. "
                    f"Supported setting group is '{spec.id}'. Supported options: {list(spec.model_type.model_fields.keys())}"
                )

            _check_scope(spec)
            try:
                instantiated = spec.model_type.model_validate(dict(settings_input))
            except ValidationError as exc:
                raise SessionError(f"Settings validation failed for '{spec.id}': {exc}") from exc
            compiled.append(instantiated)
        else:
            raise SessionError(
                f"Ambiguous or unknown settings dictionary {list(settings_input.keys())}. "
                f"Model supports multiple setting groups {list(supported_by_id.keys())}. "
                "Pass a dictionary grouped by setting ID (e.g. {'setting_id': {...}})."
            )
    else:
        raise SessionError(f"Invalid settings input format: {type(settings_input).__name__}.")

    return InferenceRequest(configs=tuple(compiled))
