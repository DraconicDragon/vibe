"""
Unified feature, configuration schema, and option specification system.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints

from vibe.exceptions import SessionError

T = TypeVar("T")


class OptionScope(str, Enum):
    """Execution lifecycle boundary where an option is evaluated."""

    LOAD = "load"
    SESSION = "session"
    INFER = "infer"


@dataclass(frozen=True)
class ValueSchema:
    """Describes the shape, type, choices, and validation rules of a configuration value."""

    kind: Literal[
        "bool",
        "int",
        "float",
        "string",
        "mapping",
        "list",
        "json",
    ]
    nullable: bool = False
    choices: tuple[Any, ...] | None = None
    choice_policy: Literal["strict", "suggested"] = "strict"
    key_schema: ValueSchema | None = None
    value_schema: ValueSchema | None = None
    known_keys: tuple[str, ...] | None = None
    allow_custom_keys: bool = False
    known_keys_source: Literal["output_categories"] | None = None

    def __post_init__(self) -> None:
        valid_kinds = {"bool", "int", "float", "string", "mapping", "list", "json"}
        if self.kind not in valid_kinds:
            raise ValueError(f"Unknown ValueSchema kind '{self.kind}'.")
        if self.choice_policy not in {"strict", "suggested"}:
            raise ValueError(f"Unknown ValueSchema choice policy '{self.choice_policy}'.")
        if self.known_keys_source not in {None, "output_categories"}:
            raise ValueError(f"Unknown ValueSchema known_keys_source '{self.known_keys_source}'.")

    def resolve_context(self, *, output_categories: tuple[str, ...]) -> ValueSchema:
        """Resolve model-provided metadata references without changing validation semantics."""
        key_schema = self.key_schema.resolve_context(output_categories=output_categories) if self.key_schema else None
        value_schema = (
            self.value_schema.resolve_context(output_categories=output_categories) if self.value_schema else None
        )
        known_keys = self.known_keys
        known_keys_source = self.known_keys_source
        if known_keys_source == "output_categories" and output_categories:
            known_keys = output_categories
            known_keys_source = None

        if (
            key_schema is not self.key_schema
            or value_schema is not self.value_schema
            or known_keys != self.known_keys
            or known_keys_source != self.known_keys_source
        ):
            return dataclasses.replace(
                self,
                key_schema=key_schema,
                value_schema=value_schema,
                known_keys=known_keys,
                known_keys_source=known_keys_source,
            )
        return self

    def validate(self, value: Any, context_name: str) -> Any:
        """Validate and normalize a value against this schema, raising SessionError on mismatch."""
        if value is None:
            if self.nullable:
                return None
            raise SessionError(f"Value for '{context_name}' cannot be null.")

        if self.kind == "bool":
            if not isinstance(value, bool):
                raise SessionError(f"Expected boolean for '{context_name}', got {type(value).__name__}.")
            return value

        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SessionError(f"Expected integer for '{context_name}', got {type(value).__name__}.")
            try:
                val_int = int(value)
            except (OverflowError, ValueError) as exc:
                raise SessionError(f"Expected finite integer for '{context_name}'.") from exc
            if val_int != value:
                raise SessionError(f"Expected integer for '{context_name}', got {type(value).__name__}.")
            if self.choices is not None and self.choice_policy == "strict" and val_int not in self.choices:
                raise SessionError(f"Invalid choice '{val_int}' for '{context_name}'. Choices: {self.choices}")
            return val_int

        if self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SessionError(f"Expected float for '{context_name}', got {type(value).__name__}.")
            try:
                val_float = float(value)
            except OverflowError as exc:
                raise SessionError(f"Expected finite float for '{context_name}'.") from exc
            if val_float != val_float or val_float in (float("inf"), float("-inf")):
                raise SessionError(f"Expected finite float for '{context_name}'.")
            if self.choices is not None and self.choice_policy == "strict" and val_float not in self.choices:
                raise SessionError(f"Invalid choice '{val_float}' for '{context_name}'. Choices: {self.choices}")
            return val_float

        if self.kind == "string":
            if isinstance(value, Enum):
                value = value.value
            if not isinstance(value, str):
                raise SessionError(f"Expected string for '{context_name}', got {type(value).__name__}.")
            val_str = value
            if self.choices is not None and self.choice_policy == "strict" and val_str not in self.choices:
                raise SessionError(f"Invalid choice '{val_str}' for '{context_name}'. Choices: {self.choices}")
            return val_str

        if self.kind == "mapping":
            if not isinstance(value, (dict, Mapping)):
                raise SessionError(f"Expected mapping/dict for '{context_name}', got {type(value).__name__}.")
            normalized_map: dict[str, Any] = {}
            seen_canonical: dict[str, Any] = {}

            for k, v in value.items():
                raw_k = k.value if isinstance(k, Enum) else k
                validated_k = self.key_schema.validate(raw_k, f"{context_name}.<key>") if self.key_schema else raw_k
                canonical_k = str(validated_k)
                if canonical_k in seen_canonical and seen_canonical[canonical_k] != k:
                    raise SessionError(
                        f"Duplicate normalized key collision in '{context_name}': '{k}' conflicts with '{seen_canonical[canonical_k]}'."
                    )
                seen_canonical[canonical_k] = k

                if self.known_keys is not None and not self.allow_custom_keys:
                    if canonical_k not in self.known_keys:
                        raise SessionError(
                            f"Unknown key '{canonical_k}' in '{context_name}'. Known keys: {self.known_keys}"
                        )

                norm_val = self.value_schema.validate(v, f"{context_name}.{canonical_k}") if self.value_schema else v
                normalized_map[canonical_k] = norm_val

            return normalized_map

        if self.kind == "list":
            if not isinstance(value, (list, tuple)):
                raise SessionError(f"Expected list for '{context_name}', got {type(value).__name__}.")
            if self.value_schema:
                return [self.value_schema.validate(item, f"{context_name}[{i}]") for i, item in enumerate(value)]
            return list(value)

        return value

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "nullable": self.nullable}
        if self.choices is not None:
            d["choices"] = list(self.choices)
            d["choice_policy"] = self.choice_policy
        if self.key_schema is not None:
            d["key_schema"] = self.key_schema.to_dict()
        if self.value_schema is not None:
            d["value_schema"] = self.value_schema.to_dict()
        if self.known_keys is not None:
            d["known_keys"] = list(self.known_keys)
            d["allow_custom_keys"] = self.allow_custom_keys
        if self.known_keys_source is not None:
            d["known_keys_source"] = self.known_keys_source
        return d


@dataclass(frozen=True)
class OptionSpec:
    """Describes one configurable parameter for UI and API consumers."""

    key: str
    schema: ValueSchema
    default: Any
    display_name: str
    description: str
    recommended: Any | None = None
    required: bool = False
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    scope: OptionScope = OptionScope.INFER

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "key": self.key,
            "display_name": self.display_name,
            "description": self.description,
            "schema": self.schema.to_dict(),
            "default": _serialize_value(self.default),
            "recommended": _serialize_value(self.recommended),
            "required": self.required,
            "scope": self.scope.value,
        }
        if self.min_val is not None:
            d["min_val"] = self.min_val
        if self.max_val is not None:
            d["max_val"] = self.max_val
        if self.step is not None:
            d["step"] = self.step
        return d


@dataclass(frozen=True)
class FeatureSpec:
    """Describes a supported configurable capability (preprocessing, inference, or postprocessing)."""

    id: str
    display_name: str
    description: str
    stage: Literal["preprocess", "inference", "postprocess"]
    binding: Literal["result_transform", "plugin"]
    config_type: type
    recommended_config: Any | None = None
    _output_categories: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("FeatureSpec.id must be a non-empty string.")
        if self.stage not in {"preprocess", "inference", "postprocess"}:
            raise ValueError(f"Feature '{self.id}' has invalid stage '{self.stage}'.")
        if self.binding not in {"result_transform", "plugin"}:
            raise ValueError(f"Feature '{self.id}' has invalid binding '{self.binding}'.")
        if self.binding == "result_transform" and self.stage != "postprocess":
            raise ValueError("Result-transform features must use the 'postprocess' stage.")
        if not isinstance(self.config_type, type) or not dataclasses.is_dataclass(self.config_type):
            raise TypeError(f"Feature '{self.id}' config_type must be a dataclass type.")
        if self.recommended_config is not None and not isinstance(self.recommended_config, self.config_type):
            raise TypeError(
                f"Feature '{self.id}' recommended_config must be an instance of '{self.config_type.__name__}'."
            )
        if self.binding == "result_transform":
            from vibe.result_transforms import ResultTransform

            if not issubclass(self.config_type, ResultTransform):
                raise TypeError(f"Feature '{self.id}' marked result_transform but config_type is not a ResultTransform.")

    @classmethod
    def from_transform(
        cls,
        transform_cls: type,
        recommended: Any | None = None,
    ) -> FeatureSpec:
        from vibe.result_transforms import ResultTransform

        if not isinstance(transform_cls, type) or not issubclass(transform_cls, ResultTransform):
            raise TypeError("FeatureSpec.from_transform() requires a ResultTransform subclass.")
        if not dataclasses.is_dataclass(transform_cls):
            raise TypeError(f"Result transform '{transform_cls.__name__}' must be a dataclass.")

        transform_id = getattr(transform_cls, "transform_id", transform_cls.__name__)
        display_name = getattr(transform_cls, "display_name", transform_id)
        description = getattr(transform_cls, "description", "")

        return cls(
            id=transform_id,
            display_name=display_name,
            description=description,
            stage="postprocess",
            binding="result_transform",
            config_type=transform_cls,
            recommended_config=recommended,
        )

    @classmethod
    def from_config(
        cls,
        config_cls: type,
        *,
        stage: Literal["preprocess", "inference", "postprocess"] | None = None,
        binding: Literal["result_transform", "plugin"] = "plugin",
        id: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        recommended: Any | None = None,
    ) -> FeatureSpec:
        if not isinstance(config_cls, type) or not dataclasses.is_dataclass(config_cls):
            config_name = getattr(config_cls, "__name__", type(config_cls).__name__)
            raise TypeError(f"Feature configuration '{config_name}' must be a dataclass.")
        if binding == "result_transform":
            from vibe.result_transforms import ResultTransform

            if not issubclass(config_cls, ResultTransform):
                raise TypeError("A result_transform feature binding requires a ResultTransform subclass.")
        feat_id = id if id is not None else getattr(config_cls, "feature_id", config_cls.__name__.lower())
        disp_name = display_name or getattr(config_cls, "display_name", feat_id.replace("_", " ").title())
        desc = description or getattr(config_cls, "description", "")
        effective_stage = stage or ("postprocess" if binding == "result_transform" else "preprocess")

        return cls(
            id=feat_id,
            display_name=disp_name,
            description=desc,
            stage=effective_stage,
            binding=binding,
            config_type=config_cls,
            recommended_config=recommended,
        )

    @property
    def option_specs(self) -> tuple[OptionSpec, ...]:
        """Dynamically generate option specifications from the dataclass fields of config_type."""
        if not dataclasses.is_dataclass(self.config_type):
            return ()

        specs: list[OptionSpec] = []
        rec_obj = self.recommended_config
        try:
            resolved_hints = get_type_hints(self.config_type)
        except Exception:
            resolved_hints = {}

        for f in dataclasses.fields(self.config_type):
            if f.metadata.get("internal", False) or not f.init:
                continue

            schema = f.metadata.get("schema")
            if schema is None:
                schema = _infer_value_schema(resolved_hints.get(f.name, f.type), f.metadata)
            elif not isinstance(schema, ValueSchema):
                raise TypeError(f"Feature '{self.id}' option '{f.name}' has an invalid ValueSchema metadata value.")
            elif f.metadata.get("choices") is not None and schema.choices is None:
                schema = dataclasses.replace(schema, choices=tuple(f.metadata["choices"]))
            schema = schema.resolve_context(output_categories=self._output_categories)

            if f.default is not dataclasses.MISSING:
                default_val = f.default
            elif f.default_factory is not dataclasses.MISSING:
                default_val = f.default_factory()
            else:
                default_val = None
            recommended_val = getattr(rec_obj, f.name, None) if rec_obj is not None else None

            specs.append(
                OptionSpec(
                    key=f.name,
                    schema=schema,
                    default=default_val,
                    display_name=f.metadata.get("display_name", f.name.replace("_", " ").title()),
                    description=f.metadata.get("description", ""),
                    recommended=recommended_val,
                    required=f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING,
                    min_val=f.metadata.get("min_val"),
                    max_val=f.metadata.get("max_val"),
                    step=f.metadata.get("step"),
                    scope=OptionScope(f.metadata.get("scope", OptionScope.INFER)),
                )
            )

        return tuple(specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "stage": self.stage,
            "binding": self.binding,
            "options": [opt.to_dict() for opt in self.option_specs],
        }


@dataclass(frozen=True)
class InferenceRequest:
    """Immutable, typed per-call configuration bundle passed to preprocessing and runtime."""

    configs: tuple[Any, ...] = ()
    _by_type: dict[type, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        by_type = {type(cfg): cfg for cfg in self.configs}
        if len(by_type) != len(self.configs):
            raise SessionError("InferenceRequest cannot contain multiple configurations of the same type.")
        object.__setattr__(self, "_by_type", by_type)

    def get(self, config_type: type[T]) -> T | None:
        """Retrieve a typed configuration instance by class."""
        return self._by_type.get(config_type)


def transform_meta(
    description: str = "",
    choices: tuple[Any, ...] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    step: float | None = None,
    display_name: str | None = None,
    scope: OptionScope = OptionScope.INFER,
    internal: bool = False,
    *,
    schema: ValueSchema | None = None,
) -> dict[str, Any]:
    """Construct dataclass metadata while preserving the legacy positional arguments."""
    m: dict[str, Any] = {
        "description": description,
        "internal": internal,
        "scope": scope,
    }
    if schema is not None:
        m["schema"] = schema
    if choices is not None:
        m["choices"] = choices
    if min_val is not None:
        m["min_val"] = min_val
    if max_val is not None:
        m["max_val"] = max_val
    if step is not None:
        m["step"] = step
    if display_name is not None:
        m["display_name"] = display_name
    return m


# region Internal Compilation & Serialization Helpers


def _infer_value_schema(type_hint: Any, metadata: Mapping[str, Any]) -> ValueSchema:
    choices = metadata.get("choices")

    # Dataclasses created under ``from __future__ import annotations`` can expose
    # simple annotations as strings. Support the safe scalar cases without trying
    # to interpret arbitrary typing expressions here.
    if isinstance(type_hint, str):
        simple_types = {
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "string": str,
        }
        type_hint = simple_types.get(type_hint, type_hint)

    if type_hint is bool:
        return ValueSchema(kind="bool")
    if type_hint is int:
        return ValueSchema(kind="int", choices=choices)
    if type_hint is float:
        return ValueSchema(kind="float", choices=choices)
    if type_hint is str:
        return ValueSchema(kind="string", choices=choices)

    # Check for simple Optional[...] / ... | None
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is not None and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            base_schema = _infer_value_schema(non_none[0], metadata)
            return dataclasses.replace(base_schema, nullable=True)

    raise SessionError(
        f"Cannot automatically infer ValueSchema for complex type '{type_hint}'. "
        "Declare an explicit `schema=ValueSchema(...)` in transform_meta()."
    )


def _serialize_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialize_value(item) for item in val]
    if isinstance(val, dict):
        return {str(k.value if isinstance(k, Enum) else k): _serialize_value(v) for k, v in val.items()}
    if dataclasses.is_dataclass(val):
        return {f.name: _serialize_value(getattr(val, f.name)) for f in dataclasses.fields(val)}
    return str(val)


def compile_features(
    features_input: Mapping[str, Mapping[str, Any]] | Sequence[Any] | None,
    supported_features: Sequence[FeatureSpec],
    *,
    allowed_scopes: Sequence[OptionScope] = (OptionScope.INFER,),
) -> tuple[InferenceRequest, list[Any]]:
    """
    Compile typed feature instances or mapping requests against supported model FeatureSpecs.

    ``allowed_scopes`` lets lifecycle-specific callers reuse the compiler. The
    inference API defaults to ``OptionScope.INFER``; load/session callers can
    opt into their own scopes without making inference silently accept them.

    Returns:
        (InferenceRequest, list[ResultTransform])
    """
    allowed_scope_set = {OptionScope(scope) for scope in allowed_scopes}

    if not features_input:
        return InferenceRequest(), []

    feature_map: dict[str, FeatureSpec] = {}
    config_type_to_spec: dict[type, FeatureSpec] = {}
    for spec in supported_features:
        if spec.id in feature_map:
            raise SessionError(f"Model declares duplicate feature ID '{spec.id}'.")
        if spec.config_type in config_type_to_spec:
            raise SessionError(
                f"Model declares configuration type '{spec.config_type.__name__}' for multiple features."
            )
        feature_map[spec.id] = spec
        config_type_to_spec[spec.config_type] = spec

    compiled_configs: list[Any] = []
    compiled_transforms: list[Any] = []
    seen_feature_ids: set[str] = set()

    # 1. Sequence of Typed Objects
    if isinstance(features_input, Sequence) and not isinstance(features_input, (str, bytes)):
        for item in features_input:
            cfg_type = type(item)
            if cfg_type not in config_type_to_spec:
                feat_id = getattr(item, "transform_id", getattr(item, "feature_id", cfg_type.__name__))
                raise SessionError(
                    f"Feature '{feat_id}' ({cfg_type.__name__}) is not supported by this model. "
                    f"Supported features: {list(feature_map.keys())}"
                )

            spec = config_type_to_spec[cfg_type]
            _validate_config_instance(item, spec, allowed_scopes=allowed_scope_set, reject_disallowed_scopes=True)
            if spec.id in seen_feature_ids:
                raise SessionError(f"Duplicate feature '{spec.id}' provided in inference request.")
            seen_feature_ids.add(spec.id)

            if spec.binding == "result_transform":
                compiled_transforms.append(item)
            else:
                compiled_configs.append(item)

    # 2. Mapping of feature_id -> option_dict
    elif isinstance(features_input, (dict, Mapping)):
        for feat_id, option_dict in features_input.items():
            if feat_id not in feature_map:
                raise SessionError(
                    f"Unknown feature '{feat_id}' requested. Supported features: {list(feature_map.keys())}"
                )
            if feat_id in seen_feature_ids:
                raise SessionError(f"Duplicate feature '{feat_id}' provided in inference request.")
            seen_feature_ids.add(feat_id)

            spec = feature_map[feat_id]
            if option_dict is None:
                provided_options: dict[str, Any] = {}
            elif isinstance(option_dict, Mapping):
                provided_options = dict(option_dict)
            else:
                raise SessionError(
                    f"Options for feature '{feat_id}' must be a mapping, got {type(option_dict).__name__}."
                )
            option_specs_by_key = {opt.key: opt for opt in spec.option_specs}

            # Check for unknown options (typos)
            for opt_key in provided_options:
                if opt_key not in option_specs_by_key:
                    raise SessionError(
                        f"Unknown option '{opt_key}' for feature '{feat_id}'. "
                        f"Available options: {list(option_specs_by_key.keys())}"
                    )
                if option_specs_by_key[opt_key].scope not in allowed_scope_set:
                    raise SessionError(
                        f"Feature option '{feat_id}.{opt_key}' has scope "
                        f"'{option_specs_by_key[opt_key].scope.value}' and is not available in this request."
                    )

            # Validate each option against ValueSchema and build kwargs
            resolved_kwargs: dict[str, Any] = {}
            for opt_key, opt_spec in option_specs_by_key.items():
                if opt_key in provided_options:
                    raw_val = provided_options[opt_key]
                    validated_val = opt_spec.schema.validate(raw_val, f"{feat_id}.{opt_key}")
                    if validated_val is not None and opt_spec.min_val is not None and validated_val < opt_spec.min_val:
                        raise SessionError(
                            f"Value {validated_val} for '{feat_id}.{opt_key}' is below minimum {opt_spec.min_val}."
                        )
                    if validated_val is not None and opt_spec.max_val is not None and validated_val > opt_spec.max_val:
                        raise SessionError(
                            f"Value {validated_val} for '{feat_id}.{opt_key}' exceeds maximum {opt_spec.max_val}."
                        )
                    resolved_kwargs[opt_key] = validated_val
                elif opt_spec.required:
                    if opt_spec.scope not in allowed_scope_set:
                        raise SessionError(
                            f"Required option '{feat_id}.{opt_key}' has scope "
                            f"'{opt_spec.scope.value}' and is not available in this request."
                        )
                    raise SessionError(f"Required option '{feat_id}.{opt_key}' was not provided.")

            try:
                instantiated = spec.config_type(**resolved_kwargs)
            except Exception as exc:
                raise SessionError(f"Failed to instantiate feature '{feat_id}': {exc}") from exc

            _validate_config_instance(instantiated, spec, allowed_scopes=allowed_scope_set)

            if spec.binding == "result_transform":
                compiled_transforms.append(instantiated)
            else:
                compiled_configs.append(instantiated)
    else:
        raise SessionError(
            f"Invalid features parameter format: expected Sequence of configs or Mapping of dicts, got {type(features_input).__name__}."
        )

    return InferenceRequest(configs=tuple(compiled_configs)), compiled_transforms


def _validate_config_instance(
    config: Any,
    spec: FeatureSpec,
    *,
    allowed_scopes: set[OptionScope] | None = None,
    reject_disallowed_scopes: bool = False,
) -> None:
    """Validate a typed configuration object against its declared feature schema."""
    if not isinstance(config, spec.config_type):
        raise SessionError(
            f"Configuration for feature '{spec.id}' must be an instance of '{spec.config_type.__name__}'."
        )

    if allowed_scopes is None:
        allowed_scopes = {OptionScope.INFER}

    values = {f.name: getattr(config, f.name) for f in dataclasses.fields(spec.config_type)}
    for option in spec.option_specs:
        if option.scope not in allowed_scopes:
            if reject_disallowed_scopes:
                raise SessionError(
                    f"Feature option '{spec.id}.{option.key}' has scope '{option.scope.value}' "
                    "and is not available in this request."
                )
            continue
        value = values.get(option.key)
        validated = option.schema.validate(value, f"{spec.id}.{option.key}")
        if validated is not None and option.min_val is not None and validated < option.min_val:
            raise SessionError(
                f"Value {validated} for '{spec.id}.{option.key}' is below minimum {option.min_val}."
            )
        if validated is not None and option.max_val is not None and validated > option.max_val:
            raise SessionError(
                f"Value {validated} for '{spec.id}.{option.key}' exceeds maximum {option.max_val}."
            )


# endregion
