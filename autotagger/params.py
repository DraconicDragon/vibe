"""
Parameter schema system.

Each plugin declares its inference parameters as a list of ParamDef objects.
The schema is used to:
  - validate user-supplied params at runtime (clear errors before inference)
  - auto-generate CLI flags
  - auto-generate API request body docs
  - drive GUI sliders/inputs

Example plugin usage:
    param_schema = ParamSchema([
        ParamDef("general_threshold", float, default=0.35, range=(0.0, 1.0),
                 label="General tag threshold"),
        ParamDef("character_threshold", float, default=0.85, range=(0.0, 1.0),
                 label="Character tag threshold"),
        ParamDef("return_all_scores", bool, default=False,
                 label="Include all tag scores in output"),
    ])
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import get_close_matches
from typing import Any

ParamType = type[float] | type[int] | type[bool] | type[str]


# region ParamSchema


@dataclass
class ParamDef:
    """
    Defines a single inference parameter a plugin accepts.

    Attributes:
        name:       Internal key, used in the params dict and as CLI flag name.
        type:       Python type (float, int, bool, str).
        default:    Value used when not supplied by the user.
        range:      For numeric types: (min, max) inclusive. None = unconstrained.
        choices:    For str/int types: allowed values. None = unconstrained.
        label:      Human-readable display name for GUIs and help text.
        description: Longer explanation, used in CLI --help and API docs.
    """

    name: str
    type: ParamType
    default: Any = None
    range: tuple[float, float] | None = None
    choices: list[Any] | None = None
    label: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name.replace("_", " ").capitalize()

    def validate(self, value: Any) -> Any:
        """
        Validate and coerce a single value against this param definition.
        Returns the coerced value, or raises ValueError with a clear message.
        """
        # Coerce type
        try:
            if self.type is bool and isinstance(value, str):
                # Handle "true"/"false" strings from CLI/API
                coerced = value.lower() in ("1", "true", "yes")
            else:
                coerced = self.type(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Param '{self.name}': expected {self.type.__name__}, got {type(value).__name__} ({value!r})"
            ) from exc

        # Range check for numerics
        if self.range is not None and isinstance(coerced, (int, float)):
            lo, hi = self.range
            if not (lo <= coerced <= hi):
                raise ValueError(f"Param '{self.name}': value {coerced} is outside allowed range [{lo}, {hi}]")

        # Choices check
        if self.choices is not None and coerced not in self.choices:
            raise ValueError(f"Param '{self.name}': value {coerced!r} is not one of {self.choices}")

        return coerced

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.__name__
        d["range"] = list(self.range) if self.range else None
        return d


class ParamSchema:
    """
    The full set of parameters a plugin declares.

    Acts as both a validator and an introspection tool.
    """

    def __init__(self, params: list[ParamDef] | None = None) -> None:
        self._params: list[ParamDef] = params or []
        self._by_name: dict[str, ParamDef] = {p.name: p for p in self._params}

    def __iter__(self):
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __bool__(self) -> bool:
        return bool(self._params)

    def get(self, name: str) -> ParamDef | None:
        return self._by_name.get(name)

    def defaults(self) -> dict[str, Any]:
        """Return a dict of all params set to their default values."""
        return {p.name: p.default for p in self._params}

    def validate(self, user_params: dict[str, Any]) -> dict[str, Any]:
        """
        Validate user-supplied params against this schema.

        - Unknown param names raise ValueError.
        - Missing params are filled from defaults.
        - Each supplied value is type-checked and range-checked.

        Returns a complete, validated params dict.
        """
        unknown = set(user_params) - set(self._by_name)
        if unknown:
            known = sorted(self._by_name.keys())
            suggestions: list[str] = []
            for bad_key in sorted(unknown):
                match = get_close_matches(bad_key, known, n=1, cutoff=0.6)
                if match:
                    suggestions.append(f"{bad_key!r} -> {match[0]!r}")

            suffix = ""
            if suggestions:
                suffix = f" Did you mean: {', '.join(suggestions)}?"
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}. This model accepts: {known}.{suffix}")

        result = self.defaults()
        for name, value in user_params.items():
            param_def = self._by_name[name]
            result[name] = param_def.validate(value)

        return result

    def to_list(self) -> list[dict[str, Any]]:
        """Serialise the schema to a list of dicts (for API docs, GUIs, etc.)."""
        return [p.to_dict() for p in self._params]


# Sentinel: a plugin with no configurable params
EMPTY_SCHEMA = ParamSchema()


# endregion ParamSchema
