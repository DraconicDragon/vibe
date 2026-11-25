"""Base class, registry, and protocol for all result transforms."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from vibe.metadata import ModelDescriptor
    from vibe.results import ModelResult
    from vibe.session import ModelSession

T = TypeVar("T", bound="ModelResult")

_REGISTERED_TRANSFORMS: list[type[ResultTransform[Any]]] = []


class ResultTransform(ABC, Generic[T]):
    """
    Abstract base class for all result transforms.

    Transforms are pure callables: transform(result) -> result.
    They auto-register on definition for dynamic UI discovery.
    """

    id: ClassVar[str]
    display_name: ClassVar[str]
    description: ClassVar[str]
    config_model: ClassVar[type[BaseModel] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and hasattr(cls, "id") and cls not in _REGISTERED_TRANSFORMS:
            _REGISTERED_TRANSFORMS.append(cls)

    @classmethod
    def get_schema(cls) -> dict[str, Any] | None:
        """Return standard OpenAPI/JSON Schema for this transform's settings."""
        return cls.config_model.model_json_schema() if cls.config_model else None

    @classmethod
    def is_supported(cls, target: ModelDescriptor | ModelSession) -> bool:
        """Check whether this transform is compatible with a model descriptor or active session."""
        return True

    @abstractmethod
    def __call__(self, result: T) -> T:
        """Apply the transformation to a single model result."""
        ...


def list_transforms() -> list[type[ResultTransform[Any]]]:
    """Return all available result transform classes for UI discovery."""
    return list(_REGISTERED_TRANSFORMS)