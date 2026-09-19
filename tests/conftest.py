from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from vibe.backends.base import (
    ArtifactMap,
    ArtifactSpec,
    Backend,
    ExecutionPlan,
    FileRole,
    ModelIdentity,
    ModelPlugin,
    ModelVariant,
    RuntimeExecutor,
)
from vibe.contracts import LabelCatalogProvider, ThresholdProvider
from vibe.metadata import (
    LabelCatalog,
    LabelInfo,
    TagFilterRecommendation,
    ThresholdTable,
)
from vibe.model_profiles import build_tagger_profile
from vibe.results import TagResult
from vibe.settings import InferenceRequest, OptionScope, SettingGroupSpec


class DummyTestSettings(BaseModel):
    """Test configuration model for settings validation tests."""

    alpha: int = Field(default=10, ge=1, le=100)
    beta: float = Field(default=0.5, ge=0.0, le=1.0)


class DummyRuntime(RuntimeExecutor):
    def run(self, inputs: Any) -> Any:
        return inputs

    def close(self) -> None:
        pass

    def supports_true_batching(self) -> bool:
        return True

    def execution_info(self) -> dict[str, Any]:
        return {"backend": "dummy"}


class DummyTaggerPlugin(ModelPlugin):
    family_name = "Testing Taggers"
    identity = ModelIdentity(
        model_id="test-dummy-tagger",
        display_name="Test Dummy Tagger",
        description="Plugin used strictly for unit testing.",
    )
    default_repo_id = "testing/dummy-tagger"
    profile = build_tagger_profile(
        categories=("general", "character"),
        recommended_filter=TagFilterRecommendation(global_threshold=0.35),
    )
    implements = (LabelCatalogProvider, ThresholdProvider)

    settings = (
        SettingGroupSpec.from_model(
            DummyTestSettings,
            id="test_settings",
            scope=OptionScope.INFER,
        ),
    )

    variants = (
        ModelVariant(
            backend=Backend.ONNX,
            artifacts=(
                ArtifactSpec(
                    id="dummy_weights",
                    name="model.onnx",
                    role=FileRole.WEIGHTS,
                ),
            ),
        ),
    )

    catalog: LabelCatalog
    thresholds: ThresholdTable

    def load_ancillary(self, artifacts: ArtifactMap) -> None:
        del artifacts
        self.catalog = LabelCatalog(
            labels=(
                LabelInfo(index=0, name="1girl", category="general"),
                LabelInfo(index=1, name="solo", category="general"),
                LabelInfo(index=2, name="hatsune_miku", category="character"),
            ),
            categories=("general", "character"),
        )
        self.thresholds = ThresholdTable(
            values={"1girl": 0.4, "solo": 0.5, "hatsune_miku": 0.7},
            source="test",
        )

    def build_runtime(self, artifacts: ArtifactMap, plan: ExecutionPlan) -> RuntimeExecutor:
        del artifacts, plan
        return DummyRuntime()

    def preprocess(self, image: Any, request: InferenceRequest | None = None) -> Any:
        del request
        return image

    def postprocess(self, raw_output: Any) -> TagResult:
        del raw_output
        return TagResult()


@pytest.fixture
def dummy_catalog() -> LabelCatalog:
    return LabelCatalog(
        labels=(
            LabelInfo(index=0, name="1girl", category="general"),
            LabelInfo(index=1, name="solo", category="general"),
            LabelInfo(index=2, name="hatsune_miku", category="character"),
        ),
        categories=("general", "character"),
    )


@pytest.fixture
def dummy_thresholds() -> ThresholdTable:
    return ThresholdTable(
        values={"1girl": 0.4, "solo": 0.5, "hatsune_miku": 0.7},
        source="unit_test",
    )
