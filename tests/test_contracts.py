from __future__ import annotations

import pytest

from vibe.contracts import (
    LabelCatalogProvider,
    ScorerView,
    TaggerView,
    ThresholdProvider,
    verify_plugin_contract,
)
from vibe.exceptions import PluginContractError
from vibe.metadata import (
    CalibrationTable,
    LabelCatalog,
    TagFilterRecommendation,
    ThresholdTable,
)


class _ValidTaggerTarget:
    implements = (LabelCatalogProvider, ThresholdProvider)

    def __init__(self, catalog: LabelCatalog, thresholds: ThresholdTable) -> None:
        self.catalog = catalog
        self.thresholds = thresholds


class _MissingAttrTarget:
    implements = (LabelCatalogProvider, ThresholdProvider)

    def __init__(self, catalog: LabelCatalog) -> None:
        self.catalog = catalog
        # Missing self.thresholds!


class _WrongTypeTarget:
    implements = (LabelCatalogProvider,)

    def __init__(self) -> None:
        self.catalog = "not_a_label_catalog"  # Wrong type!


class _ZombieAttrTarget:
    # Declares only LabelCatalogProvider, but also populates calibration!
    implements = (LabelCatalogProvider,)

    def __init__(self, catalog: LabelCatalog) -> None:
        self.catalog = catalog
        self.calibration = CalibrationTable(x=[0.0, 1.0], y=[0.0, 1.0], source="test")


def test_verify_plugin_contract_passes_valid_target(
    dummy_catalog: LabelCatalog, dummy_thresholds: ThresholdTable
) -> None:
    target = _ValidTaggerTarget(dummy_catalog, dummy_thresholds)
    # Should complete with no exception
    verify_plugin_contract(target)


def test_verify_plugin_contract_fails_missing_attribute(dummy_catalog: LabelCatalog) -> None:
    target = _MissingAttrTarget(dummy_catalog)
    with pytest.raises(PluginContractError, match="missing 'thresholds'"):
        verify_plugin_contract(target)


def test_verify_plugin_contract_fails_wrong_type() -> None:
    target = _WrongTypeTarget()
    with pytest.raises(PluginContractError, match="'catalog' has type str, expected LabelCatalog"):
        verify_plugin_contract(target)


def test_verify_plugin_contract_fails_zombie_attribute(dummy_catalog: LabelCatalog) -> None:
    target = _ZombieAttrTarget(dummy_catalog)
    with pytest.raises(PluginContractError, match="satisfies protocols \\['CalibrationProvider'\\]"):
        verify_plugin_contract(target)


def test_tagger_view_construction_and_access(dummy_catalog: LabelCatalog, dummy_thresholds: ThresholdTable) -> None:
    rec = TagFilterRecommendation(global_threshold=0.35)
    view = TaggerView(
        catalog=dummy_catalog,
        thresholds=dummy_thresholds,
        recommendation=rec,
    )

    assert view.catalog is dummy_catalog
    assert view.thresholds is dummy_thresholds
    assert view.recommendation is rec
    assert view.catalog.names == ("1girl", "solo", "hatsune_miku")


def test_tagger_view_rejects_invalid_types(dummy_thresholds: ThresholdTable) -> None:
    with pytest.raises(AssertionError, match="TaggerView constructed with invalid catalog type"):
        TaggerView(
            catalog="invalid",  # type: ignore[arg-type]
            thresholds=dummy_thresholds,
            recommendation=None,
        )


def test_scorer_view_construction_and_access(dummy_catalog: LabelCatalog) -> None:
    calib = CalibrationTable(x=[0.1, 0.9], y=[0.0, 1.0], source="test")
    view = ScorerView(catalog=dummy_catalog, calibration=calib)

    assert view.catalog is dummy_catalog
    assert view.calibration is calib
    assert view.calibration.to_dict()["source"] == "test"
