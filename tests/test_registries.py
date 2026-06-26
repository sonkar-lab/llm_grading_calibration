from __future__ import annotations

from grading_calibration.calibration.isotonic import IsotonicCalibrator
from grading_calibration.calibration.platt import PlattScalingCalibrator
from grading_calibration.calibration.temperature import TemperatureScalingCalibrator
from grading_calibration.confidence.hybrid import HybridConfidence
from grading_calibration.confidence.perturbation_confidence import PerturbationConfidence
from grading_calibration.confidence.self_consistency_confidence import SelfConsistencyConfidence
from grading_calibration.confidence.self_reported import SelfReportedConfidence
from grading_calibration.confidence.token_probability import TokenProbabilityConfidence
from grading_calibration.datasets.beetle_adapter import BeetleAdapter
from grading_calibration.datasets.ricechem_adapter import RiceChemAdapter
from grading_calibration.datasets.scientsbank_adapter import SciEntsBankAdapter
from grading_calibration.calibration.registry import get_calibrator, list_calibrators
from grading_calibration.confidence.registry import (
    get_confidence_estimator,
    list_confidence_estimators,
)
from grading_calibration.datasets.registry import get_dataset_adapter, list_dataset_adapters


def test_dataset_registry_is_populated_with_existing_adapters() -> None:
    assert list_dataset_adapters() == ["beetle", "ricechem", "scientsbank"]
    assert get_dataset_adapter("ricechem") is RiceChemAdapter
    assert get_dataset_adapter("beetle") is BeetleAdapter
    assert get_dataset_adapter("scientsbank") is SciEntsBankAdapter


def test_confidence_registry_is_populated_with_existing_estimators() -> None:
    assert list_confidence_estimators() == [
        "hybrid",
        "perturbation",
        "self_consistency",
        "self_reported",
        "token_probability",
    ]
    assert get_confidence_estimator("self_reported") is SelfReportedConfidence
    assert get_confidence_estimator("token_probability") is TokenProbabilityConfidence
    assert get_confidence_estimator("self_consistency") is SelfConsistencyConfidence
    assert get_confidence_estimator("perturbation") is PerturbationConfidence
    assert get_confidence_estimator("hybrid") is HybridConfidence


def test_calibration_registry_is_populated_with_existing_calibrators() -> None:
    assert list_calibrators() == ["isotonic", "platt", "temperature"]
    assert get_calibrator("platt") is PlattScalingCalibrator
    assert get_calibrator("temperature") is TemperatureScalingCalibrator
    assert get_calibrator("isotonic") is IsotonicCalibrator


def test_calibration_submodules_are_importable_from_package() -> None:
    import grading_calibration.calibration.base as base
    import grading_calibration.calibration.isotonic as isotonic
    import grading_calibration.calibration.platt as platt
    import grading_calibration.calibration.temperature as temperature

    assert hasattr(base, "BaseCalibrator")
    assert isotonic.IsotonicCalibrator is IsotonicCalibrator
    assert platt.PlattScalingCalibrator is PlattScalingCalibrator
    assert temperature.TemperatureScalingCalibrator is TemperatureScalingCalibrator
