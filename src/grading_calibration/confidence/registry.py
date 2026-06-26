"""Confidence estimator registry.

This module is intentionally additive: it exposes existing estimator classes
without rerouting current call sites.
"""

from __future__ import annotations

from .base import BaseConfidenceEstimator
from .hybrid import HybridConfidence
from .perturbation_confidence import PerturbationConfidence
from .self_consistency_confidence import SelfConsistencyConfidence
from .self_reported import SelfReportedConfidence
from .token_probability import TokenProbabilityConfidence


_REGISTRY: dict[str, type[BaseConfidenceEstimator]] = {}


def register_confidence_estimator(
    name: str, cls: type[BaseConfidenceEstimator]
) -> type[BaseConfidenceEstimator]:
    """Register a confidence estimator class by lowercase name."""
    _REGISTRY[name.lower()] = cls
    return cls


def get_confidence_estimator(name: str) -> type[BaseConfidenceEstimator]:
    """Return the confidence estimator class registered for ``name``."""
    key = name.lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown confidence estimator: {name}") from exc


def list_confidence_estimators() -> list[str]:
    """Return registered confidence estimator names."""
    return sorted(_REGISTRY)


register_confidence_estimator("self_reported", SelfReportedConfidence)
register_confidence_estimator("token_probability", TokenProbabilityConfidence)
register_confidence_estimator("self_consistency", SelfConsistencyConfidence)
register_confidence_estimator("perturbation", PerturbationConfidence)
register_confidence_estimator("hybrid", HybridConfidence)
