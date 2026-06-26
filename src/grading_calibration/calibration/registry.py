"""Calibration method registry.

This registry is additive and does not change existing calibration dispatch.
"""

from __future__ import annotations

from .base import BaseCalibrator
from .isotonic import IsotonicCalibrator
from .platt import PlattScalingCalibrator
from .temperature import TemperatureScalingCalibrator


_REGISTRY: dict[str, type[BaseCalibrator]] = {}


def register_calibrator(name: str, cls: type[BaseCalibrator]) -> type[BaseCalibrator]:
    """Register a calibrator class by lowercase name."""
    _REGISTRY[name.lower()] = cls
    return cls


def get_calibrator(name: str) -> type[BaseCalibrator]:
    """Return the calibrator class registered for ``name``."""
    key = name.lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown calibrator: {name}") from exc


def list_calibrators() -> list[str]:
    """Return registered calibrator names."""
    return sorted(_REGISTRY)


register_calibrator("platt", PlattScalingCalibrator)
register_calibrator("temperature", TemperatureScalingCalibrator)
register_calibrator("isotonic", IsotonicCalibrator)
