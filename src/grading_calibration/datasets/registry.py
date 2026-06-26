"""Dataset adapter registry.

This is additive infrastructure for the reorganization; existing dispatch paths
continue to work unchanged.
"""

from __future__ import annotations


from .base import BaseDatasetAdapter
from .beetle_adapter import BeetleAdapter
from .ricechem_adapter import RiceChemAdapter
from .scientsbank_adapter import SciEntsBankAdapter


_REGISTRY: dict[str, type[BaseDatasetAdapter]] = {}


def register_dataset_adapter(name: str, cls: type[BaseDatasetAdapter]) -> type[BaseDatasetAdapter]:
    """Register a dataset adapter class by lowercase name."""
    _REGISTRY[name.lower()] = cls
    return cls


def get_dataset_adapter(name: str) -> type[BaseDatasetAdapter]:
    """Return the dataset adapter class registered for ``name``."""
    key = name.lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset adapter: {name}") from exc


def list_dataset_adapters() -> list[str]:
    """Return registered dataset adapter names."""
    return sorted(_REGISTRY)


register_dataset_adapter("ricechem", RiceChemAdapter)
register_dataset_adapter("beetle", BeetleAdapter)
register_dataset_adapter("scientsbank", SciEntsBankAdapter)
