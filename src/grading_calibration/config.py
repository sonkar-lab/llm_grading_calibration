"""Optional YAML config helpers for CLI argparse parsers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parser_destinations(parser: argparse.ArgumentParser, *, include_config: bool = False) -> set[str]:
    """Return argparse destination names accepted by a parser."""
    destinations: set[str] = set()
    for action in parser._actions:
        if action.dest in (argparse.SUPPRESS, "help"):
            continue
        if action.dest == "config" and not include_config:
            continue
        destinations.add(action.dest)
    return destinations


def validate_config_keys(
    config: dict[str, Any],
    parser: argparse.ArgumentParser,
    *,
    source: str | Path = "config",
) -> None:
    """Raise a clear error if a config key is not an argparse destination."""
    allowed = parser_destinations(parser)
    unknown = sorted(set(config) - allowed)
    if unknown:
        preview = ", ".join(sorted(allowed)[:12])
        raise ValueError(
            f"Unknown config key(s) in {source}: {', '.join(unknown)}. "
            f"Use argparse destination names, e.g. {preview}."
        )


def load_config(path: str | Path, parser: argparse.ArgumentParser | None = None) -> dict[str, Any]:
    """Load a flat YAML config dict, optionally validating keys against a parser."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        config: dict[str, Any] = {}
    elif not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    else:
        config = dict(data)

    nested = sorted(k for k, v in config.items() if isinstance(v, dict))
    if nested:
        raise ValueError(
            f"Config file must be a flat mapping of argparse destinations to values; "
            f"nested keys found in {config_path}: {', '.join(nested)}"
        )

    if parser is not None:
        validate_config_keys(config, parser, source=config_path)

    return config


def apply_config_to_parser(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    """Apply YAML values as argparse defaults so explicit CLI flags still win."""
    validate_config_keys(config, parser)
    normalized = dict(config)
    for action in parser._actions:
        if action.dest in normalized and isinstance(action.default, tuple) and isinstance(normalized[action.dest], list):
            normalized[action.dest] = tuple(normalized[action.dest])
    parser.set_defaults(**normalized)
