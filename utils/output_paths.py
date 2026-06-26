"""
Shared output-path utilities for organized saving across steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json


SCHEMA_VERSION = "1.0"


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class OutputConfig:
    output_root: Optional[Path]
    dataset: str
    run_id: str
    save_legacy: bool = False
    write_manifest: bool = False
    schema_version: str = SCHEMA_VERSION


@dataclass
class OutputManager:
    config: OutputConfig
    _entries: list[dict[str, Any]] = field(default_factory=list)

    def enabled(self) -> bool:
        return self.config.output_root is not None

    def root(self) -> Path:
        if self.config.output_root is None:
            raise ValueError("output_root is None; OutputManager is disabled")
        return self.config.output_root / self.config.dataset / self.config.run_id

    def step_root(self, step: str) -> Path:
        root = self.root() / step
        root.mkdir(parents=True, exist_ok=True)
        return root

    def step_dir(self, step: str, kind: str) -> Path:
        out = self.step_root(step) / kind
        out.mkdir(parents=True, exist_ok=True)
        return out

    def register(
        self,
        *,
        step: str,
        artifact: str,
        path: Path,
        model: Optional[str] = None,
        method: Optional[str] = None,
        dataset: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "step": step,
            "artifact": artifact,
            "path": str(path),
            "model": model,
            "method": method,
            "dataset": dataset or self.config.dataset,
            "run_id": self.config.run_id,
        }
        if extra:
            entry.update(extra)
        self._entries.append(entry)

    def write_manifest(self) -> Optional[Path]:
        if not self.enabled() or not self.config.write_manifest:
            return None
        manifest = {
            "schema_version": self.config.schema_version,
            "dataset": self.config.dataset,
            "run_id": self.config.run_id,
            "outputs": self._entries,
        }
        path = self.root() / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return path


def add_output_args(parser) -> None:
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Base output directory for organized saving (optional).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Shared run identifier for organized outputs (default: timestamp).",
    )
    parser.add_argument(
        "--save-legacy",
        action="store_true",
        help="Also write outputs to legacy locations.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write manifest.json listing outputs under the organized root.",
    )


def build_output_manager(
    *,
    output_root: Optional[str],
    dataset: str,
    run_id: Optional[str],
    save_legacy: bool,
    write_manifest: bool,
) -> Optional[OutputManager]:
    if not output_root:
        return None
    root = Path(output_root)
    rid = run_id or default_run_id()
    cfg = OutputConfig(
        output_root=root,
        dataset=dataset,
        run_id=rid,
        save_legacy=save_legacy,
        write_manifest=write_manifest,
    )
    return OutputManager(cfg)


def infer_run_id_from_path(path: Path, output_root: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(output_root.resolve())
    except Exception:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    # outputs/<dataset>/<run_id>/...
    return parts[1]
