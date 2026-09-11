"""
config/paths.py
===============
Centralised path resolution for the ECG-Creatinine prediction pipeline.

All pipeline scripts import from this module so that no absolute paths are
hardcoded anywhere else.  The project root is resolved in priority order:

  1. Environment variable  ECG_PROJECT_ROOT  (set this in Colab / CI)
  2. The parent directory of this file (works for local development)

Usage
-----
    from config.paths import Paths, load_config

    cfg  = load_config()
    p    = Paths(cfg)
    print(p.hdf5_file)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    """Resolve project root from env var or file location."""
    env_root = os.environ.get("ECG_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"ECG_PROJECT_ROOT='{env_root}' does not point to a directory."
            )
        return root
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the pipeline YAML configuration.

    Parameters
    ----------
    config_path:
        Explicit path to pipeline_config.yaml.  If None, resolves
        automatically to <project_root>/config/pipeline_config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = _find_project_root() / "config" / "pipeline_config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
        return dict(loaded) if isinstance(loaded, dict) else {}


class Paths:
    """Resolve and expose all project I/O paths.

    Parameters
    ----------
    cfg:
        Parsed pipeline configuration (from load_config()).
    root:
        Override project root (useful for tests).
    """

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        root: str | Path | None = None,
    ) -> None:
        self._root = Path(root).resolve() if root else _find_project_root()
        self._cfg = cfg or load_config()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def data_raw(self) -> Path:
        return self._root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self._root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self._root / "data" / "processed"

    @property
    def synthetic_dir(self) -> Path:
        return self._root / "data" / "synthetic"

    @property
    def reports_dir(self) -> Path:
        return self._root / "reports"

    @property
    def synthetic_ecg_manifest(self) -> Path:
        return self.synthetic_dir / "ecg_manifest.parquet"

    @property
    def synthetic_lab_manifest(self) -> Path:
        return self.synthetic_dir / "lab_creatinine.parquet"

    @property
    def synthetic_waveforms_dir(self) -> Path:
        return self.synthetic_dir / "waveforms"

    @property
    def cohort_matched(self) -> Path:
        return self.data_interim / "cohort_matched.parquet"

    @property
    def hdf5_file(self) -> Path:
        return self.data_processed / "ecg_creatinine.h5"

    @property
    def qc_report(self) -> Path:
        return self.reports_dir / "dataset_qc_report.html"

    def make_all_dirs(self) -> None:
        """Create all project directories if they do not already exist."""
        dirs = [
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.synthetic_dir,
            self.synthetic_waveforms_dir,
            self.reports_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"Paths(\n"
            f"  root           = {self.root}\n"
            f"  hdf5_file      = {self.hdf5_file}\n"
            f")"
        )
