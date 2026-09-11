"""
ecg_creatinine/hdf5_utils.py
==============================
HDF5 read/write utilities for the ECG-creatinine dataset.

HDF5 Schema (v1.0.0)
---------------------
/train/
    /<study_id>/
        waveform        float32 [12, 5000]  z-scored, filtered
        waveform_raw    float32 [12, 5000]  filtered, NOT z-scored (mV)
        label           scalar  float32     serum creatinine (mg/dL)
        attrs:
            subject_id, study_id, ecg_time, lab_time,
            delta_t_hours, lead_names, fs, pipeline_version, seed
/val/   (same)
/test/  (same)
/metadata/
    cohort_stats        JSON string (N per split, creatinine stats)
    config_snapshot     YAML string (full pipeline_config.yaml)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class HDF5Writer:
    """Context-manager wrapper for writing the ECG-creatinine HDF5 dataset.

    Usage
    -----
        with HDF5Writer(path, cfg, compression="gzip", level=4) as writer:
            writer.write_record(
                split="train",
                study_id="p10000032_s10000000",
                subject_id=10000032,
                waveform=arr_norm,        # [12, 5000] float32
                waveform_raw=arr_raw,     # [12, 5000] float32
                label=1.2,               # creatinine mg/dL
                meta={...},
            )
            writer.write_metadata(cohort_stats, config_yaml_str)
    """

    def __init__(
        self,
        path: str | Path,
        cfg: dict[str, Any],
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.compression = compression
        self.compression_opts = compression_opts
        self._fh: h5py.File | None = None

    def __enter__(self) -> "HDF5Writer":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = h5py.File(self.path, "w")
        # Root metadata
        self._fh.attrs["schema_version"] = SCHEMA_VERSION
        self._fh.attrs["pipeline"] = "ECG-Creatinine IIT Hyderabad"
        # Pre-create split groups
        for split in ("train", "val", "test"):
            self._fh.require_group(split)
        self._fh.require_group("metadata")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write_record(
        self,
        split: str,
        study_id: str,
        subject_id: int,
        waveform: np.ndarray,
        waveform_raw: np.ndarray,
        label: float,
        meta: dict[str, Any],
    ) -> None:
        """Write one (ECG, creatinine) pair to the HDF5 file.

        Parameters
        ----------
        split:        'train' | 'val' | 'test'
        study_id:     Unique ECG study identifier string.
        subject_id:   MIMIC-IV subject_id (int).
        waveform:     [12, 5000] float32 normalised waveform.
        waveform_raw: [12, 5000] float32 filtered (unnormalised) waveform.
        label:        Serum creatinine in mg/dL (float32).
        meta:         Dict of additional metadata (ecg_time, lab_time, etc.).
        """
        if self._fh is None:
            raise RuntimeError("HDF5Writer must be used as a context manager.")
        split_grp = self._fh.require_group(split)
        grp = split_grp.require_group(str(study_id))

        # Waveforms
        grp.create_dataset(
            "waveform",
            data=waveform.astype(np.float32),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        grp.create_dataset(
            "waveform_raw",
            data=waveform_raw.astype(np.float32),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        # Label
        grp.create_dataset("label", data=np.float32(label))

        # Attributes
        grp.attrs["subject_id"]      = int(subject_id)
        grp.attrs["study_id"]        = str(study_id)
        grp.attrs["ecg_time"]        = str(meta.get("ecg_time", ""))
        grp.attrs["lab_time"]        = str(meta.get("lab_time", ""))
        grp.attrs["delta_t_hours"]   = float(meta.get("delta_t_hours", 0.0))
        grp.attrs["lead_names"]      = json.dumps(meta.get("lead_names", []))
        grp.attrs["fs"]              = float(meta.get("fs", 500.0))
        grp.attrs["pipeline_version"]= str(meta.get("pipeline_version", "0.1.0"))
        grp.attrs["seed"]            = int(meta.get("seed", 42))
        grp.attrs["git_hash"]        = str(meta.get("git_hash", "unknown"))
        # Demographics for eGFR computation (Inker et al., 2021 CKD-EPI)
        grp.attrs["age_years"]       = int(meta.get("age_years", -1))
        grp.attrs["sex"]             = int(meta.get("sex", -1))  # 0=female, 1=male

    def write_metadata(
        self,
        cohort_stats: dict[str, Any],
        config_yaml_str: str,
    ) -> None:
        """Write global metadata to /metadata/ group."""
        if self._fh is None:
            raise RuntimeError("HDF5Writer must be used as a context manager.")
        meta_grp = self._fh.require_group("metadata")
        meta_grp.create_dataset(
            "cohort_stats",
            data=json.dumps(cohort_stats, default=str),
        )
        meta_grp.create_dataset("config_snapshot", data=config_yaml_str)


# ---------------------------------------------------------------------------
# Reader (PyTorch-friendly)
# ---------------------------------------------------------------------------

class HDF5Reader:
    """Read ECG-creatinine records from the HDF5 dataset.

    Usage
    -----
        reader = HDF5Reader(path)
        waveform, label, meta = reader.get_record("train", study_id)
        all_ids = reader.get_study_ids("train")
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")

    def get_study_ids(self, split: str) -> list[str]:
        """Return all study IDs for a given split."""
        with h5py.File(self.path, "r") as fh:
            grp = fh.get(split)
            if isinstance(grp, h5py.Group):
                return list(grp.keys())
            return []

    def get_record(
        self,
        split: str,
        study_id: str,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Load one record.

        Returns
        -------
        waveform : np.ndarray [12, 5000] float32
        label    : float (creatinine mg/dL)
        meta     : dict
        """
        with h5py.File(self.path, "r") as fh:
            grp = fh.get(f"{split}/{study_id}")
            if not isinstance(grp, h5py.Group):
                raise KeyError(f"Record {split}/{study_id} not found in HDF5.")
            wf_ds = grp.get("waveform")
            assert isinstance(wf_ds, h5py.Dataset)
            waveform = wf_ds[:].astype(np.float32)

            lbl_ds = grp.get("label")
            assert isinstance(lbl_ds, h5py.Dataset)
            label = float(lbl_ds[()])
            meta = dict(grp.attrs)
        return waveform, label, meta

    def get_all_labels(self, split: str) -> np.ndarray:
        """Efficiently load all creatinine labels for a split (no waveforms)."""
        labels = []
        with h5py.File(self.path, "r") as fh:
            grp = fh.get(split)
            if isinstance(grp, h5py.Group):
                for sid in grp.keys():
                    lbl_item = grp.get(f"{sid}/label")
                    if isinstance(lbl_item, h5py.Dataset):
                        labels.append(float(lbl_item[()]))
        return np.array(labels, dtype=np.float32)

    def get_metadata(self) -> dict[str, Any]:
        """Read global metadata from /metadata/ group."""
        with h5py.File(self.path, "r") as fh:
            stats_ds = fh.get("metadata/cohort_stats")
            assert isinstance(stats_ds, h5py.Dataset)
            stats = json.loads(stats_ds[()])

            cfg_ds = fh.get("metadata/config_snapshot")
            assert isinstance(cfg_ds, h5py.Dataset)
            cfg_str = cfg_ds[()]
            if isinstance(cfg_str, bytes):
                cfg_str = cfg_str.decode()
        return {"cohort_stats": stats, "config_snapshot": yaml.safe_load(cfg_str)}

    def split_sizes(self) -> dict[str, int]:
        """Return number of records in each split."""
        with h5py.File(self.path, "r") as fh:
            sizes: dict[str, int] = {}
            for split in ("train", "val", "test"):
                grp = fh.get(split)
                sizes[split] = len(grp) if isinstance(grp, h5py.Group) else 0
            return sizes
