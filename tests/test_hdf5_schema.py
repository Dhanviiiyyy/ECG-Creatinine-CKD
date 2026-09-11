"""
tests/test_hdf5_schema.py
===========================
Unit tests for HDF5 schema integrity (HDF5Writer / HDF5Reader).

Tests
-----
* Written records have correct waveform shape [12, 5000].
* Label is stored as float32.
* All required attributes are present.
* get_all_labels returns correct count.
* split_sizes returns correct counts.
* No subject overlap in the fixture HDF5.
"""

from __future__ import annotations

import json
from typing import cast
import numpy as np
import pytest
import h5py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_creatinine.hdf5_utils import HDF5Reader, HDF5Writer


class TestHDF5Writer:
    def test_writer_creates_file(self, tmp_path, cfg):
        hdf5_path = tmp_path / "test_write.h5"
        wf = np.random.default_rng(0).standard_normal((12, 5000)).astype(np.float32)
        with HDF5Writer(hdf5_path, cfg) as w:
            w.write_record(
                split="train",
                study_id="s000001",
                subject_id=1,
                waveform=wf,
                waveform_raw=wf,
                label=1.2,
                meta={
                    "ecg_time": "2150-01-01", "lab_time": "2150-01-01",
                    "delta_t_hours": 2.0, "lead_names": [],
                    "fs": 500.0, "pipeline_version": "0.1.0",
                    "seed": 42, "git_hash": "abc123",
                },
            )
            w.write_metadata({"train": {}}, "")
        assert hdf5_path.exists()

    def test_schema_version_attribute(self, tmp_path, cfg):
        hdf5_path = tmp_path / "test_schema.h5"
        wf = np.zeros((12, 5000), dtype=np.float32)
        with HDF5Writer(hdf5_path, cfg) as w:
            w.write_record("train", "s1", 1, wf, wf, 1.0, {})
            w.write_metadata({}, "")
        with h5py.File(hdf5_path, "r") as fh:
            assert "schema_version" in fh.attrs

    def test_waveform_shape_in_hdf5(self, tmp_path, cfg):
        hdf5_path = tmp_path / "test_shape.h5"
        wf = np.random.default_rng(1).standard_normal((12, 5000)).astype(np.float32)
        with HDF5Writer(hdf5_path, cfg) as w:
            w.write_record("train", "s1", 1, wf, wf, 0.9, {})
            w.write_metadata({}, "")
        with h5py.File(hdf5_path, "r") as fh:
            stored = cast(h5py.Dataset, fh["train/s1/waveform"])[:]
            assert stored.shape == (12, 5000)

    def test_label_stored_correctly(self, tmp_path, cfg):
        hdf5_path = tmp_path / "test_label.h5"
        wf = np.zeros((12, 5000), dtype=np.float32)
        expected_label = 3.7
        with HDF5Writer(hdf5_path, cfg) as w:
            w.write_record("train", "s1", 1, wf, wf, expected_label, {})
            w.write_metadata({}, "")
        with h5py.File(hdf5_path, "r") as fh:
            stored_label = float(cast(h5py.Dataset, fh["train/s1/label"])[()])
        assert abs(stored_label - expected_label) < 1e-4

    def test_all_splits_created(self, tmp_path, cfg):
        hdf5_path = tmp_path / "test_splits.h5"
        wf = np.zeros((12, 5000), dtype=np.float32)
        with HDF5Writer(hdf5_path, cfg) as w:
            w.write_record("train", "s1", 1, wf, wf, 1.0, {})
            w.write_record("val",   "s2", 2, wf, wf, 2.0, {})
            w.write_record("test",  "s3", 3, wf, wf, 3.0, {})
            w.write_metadata({}, "")
        with h5py.File(hdf5_path, "r") as fh:
            assert "train" in fh
            assert "val" in fh
            assert "test" in fh


class TestHDF5Reader:
    def test_get_study_ids(self, tmp_hdf5):
        reader = HDF5Reader(tmp_hdf5)
        ids = reader.get_study_ids("train")
        assert len(ids) > 0
        assert isinstance(ids[0], str)

    def test_get_record_returns_correct_shape(self, tmp_hdf5):
        reader = HDF5Reader(tmp_hdf5)
        ids = reader.get_study_ids("train")
        wf, label, meta = reader.get_record("train", ids[0])
        assert wf.shape == (12, 5000)
        assert wf.dtype == np.float32
        assert isinstance(label, float)
        assert label > 0

    def test_get_all_labels_length(self, tmp_hdf5):
        reader = HDF5Reader(tmp_hdf5)
        ids = reader.get_study_ids("train")
        labels = reader.get_all_labels("train")
        assert len(labels) == len(ids)

    def test_split_sizes_sum_to_total(self, tmp_hdf5):
        reader = HDF5Reader(tmp_hdf5)
        sizes = reader.split_sizes()
        total = sum(sizes.values())
        assert total == 100  # dummy_matched_df has 100 records

    def test_no_subject_overlap_in_fixture(self, tmp_hdf5):
        """No subject_id should appear in more than one split (fixture check)."""
        reader = HDF5Reader(tmp_hdf5)
        subject_sets = {}
        with h5py.File(reader.path, "r") as fh:
            for split in ["train", "val", "test"]:
                sids: set[int] = set()
                split_grp = fh[split]
                if isinstance(split_grp, h5py.Group):
                    for sid_key in split_grp.keys():
                        item = split_grp[sid_key]
                        if isinstance(item, (h5py.Group, h5py.Dataset)):
                            sids.add(int(np.asarray(item.attrs["subject_id"]).item()))
                subject_sets[split] = sids

        train_val = subject_sets["train"] & subject_sets["val"]
        train_test = subject_sets["train"] & subject_sets["test"]
        val_test   = subject_sets["val"]   & subject_sets["test"]
        assert len(train_val)  == 0, f"Train-Val subject overlap: {train_val}"
        assert len(train_test) == 0, f"Train-Test subject overlap: {train_test}"
        assert len(val_test)   == 0, f"Val-Test subject overlap: {val_test}"

    def test_metadata_readable(self, tmp_hdf5):
        reader = HDF5Reader(tmp_hdf5)
        meta = reader.get_metadata()
        assert "cohort_stats" in meta
        assert "config_snapshot" in meta

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HDF5Reader(tmp_path / "nonexistent.h5")
