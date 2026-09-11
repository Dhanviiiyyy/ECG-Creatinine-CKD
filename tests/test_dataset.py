"""
tests/test_dataset.py
======================
Unit tests for datasets/ecg_dataset.py.
Uses the tmp_hdf5 fixture from conftest.py (100 synthetic records).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.ecg_dataset import ECGCreatinineDataset, build_dataloaders


class TestECGCreatinineDataset:
    def test_len_train(self, tmp_hdf5):
        ds = ECGCreatinineDataset(tmp_hdf5, split="train")
        assert len(ds) > 0

    def test_getitem_keys(self, tmp_hdf5):
        ds   = ECGCreatinineDataset(tmp_hdf5, split="train")
        item = ds[0]
        assert "waveform"   in item
        assert "label"      in item
        assert "age_years"  in item
        assert "sex"        in item
        assert "subject_id" in item
        assert "study_id"   in item
        assert "delta_t_h"  in item

    def test_waveform_shape(self, tmp_hdf5):
        ds   = ECGCreatinineDataset(tmp_hdf5, split="train")
        item = ds[0]
        assert item["waveform"].shape == (12, 5000)

    def test_waveform_dtype(self, tmp_hdf5):
        ds   = ECGCreatinineDataset(tmp_hdf5, split="train")
        item = ds[0]
        assert item["waveform"].dtype == torch.float32

    def test_label_positive(self, tmp_hdf5):
        ds = ECGCreatinineDataset(tmp_hdf5, split="train")
        for i in range(min(5, len(ds))):
            assert ds[i]["label"].item() > 0, "Creatinine must be positive"

    def test_label_dtype(self, tmp_hdf5):
        ds   = ECGCreatinineDataset(tmp_hdf5, split="train")
        item = ds[0]
        assert item["label"].dtype == torch.float32

    def test_split_val(self, tmp_hdf5):
        ds = ECGCreatinineDataset(tmp_hdf5, split="val")
        assert len(ds) > 0

    def test_split_test(self, tmp_hdf5):
        ds = ECGCreatinineDataset(tmp_hdf5, split="test")
        assert len(ds) > 0

    def test_invalid_split_raises(self, tmp_hdf5):
        with pytest.raises(ValueError, match="split must be"):
            ECGCreatinineDataset(tmp_hdf5, split="holdout")

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ECGCreatinineDataset(tmp_path / "nonexistent.h5", split="train")

    def test_transform_applied(self, tmp_hdf5):
        # Transform that adds a constant -- verifiable
        def add_one(wf): return wf + 1.0
        ds_raw   = ECGCreatinineDataset(tmp_hdf5, split="train")
        ds_trans = ECGCreatinineDataset(tmp_hdf5, split="train", transform=add_one)
        wf_raw   = ds_raw[0]["waveform"]
        wf_trans = ds_trans[0]["waveform"]
        assert torch.allclose(wf_trans, wf_raw + 1.0)

    def test_preload_same_as_lazy(self, tmp_hdf5):
        ds_lazy   = ECGCreatinineDataset(tmp_hdf5, split="train", preload=False)
        ds_preload= ECGCreatinineDataset(tmp_hdf5, split="train", preload=True)
        wf_lazy   = ds_lazy[0]["waveform"].numpy()
        wf_pre    = ds_preload[0]["waveform"].numpy()
        np.testing.assert_array_almost_equal(wf_lazy, wf_pre, decimal=5)


class TestBuildDataloaders:
    def test_returns_all_splits(self, tmp_hdf5):
        loaders = build_dataloaders(tmp_hdf5, batch_size=8)
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_batch_shape(self, tmp_hdf5):
        loaders = build_dataloaders(tmp_hdf5, batch_size=8)
        batch   = next(iter(loaders["train"]))
        # last batch may be smaller if < 8 records in train split
        assert batch["waveform"].shape[1:] == (12, 5000)
        assert batch["label"].ndim == 1

    def test_train_shuffled_over_epochs(self, tmp_hdf5):
        """Two epochs should yield different ordering (with high probability)."""
        loaders = build_dataloaders(tmp_hdf5, batch_size=4)
        ids_e1  = [b["study_id"] for b in loaders["train"]]
        ids_e2  = [b["study_id"] for b in loaders["train"]]
        # With >= 8 records and shuffle=True, orderings almost certainly differ
        # (probability of same order = 1/n! -- effectively zero for n>=8)
        if len(ids_e1) > 1:
            assert ids_e1 != ids_e2 or len(ids_e1) == 1, \
                "Train loader should shuffle between epochs"
