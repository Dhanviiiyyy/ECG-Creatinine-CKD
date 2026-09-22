"""
datasets/ecg_dataset.py
========================
PyTorch Dataset and DataLoader builder for the HDF5 ECG-creatinine dataset.
Supports both lazy on-demand reading and full-RAM preloading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ecg_creatinine.hdf5_utils import HDF5Reader


class ECGCreatinineDataset(Dataset):
    """
    PyTorch Dataset wrapping the ECG-creatinine HDF5 dataset.

    Parameters
    ----------
    hdf5_path : str | Path
        Path to the dataset.h5 file.
    split : str
        'train' | 'val' | 'test'.
    transform : Optional[Callable]
        Optional callable applied to the waveform tensor.
    preload : bool
        If True, loads all waveforms and metadata into RAM at initialization.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        split: str = "train",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        preload: bool = False,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.hdf5_path}")

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'.")

        self.split = split
        self.transform = transform
        self.preload = preload

        # Read study IDs using HDF5Reader
        reader = HDF5Reader(self.hdf5_path)
        self.study_ids = reader.get_study_ids(split)

        self._preloaded_data: Optional[list[dict[str, Any]]] = None
        if self.preload:
            self._preload_all()

    def _preload_all(self) -> None:
        """Preload all records for this split into memory."""
        reader = HDF5Reader(self.hdf5_path)
        self._preloaded_data = []
        for sid in self.study_ids:
            wf, label, meta = reader.get_record(self.split, sid)
            self._preloaded_data.append({
                "waveform": torch.tensor(wf, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.float32),
                "age_years": int(meta.get("age_years", -1)),
                "sex": int(meta.get("sex", -1)),
                "subject_id": int(meta.get("subject_id", -1)),
                "study_id": str(sid),
                "delta_t_h": float(meta.get("delta_t_hours", 0.0)),
            })

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._preloaded_data is not None:
            item = self._preloaded_data[idx]
            wf = item["waveform"]
            if self.transform is not None:
                wf = self.transform(wf)
            return {
                "waveform": wf,
                "label": item["label"],
                "age_years": item["age_years"],
                "sex": item["sex"],
                "subject_id": item["subject_id"],
                "study_id": item["study_id"],
                "delta_t_h": item["delta_t_h"],
            }

        # Lazy reading
        sid = self.study_ids[idx]
        with h5py.File(self.hdf5_path, "r") as fh:
            grp = fh[f"{self.split}/{sid}"]
            wf = grp["waveform"][:].astype(np.float32)
            label = float(grp["label"][()])
            meta = dict(grp.attrs)

        wf_tensor = torch.tensor(wf, dtype=torch.float32)
        if self.transform is not None:
            wf_tensor = self.transform(wf_tensor)

        return {
            "waveform": wf_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "age_years": int(meta.get("age_years", -1)),
            "sex": int(meta.get("sex", -1)),
            "subject_id": int(meta.get("subject_id", -1)),
            "study_id": str(sid),
            "delta_t_h": float(meta.get("delta_t_hours", 0.0)),
        }


def build_dataloaders(
    hdf5_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    preload: bool = False,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> dict[str, DataLoader]:
    """
    Constructs train, validation, and test PyTorch DataLoaders.

    Returns
    -------
    dict[str, DataLoader]
        {"train": train_loader, "val": val_loader, "test": test_loader}
    """
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    train_ds = ECGCreatinineDataset(hdf5_path, split="train", preload=preload)
    val_ds   = ECGCreatinineDataset(hdf5_path, split="val", preload=preload)
    test_ds  = ECGCreatinineDataset(hdf5_path, split="test", preload=preload)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


__all__ = ["ECGCreatinineDataset", "build_dataloaders"]
