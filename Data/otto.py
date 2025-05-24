"""
otto.py – Data loaders for the **Otto Group Product Classification Challenge**
-----------------------------------------------------------------------
Mirrors the API of `cifar100.py` (arXiv:2503.22725) and adds automatic
**Kaggle download** if the CSV files are not found locally. You can point
the download routine to a custom directory containing `kaggle.json` by
passing `kaggle_config_dir=<path>` or by exporting the environment variable
`KAGGLE_CONFIG_DIR`.

Dataset details
---------------
* 9 classes: `Class_1` … `Class_9`
* 93 numerical features per sample (`id` + 93 floats, `target` label)
* Train size ≈ 200 000 rows

Usage
-----
>>> from otto import get_train_valid_loader, get_test_loader
>>> train_loader, val_loader = get_train_valid_loader(
...     batch_size=256,
...     random_seed=42,
...     valid_size=0.1,
...     kaggle_config_dir="./.kaggle")   # <- ваша папка с kaggle.json

>>> test_loader = get_test_loader(batch_size=256)

If `train.csv` and `test.csv` are absent in `data_dir` (default: `./data/otto`),
the script downloads and unpacks them automatically via the Kaggle CLI:
`kaggle competitions download -c otto-group-product-classification-challenge`.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# -----------------------------------------------------------------------------
# Utility: auto‑download dataset
# -----------------------------------------------------------------------------

def ensure_otto_dataset(
    data_dir: Union[str, Path] = "./data/otto",
    kaggle_config_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Ensure that `train.csv` and `test.csv` exist in *data_dir*.

    If missing, download them via Kaggle CLI. If *kaggle_config_dir* is
    provided, it is passed to the subprocess through the `KAGGLE_CONFIG_DIR`
    environment variable so that Kaggle can locate *kaggle.json* outside the
    default `~/.kaggle` directory.
    """
    data_dir = Path(data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    if train_csv.exists() and test_csv.exists():
        return  # already present

    print("[otto] CSV files not found – downloading from Kaggle …")

    data_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if kaggle_config_dir is not None:
        env["KAGGLE_CONFIG_DIR"] = str(Path(kaggle_config_dir).expanduser().resolve())

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "otto-group-product-classification-challenge",
        "-p",
        str(data_dir),
    ]
    subprocess.run(cmd, check=True, env=env)

    zip_path = data_dir / "otto-group-product-classification-challenge.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        zip_path.unlink()
    else:
        raise RuntimeError("Expected zip file not found after Kaggle download.")


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------

class OttoDataset(Dataset):
    """PyTorch Dataset wrapping the Otto CSV.

    The dataset standardises features using *mean* and *std* passed in from the
    outer function so that train/val/test share the same statistics computed
    on the *training split only*.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        label: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        df = pd.read_csv(csv_path)
        if label:
            self.y = (
                df["target"].map(lambda s: int(s.lstrip("Class_")) - 1).values.astype(np.int64)
            )
            self.X = df.drop(["id", "target"], axis=1).values.astype(dtype)
        else:
            self.y = None
            self.X = df.drop(["id"], axis=1).values.astype(dtype)

        if mean is not None and std is not None:
            self.X = (self.X - mean) / std

        self.X = torch.from_numpy(self.X)
        if self.y is not None:
            self.y = torch.from_numpy(self.y)

    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401, ANN101
        return len(self.X)

    # ---------------------------------------------------------------------
    def __getitem__(self, idx):  # noqa: ANN001
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------------
# Data‑loader helpers (API compatible with cifar100.py)
# -----------------------------------------------------------------------------

def _compute_train_statistics(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X = df.drop(["id", "target"], axis=1).values.astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # avoid division by zero
    return mean, std


def get_train_valid_loader(
    batch_size: int,
    random_seed: int,
    valid_size: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = False,
    get_val_temp: float = 0,
    data_dir: Union[str, Path] = "./data/otto",
    kaggle_config_dir: Optional[Union[str, Path]] = None,
):
    """Return *train* and *validation* DataLoader for Otto.

    *Parameters mirror* `cifar100.get_train_valid_loader`, except `augment` is
    omitted (there is no augmentation for tabular data) and additional args for
    dataset location.
    """
    assert 0 <= valid_size <= 1, "valid_size must be in [0, 1]"

    ensure_otto_dataset(data_dir, kaggle_config_dir)
    csv_path = Path(data_dir) / "train.csv"

    # statistics on *train split* only (before we sub‑split)
    mean, std = _compute_train_statistics(csv_path)

    full_dataset = OttoDataset(csv_path, mean=mean, std=std)
    num_train = len(full_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    if get_val_temp > 0:
        temp_split = int(np.floor(get_val_temp * split))
        valid_idx, valid_temp_idx = valid_idx[temp_split:], valid_idx[:temp_split]
        valid_temp_sampler = SubsetRandomSampler(valid_temp_idx)
        valid_temp_loader = DataLoader(
            full_dataset,  # share dataset but use separate sampler
            batch_size=batch_size,
            sampler=valid_temp_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if get_val_temp > 0:
        return train_loader, valid_loader, valid_temp_loader
    return train_loader, valid_loader


def get_test_loader(
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = False,
    data_dir: Union[str, Path] = "./data/otto",
    kaggle_config_dir: Optional[Union[str, Path]] = None,
):
    """Return a DataLoader for `test.csv` (if labels are absent, returns only X)."""
    ensure_otto_dataset(data_dir, kaggle_config_dir)
    csv_path = Path(data_dir) / "test.csv"

    # Re‑use mean/std from training data to standardise test features
    train_csv = Path(data_dir) / "train.csv"
    mean, std = _compute_train_statistics(train_csv)

    dataset = OttoDataset(csv_path, label=False, mean=mean, std=std)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader
