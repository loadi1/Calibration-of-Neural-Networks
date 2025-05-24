import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch


def _load_otto_csv(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"]).values.astype(np.float32)
    #y_str = df["target"]  # строки вида "Class_1"
    # print(y_str)
    # print(y_str.dtype)
    y = df["target"].str.replace("Class_", "").astype(int).sub(1).astype(np.int64).values
    return X, y


def get_otto_loaders(csv_path: str = "./data/otto/train.csv", batch_size: int = 256, val_split: float = 0.1):
    X, y = _load_otto_csv(csv_path)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=val_split, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader