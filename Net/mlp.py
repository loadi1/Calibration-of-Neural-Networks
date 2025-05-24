"""mlp.py – Baseline multilayer perceptron for the Otto dataset
----------------------------------------------------------------
Provides:
  • `MLPClassifier` – a simple fully‑connected network (93 → 512 → 256 → 128 → 9)
  • Training / evaluation helpers (`train_epoch`, `eval_epoch`, `fit`)
  • CLI interface compatible with `otto.py` data loaders:
        $ python mlp.py --data ./data/otto --epochs 30 --batch-size 256

The script **does not** perform any post‑hoc calibration – logits are returned
raw. This keeps the model clean so that calibration can be added later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
#  Network definition
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """Simple MLP with BatchNorm and Dropout.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector (93 for Otto).
    hidden_dims : Sequence[int]
        Sizes of hidden layers.
    num_classes : int
        Number of output classes (9 for Otto).
    dropout : float
        Dropout probability applied *after* each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 93,
        hidden_dims: Sequence[int] = (512, 256, 128),
        num_classes: int = 9,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.backbone(x)
        logits = self.classifier(x)
        return logits


# ---------------------------------------------------------------------------
#  Training helpers
# ---------------------------------------------------------------------------

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top‑1 accuracy."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy_from_logits(logits, targets) * inputs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy_from_logits(logits, targets) * inputs.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 30,
) -> None:
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
    print(f"Best val acc: {best_val_acc:.4f}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    import Data.otto as otto  # local module with data loaders

    parser = argparse.ArgumentParser(description="Train baseline MLP on Otto")
    parser.add_argument("--data", type=Path, default=Path("./data/otto"), help="Data directory")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader, val_loader = otto.get_train_valid_loader(
        batch_size=args.batch_size,
        random_seed=args.seed,
        valid_size=0.1,
        kaggle_config_dir=None,
        data_dir=args.data,
    )

    model = MLPClassifier().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    fit(model, train_loader, val_loader, optimizer, device=torch.device(args.device), epochs=args.epochs)


if __name__ == "__main__":
    cli()
