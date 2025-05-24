import torch.nn as nn

class OttoMLP(nn.Module):
    """Простой MLP 93→512→256→128→9."""
    def __init__(self, in_features: int = 93, num_classes: int = 9):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)