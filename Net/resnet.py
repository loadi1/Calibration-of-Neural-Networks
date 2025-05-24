import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Wrapper(nn.Module):
    """ResNet‑50 с гибким числом классов и опцией заморозки слоев."""
    def __init__(self, num_classes: int, pretrained: bool = False, freeze_backbone: bool = False):
        super().__init__()
        self.net = resnet50(weights="DEFAULT" if pretrained else None)
        if freeze_backbone:
            for p in self.net.parameters():
                p.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)