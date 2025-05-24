import torch.nn as nn
import torch

class CELossWithLabelSmoothing(nn.Module):
    """Cross‑Entropy с label smoothing (α)."""
    def __init__(self, alpha: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = logits.new_full((targets.size(0), num_classes), self.alpha / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.alpha)
        return self.ce(nn.functional.log_softmax(logits, dim=1), true_dist)