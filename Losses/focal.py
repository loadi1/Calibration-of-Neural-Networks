import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Классический Focal Loss для многоклассового CE.
    Args:
        gamma (float): сила фокуса (γ). 0 → обычная CE.
        reduction (str): 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)  # probs
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)  # log p_t
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss