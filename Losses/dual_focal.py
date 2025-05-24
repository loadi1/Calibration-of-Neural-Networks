import torch
import torch.nn as nn
import torch.nn.functional as F

class DualFocalLoss(nn.Module):
    """Dual Focal Loss (Tao et al., ICML 2023). Учитывает второй по величине класс.
    Args:
        gamma (float): параметр γ.
        reduction (str): как в Focal
    """
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # prob GT
        # стадия получения второго класса prob_j
        sorted_p, _ = p.sort(dim=1, descending=True)
        p_second = torch.where(sorted_p[:, 0].eq(pt), sorted_p[:, 1], sorted_p[:, 0])
        weight = (1 - pt + p_second) ** self.gamma
        loss = -weight * logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss