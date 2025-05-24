import torch
import torch.nn as nn
import torch.nn.functional as F

class BSCELossGra(nn.Module):
    """BSCE-GRA (Brier-Score–weighted CE с gradient detach).
    Веса w = BS(sample) (detach), градиент масштабируется.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def _brier(probs, targets):
        one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
        return ((probs - one_hot) ** 2).sum(1)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        w = self._brier(probs, targets).detach()  # без градиента
        ce = F.cross_entropy(logits, targets, reduction='none')
        loss = w * ce  # масштабируем значение CE, но в гр. потоке w не участвует
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss