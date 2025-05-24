import torch, torch.nn.functional as F

def brier_score(logits: torch.Tensor, labels: torch.Tensor):
    prob = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=prob.size(1)).float()
    return ((prob - one_hot) ** 2).sum(1).mean().item()