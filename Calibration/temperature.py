import torch, torch.nn as nn, torch.nn.functional as F

class TemperatureScaler(nn.Module):
    """Учёт температуры (T>0). Подбирается на валидации, затем применяется к логитам."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor):
        # логиты / T
        return logits / self.temperature.clamp(min=1e-6)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50):
        self.cuda(); logits, labels = logits.cuda(), labels.cuda()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def _loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        optimizer.step(_loss)
        return self.temperature.item()