import torch
import torch.nn.functional as F


def _softmax_logits(logits: torch.Tensor):
    return F.softmax(logits, dim=1)


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    probs = _softmax_logits(logits)
    conf, preds = probs.max(1)
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            acc_bin = (preds[mask] == labels[mask]).float().mean()
            conf_bin = conf[mask].mean()
            ece += (mask.float().mean()) * (acc_bin - conf_bin).abs()
    return ece.item()


def adaptive_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    probs = _softmax_logits(logits)
    conf, preds = probs.max(1)
    # сортируем по уверенности, делим на n_bins равных по размеру блоков
    sorted_conf, idx = torch.sort(conf)
    sorted_labels = labels[idx]
    sorted_preds = preds[idx]
    bin_size = int(len(conf) / n_bins)
    ece = 0.0
    for i in range(n_bins):
        start, end = i * bin_size, (i + 1) * bin_size if i < n_bins - 1 else len(conf)
        if end - start == 0:
            continue
        conf_bin = sorted_conf[start:end].mean()
        acc_bin = (sorted_preds[start:end] == sorted_labels[start:end]).float().mean()
        ece += (end - start) / len(conf) * (acc_bin - conf_bin).abs()
    return ece


def classwise_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """
    Classwise ECE: average over K binary‐ECEs, one per class.
    """
    probs = _softmax_logits(logits)       # (N, K)
    N, K   = probs.shape
    ece_total = 0.0

    # prepare bin edges once
    device = logits.device
    bins   = torch.linspace(0.0, 1.0, n_bins + 1, device=device)

    # for each class, compute a binary ECE
    for c in range(K):
        p_c = probs[:, c]                 # confidences for class c
        y_c = (labels == c).float()       # 1 if true label==c, else 0

        # skip if class never appears
        if y_c.sum().item() == 0:
            continue

        ece_c = 0.0
        # binning
        for i in range(n_bins):
            # [bins[i], bins[i+1]) except include right edge for last bin
            if i < n_bins - 1:
                mask = (p_c >= bins[i]) & (p_c < bins[i+1])
            else:
                mask = (p_c >= bins[i]) & (p_c <= bins[i+1])

            if mask.sum().item() == 0:
                continue

            # average predicted confidence in bin
            conf_bin = p_c[mask].mean()
            # fraction of true positives in bin
            acc_bin  = y_c[mask].mean()
            # weight by fraction of samples in bin
            weight   = mask.float().mean()

            ece_c += weight * torch.abs(acc_bin - conf_bin)

        ece_total += ece_c

    # average over K classes that actually appear
    return (ece_total / K).item()