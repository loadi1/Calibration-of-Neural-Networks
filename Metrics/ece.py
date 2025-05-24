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
    probs = _softmax_logits(logits)
    num_classes = probs.size(1)
    ece_total = 0.0
    for c in range(num_classes):
        conf = probs[:, c]
        mask = labels == c
        conf_c = conf[mask]
        if conf_c.numel() == 0:
            continue
        bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        ece_c = 0.0
        for i in range(n_bins):
            m = (conf_c > bins[i]) & (conf_c <= bins[i + 1])
            if m.any():
                acc_bin = (conf_c[m].argmax(dim=0) == 0).float().mean()  # accuracy = 1 т.к. true label
                conf_bin = conf_c[m].mean()
                ece_c += (m.float().mean()) * (acc_bin - conf_bin).abs()
        ece_total += ece_c / num_classes
    return ece_total