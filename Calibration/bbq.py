"""Bayesian Binning into Quantiles (BBQ) для многоклассовой калибровки.
Упрощён, но ближе к оригиналу (Naeini et al., 2015).
Алгоритм:
1. Сортируем выборку по max‑confidence; формируем `M` квантильных бинов
   с ~равным количеством примеров.
2. Для каждого бина считаем апостериор Dirichlet(α=1) → предиктивное   
   p(class=k | bin) = (α + n_k) / (K·α + n_bin).
3. Итоговая вероятность для любого сэмпла = p(bin)·p(class|bin).
   Здесь p(bin) ≈ n_bin / N (uniform prior на структуру деревьев опускаем).

Работает для K‑классов; без зависимостей, кроме numpy.
"""
from __future__ import annotations
import numpy as np
from bisect import bisect_right

class BBQCalibrator:
    def __init__(self, n_bins: int = 15, alpha: float = 1.0):
        self.n_bins = n_bins
        self.alpha  = alpha
        self.bin_edges: list[float] = []      # границы confidence
        self.bin_dirichlet: list[np.ndarray] = []  # α-вектора размера K
        self.prior_bin: np.ndarray | None = None   # p(bin)
        self.K = None

    def _max_conf(self, logits: np.ndarray):
        prob = softmax_np(logits)
        return prob.max(1)

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        prob = softmax_np(logits)
        conf = prob.max(1)
        self.K = prob.shape[1]

        # сортируем по уверенности
        order = np.argsort(conf)
        conf_sorted   = conf[order]
        prob_sorted   = prob[order]
        labels_sorted = labels[order]

        N = len(conf)
        edges = [0]
        for i in range(1, self.n_bins):
            edges.append(int(i * N / self.n_bins))
        edges.append(N)

        self.bin_edges = [conf_sorted[idx] for idx in edges[1:-1]]  # пороги
        self.bin_dirichlet = []
        n_bin = []
        for start, end in zip(edges[:-1], edges[1:]):
            y_bin = labels_sorted[start:end]
            counts = np.bincount(y_bin, minlength=self.K)
            alpha_vec = counts + self.alpha   # Dirichlet posterior α
            self.bin_dirichlet.append(alpha_vec)
            n_bin.append(end - start)
        self.prior_bin = np.array(n_bin) / N
        return self

    def _posterior_prob(self, bin_idx: int):
        α = self.bin_dirichlet[bin_idx]
        return α / α.sum()

    def predict_proba(self, logits: np.ndarray):
        prob = softmax_np(logits)
        conf = prob.max(1)
        out = np.zeros_like(prob)
        for i, c in enumerate(conf):
            bin_idx = bisect_right(self.bin_edges, c)
            out[i] = self.prior_bin[bin_idx] * self._posterior_prob(bin_idx)
        # нормируем — p(bin)·p(class|bin) уже суммируется в 1, но safeguard
        out /= out.sum(1, keepdims=True)
        return out

# -------------------------------------------------------------------------

def softmax_np(z: np.ndarray):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

# convenience wrapper ------------------------------------------------------

def bbq_calibrate(val_logits: np.ndarray, val_labels: np.ndarray, n_bins: int = 15):
    calib = BBQCalibrator(n_bins=n_bins)
    calib.fit(val_logits, val_labels)
    return calib