import numpy as np
from sklearn.isotonic import IsotonicRegression


def isotonic_calibrate(val_probs: np.ndarray, val_labels: np.ndarray):
    """Поэлементная изотоническая регрессия для каждого класса."""
    n, C = val_probs.shape
    calibrators = []
    for c in range(C):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(val_probs[:, c], (val_labels == c).astype(float))
        calibrators.append(ir)
    return calibrators


def apply_isotonic(calibrators, probs: np.ndarray):
    out = np.zeros_like(probs)
    for c, ir in enumerate(calibrators):
        out[:, c] = ir.transform(probs[:, c])
    out /= out.sum(1, keepdims=True)  # нормировка
    return out