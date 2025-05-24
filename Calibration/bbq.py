import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB


def bbq_calibrate(val_logits: np.ndarray, val_labels: np.ndarray):
    """Используем CalibratedClassifierCV как приближение BBQ (байесовское биннинг)."""
    base = GaussianNB()
    calibrated = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv="prefit")
    base.fit(val_logits, val_labels)
    calibrated.fit(val_logits, val_labels)
    return calibrated