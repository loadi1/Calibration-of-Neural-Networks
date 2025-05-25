import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn import __version__ as skl_ver

def bbq_calibrate(val_logits: np.ndarray, val_labels: np.ndarray):
    """Используем CalibratedClassifierCV как приближение BBQ (байесовское биннинг)."""
    base = GaussianNB()
    base.fit(val_logits, val_labels)          # cv='prefit' ждёт уже обученный

    # Совместимость с версиями sklearn
    kw = dict(method="isotonic", cv="prefit")
    if tuple(map(int, skl_ver.split('.')[:2])) >= (1, 2):
        kw["estimator"] = base               # новое имя параметра
    else:
        kw["base_estimator"] = base          # старое имя

    calibrated = CalibratedClassifierCV(**kw)
    calibrated.fit(val_logits, val_labels)    # дообучаем изотонику
    return calibrated