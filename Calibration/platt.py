import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def platt_scale(val_logits: np.ndarray, val_labels: np.ndarray):
    """Возвращает обученную логистическую регрессию для калибровки (мультиноминальная)."""
    n_classes = val_logits.shape[1]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, multi_class="multinomial"))
    pipe.fit(val_logits, val_labels)
    return pipe