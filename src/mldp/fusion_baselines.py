from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


@dataclass
class BaselineResult:
    name: str
    predictions: List[str]
    extra: Dict[str, float]


class TemperatureScaler:
    """Simple temperature scaling with grid search on validation NLL."""

    def __init__(self):
        self.temperature = 1.0

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, logits: np.ndarray, y_idx: np.ndarray, grid: Optional[np.ndarray] = None) -> "TemperatureScaler":
        if grid is None:
            grid = np.linspace(0.5, 5.0, 46)

        best_t = 1.0
        best_nll = float("inf")
        for t in grid:
            probs = self._softmax(logits / t)
            nll = -np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)).mean()
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        self.temperature = best_t
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature


def _argmax_labels(probs: np.ndarray, classes: List[str]) -> List[str]:
    return [classes[i] for i in probs.argmax(axis=1)]


def avg_prob_fusion(p_text: np.ndarray, p_img: np.ndarray, classes: List[str]) -> BaselineResult:
    p = 0.5 * (p_text + p_img)
    return BaselineResult("avg_prob", _argmax_labels(p, classes), {})


def weighted_prob_fusion(
    p_text_val: np.ndarray,
    p_img_val: np.ndarray,
    y_val_idx: np.ndarray,
    p_text_test: np.ndarray,
    p_img_test: np.ndarray,
    classes: List[str],
) -> BaselineResult:
    best_alpha = 0.5
    best_acc = -1.0
    for alpha in np.linspace(0.0, 1.0, 101):
        p = alpha * p_text_val + (1 - alpha) * p_img_val
        acc = (p.argmax(axis=1) == y_val_idx).mean()
        if acc > best_acc:
            best_acc = float(acc)
            best_alpha = float(alpha)

    p_test = best_alpha * p_text_test + (1 - best_alpha) * p_img_test
    return BaselineResult("weighted_prob", _argmax_labels(p_test, classes), {"alpha": best_alpha})


def logreg_fusion(
    z_text_val: np.ndarray,
    z_img_val: np.ndarray,
    y_val_idx: np.ndarray,
    z_text_test: np.ndarray,
    z_img_test: np.ndarray,
    classes: List[str],
) -> BaselineResult:
    x_val = np.concatenate([z_text_val, z_img_val], axis=1)
    x_test = np.concatenate([z_text_test, z_img_test], axis=1)

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(x_val, y_val_idx)
    y_test = clf.predict(x_test)
    return BaselineResult("logreg", [classes[i] for i in y_test], {})


def mlp_fusion(
    z_text_val: np.ndarray,
    z_img_val: np.ndarray,
    y_val_idx: np.ndarray,
    z_text_test: np.ndarray,
    z_img_test: np.ndarray,
    classes: List[str],
) -> BaselineResult:
    x_val = np.concatenate([z_text_val, z_img_val], axis=1)
    x_test = np.concatenate([z_text_test, z_img_test], axis=1)

    clf = MLPClassifier(hidden_layer_sizes=(128,), random_state=42, max_iter=300)
    clf.fit(x_val, y_val_idx)
    y_test = clf.predict(x_test)
    return BaselineResult("mlp", [classes[i] for i in y_test], {})
