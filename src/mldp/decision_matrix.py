from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Tuple


PairKey = Tuple[Hashable, Hashable]


@dataclass
class MatrixCell:
    weight_text: float
    weight_image: float
    support: int


class DecisionMatrixFusion:
    """Validation-trained decision matrix W for text/image label fusion."""

    def __init__(self, min_support: int = 5):
        self.min_support = min_support
        self.matrix: Dict[PairKey, MatrixCell] = {}
        self.default_weights: Tuple[float, float] = (0.5, 0.5)
        self.global_text_acc = 0.0
        self.global_image_acc = 0.0

    def fit(
        self,
        y_true: Iterable[Hashable],
        y_text: Iterable[Hashable],
        y_image: Iterable[Hashable],
    ) -> "DecisionMatrixFusion":
        y_true = list(y_true)
        y_text = list(y_text)
        y_image = list(y_image)

        if not (len(y_true) == len(y_text) == len(y_image)):
            raise ValueError("y_true, y_text, y_image lengths must match")

        n = len(y_true)
        if n == 0:
            raise ValueError("Empty validation data")

        self.global_text_acc = sum(int(t == y) for t, y in zip(y_text, y_true)) / n
        self.global_image_acc = sum(int(v == y) for v, y in zip(y_image, y_true)) / n
        self.default_weights = (1.0, 0.0) if self.global_text_acc >= self.global_image_acc else (0.0, 1.0)

        grouped: Dict[PairKey, List[int]] = {}
        for idx, (t, v) in enumerate(zip(y_text, y_image)):
            grouped.setdefault((t, v), []).append(idx)

        for key, ids in grouped.items():
            support = len(ids)
            if support < self.min_support:
                self.matrix[key] = MatrixCell(*self.default_weights, support)
                continue

            text_correct = sum(int(y_text[i] == y_true[i]) for i in ids)
            image_correct = sum(int(y_image[i] == y_true[i]) for i in ids)

            if text_correct > image_correct:
                cell = MatrixCell(1.0, 0.0, support)
            elif image_correct > text_correct:
                cell = MatrixCell(0.0, 1.0, support)
            else:
                # tie: use equal fusion; when predictions disagree fallback by default weights
                cell = MatrixCell(0.5, 0.5, support)

            self.matrix[key] = cell

        return self

    def _weights_for_pair(self, text_pred: Hashable, image_pred: Hashable) -> Tuple[float, float]:
        cell = self.matrix.get((text_pred, image_pred))
        if cell is None:
            return self.default_weights
        return (cell.weight_text, cell.weight_image)

    def predict(self, y_text: Iterable[Hashable], y_image: Iterable[Hashable]) -> List[Hashable]:
        preds = []
        for t, v in zip(y_text, y_image):
            wt, wv = self._weights_for_pair(t, v)
            if wt > wv:
                preds.append(t)
            elif wv > wt:
                preds.append(v)
            else:
                preds.append(t if t == v else (t if self.default_weights[0] >= self.default_weights[1] else v))
        return preds

    def decision_stats(self) -> dict:
        return {
            "pairs": len(self.matrix),
            "min_support": self.min_support,
            "default_weights": {"text": self.default_weights[0], "image": self.default_weights[1]},
            "global_text_acc": self.global_text_acc,
            "global_image_acc": self.global_image_acc,
        }
