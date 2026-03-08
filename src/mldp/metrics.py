from __future__ import annotations

from typing import Callable, Iterable, Sequence, Set, Tuple

import numpy as np


def micro_f1(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths must match")
    if not y_true:
        return 0.0

    tp = sum(int(t == p) for t, p in zip(y_true, y_pred))
    fp = len(y_true) - tp
    fn = fp

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return 2 * precision * recall / (precision + recall + 1e-12)


def normalize_numeric_text(value: str) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    s = s.replace(".", "-") if len(s) >= 8 and s.count(".") == 2 else s
    return s


def numeric_exact_match(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    fields: Sequence[str],
    numeric_fields: Set[str],
) -> float:
    if not (len(y_true) == len(y_pred) == len(fields)):
        raise ValueError("y_true, y_pred, fields lengths must match")

    ids = [i for i, f in enumerate(fields) if f in numeric_fields]
    if not ids:
        return 0.0

    ok = 0
    for i in ids:
        ok += int(normalize_numeric_text(y_true[i]) == normalize_numeric_text(y_pred[i]))

    return ok / len(ids)


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI for a pairwise metric."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths must match")
    if len(y_true) == 0:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        ids = rng.integers(0, n, size=n)
        vals[i] = metric_fn(y_true[ids], y_pred[ids])

    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    return (
        float(vals.mean()),
        float(np.quantile(vals, lower_q)),
        float(np.quantile(vals, upper_q)),
    )
