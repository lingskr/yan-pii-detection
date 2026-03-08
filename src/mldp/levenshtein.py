from __future__ import annotations

from typing import Iterable, Optional


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance with O(min(m,n)) memory."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            ins = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(ins, delete, replace))
        previous = current
    return previous[-1]


def correct_with_levenshtein(
    text_prediction: str,
    ocr_candidates: Iterable[str],
    threshold: int = 3,
    default_value: Optional[str] = None,
) -> str:
    """Pick OCR candidate closest to text prediction under edit-distance threshold."""
    candidates = [c for c in ocr_candidates if c is not None and str(c).strip() != ""]
    text_prediction = "" if text_prediction is None else str(text_prediction)

    if not candidates:
        return default_value if default_value is not None else text_prediction

    best = min(candidates, key=lambda c: edit_distance(text_prediction, str(c)))
    dist = edit_distance(text_prediction, str(best))

    if dist <= threshold:
        return str(best)

    return default_value if default_value is not None else text_prediction
