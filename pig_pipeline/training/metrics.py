from __future__ import annotations

from typing import Any

from sklearn.metrics import classification_report, f1_score


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def per_class_report(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
