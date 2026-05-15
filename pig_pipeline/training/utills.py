from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from pig_pipeline.training.metrics import macro_f1, per_class_report


def flush_torch_memory_2() -> None:
    """Release Python and CUDA cached memory between large inference phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def flush_torch_memory() -> None:
    """Release Python and CUDA cached memory when multiple GPUs are in use."""
    gc.collect()
    if not torch.cuda.is_available():
        return

    device_count = torch.cuda.device_count()
    if device_count <= 1:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        return

    current_device = torch.cuda.current_device()
    try:
        for idx in range(device_count):
            torch.cuda.set_device(idx)
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    finally:
        torch.cuda.set_device(current_device)


# METRICS AND PLOTTING UTILS


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    return_predictions: bool = False,
) -> dict[str, Any]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    metrics: dict[str, Any] = {
        "top1": float(np.mean(y_true_arr == y_pred_arr)),
        "macro_f1": macro_f1(y_true, y_pred),
        "report": per_class_report(y_true, y_pred),
    }
    if return_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
    return metrics


def calibrate_probs(
    val_probs: np.ndarray,
    y_true: np.ndarray,
    test_probs: np.ndarray,
) -> np.ndarray:
    """Fit per-fold temperature scaling on validation data and return calibrated test probs."""
    from netcal.scaling import TemperatureScaling

    val_probs_safe = np.array(val_probs, dtype=np.float32, copy=True, order="C")
    y_true_safe = np.array(y_true, dtype=np.int64, copy=True, order="C")
    test_probs_safe = np.array(test_probs, dtype=np.float32, copy=True, order="C")

    calibrator = TemperatureScaling()
    calibrator.fit(val_probs_safe, y_true_safe, tensorboard=False)
    calibrated = calibrator.transform(test_probs_safe, mean_estimate=True)
    return np.array(calibrated, dtype=np.float32, copy=False)


def load_class_names(
    class_file: str | Path, fallback_n_classes: int | None = None
) -> list[str]:
    class_path = Path(class_file)
    if class_path.exists():
        lines = [
            line.strip() for line in class_path.read_text(encoding="utf-8").splitlines()
        ]
        names = [line for line in lines if line]
        if names:
            return names

    if fallback_n_classes is None:
        raise FileNotFoundError(f"Class name file not found or empty: {class_path}")
    return [str(i) for i in range(int(fallback_n_classes))]


def save_metrics_json(metrics: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_path


def save_classification_plots(
    y_true: list[int],
    y_pred: list[int],
    report: dict[str, Any],
    class_names: list[str],
    out_dir: str | Path,
    prefix: str,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in [
        (axes[0], cm, "d", "Confusion Matrix (counts)"),
        (axes[1], cm_norm, ".2f", "Confusion Matrix (normalised)"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(str(cm_path), dpi=150)
    plt.close(fig)

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25

    bar_path = out_dir / f"{prefix}_per_class_metrics.png"
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 2), 5))
    for i, metric in enumerate(metrics):
        values = [
            report.get(str(c), {}).get(metric, 0.0) for c in range(len(class_names))
        ]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Per-class precision / recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(bar_path), dpi=150)
    plt.close(fig)

    return cm_path, bar_path
