from __future__ import annotations

import argparse
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

from pig_pipeline.config import load_config
from pig_pipeline.training.utills import flush_torch_memory
from pig_pipeline.training.yolo import (
    evaluate_on_split,
    load_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("yolo_ood_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained YOLO model on an OOD CSV and produce metric plots."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (uses data.* and inference.* settings)",
    )
    parser.add_argument(
        "--yolo-run-dir",
        required=True,
        help="YOLO training run directory that contains weights/best.pt",
    )
    parser.add_argument(
        "--weights-name",
        default="best",
        help="Name of the weights file to use (default: best.pt)",
    )
    parser.add_argument(
        "--ood-csv",
        default="/workspace/github/pig_posture_kaggle_project/data/generated/train1_local_run/prepared/train_ood_metadata.csv",
        help="CSV with OOD labels that already has crop_path and class_id columns",
    )
    return parser.parse_args()


def _resolve_weights(run_dir: Path, weights_name: str) -> Path:
    weights_path = run_dir / "weights" / f"{weights_name}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find weights at: {weights_path}")
    return weights_path


def _save_confusion_matrix(
    y_true: list[int], y_pred: list[int], class_names: list[str], out_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

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
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _save_per_class_bars(report: dict, class_names: list[str], out_path: Path) -> None:
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25

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
    ax.set_title("Per-class precision / recall / F1", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    inf_cfg = cfg.get("inference", {})
    inf_args = {
        "chunk": int(inf_cfg.get("chunk", 1000)),
        "imgsz": int(inf_cfg.get("imgsz", int(data_cfg["img_size"]))),
        "batch": int(inf_cfg.get("batch", 32)),
    }

    yolo_run_dir = Path(args.yolo_run_dir).resolve()
    ood_csv = Path(args.ood_csv).resolve()

    if not ood_csv.exists():
        raise FileNotFoundError(f"OOD CSV does not exist: {ood_csv}")

    weights_path = _resolve_weights(yolo_run_dir, args.weights_name)
    logger.info("Using weights: %s", weights_path)

    ood_df = pd.read_csv(ood_csv)
    if "crop_path" not in ood_df.columns or "class_id" not in ood_df.columns:
        raise ValueError("OOD CSV must contain 'crop_path' and 'class_id' columns")
    logger.info("Loaded OOD CSV: %d samples from %s", len(ood_df), ood_csv)

    model = load_model(str(weights_path))
    flush_torch_memory()

    logger.info(
        "Running OOD inference on %d samples (inf_args=%s)...", len(ood_df), inf_args
    )
    metrics = evaluate_on_split(model, ood_df, inf_args=inf_args, return_predictions=True)
    flush_torch_memory()

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    n_classes = max(max(y_true), max(y_pred)) + 1
    class_names = [str(i) for i in range(n_classes)]

    out_dir = yolo_run_dir / f"ood_validation_{args.weights_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics_ood.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "top1": metrics["top1"],
                "macro_f1": metrics["macro_f1"],
                "report": metrics["report"],
            },
            f,
            indent=2,
        )
    logger.info("top1=%.4f  macro_f1=%.4f", metrics["top1"], metrics["macro_f1"])
    logger.info("Saved metrics: %s", metrics_path)

    cm_path = out_dir / "confusion_matrix.png"
    _save_confusion_matrix(y_true, y_pred, class_names, cm_path)
    logger.info("Saved confusion matrix: %s", cm_path)

    bar_path = out_dir / "per_class_metrics.png"
    _save_per_class_bars(metrics["report"], class_names, bar_path)
    logger.info("Saved per-class bar chart: %s", bar_path)

    logger.info("All OOD validation outputs saved to: %s", out_dir)


if __name__ == "__main__":
    main()
