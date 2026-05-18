from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from pig_pipeline.config import load_config
from pig_pipeline.training.torchvision import (
    evaluate_on_split,
    load_torchvision_model_from_checkpoint,
)
from pig_pipeline.training.utills import (
    load_class_names,
    save_classification_plots,
    save_metrics_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("torchvision_ood_validation")

# Enable Tensor Core optimization for float32 matmul operations
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained torchvision model on an OOD CSV and produce metric plots."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (uses model.*, train.*, data.* and inference.* settings)",
    )
    parser.add_argument(
        "--torchvision-run-dir",
        required=True,
        help="Torchvision training run directory that contains .ckpt files (usually .../single_strategy_torchvision/runs)",
    )
    parser.add_argument(
        "--weights-name",
        default="best",
        help="Name of the checkpoint file to use (default: best.ckpt)",
    )
    parser.add_argument(
        "--ood-csv",
        default="/workspace/github/pig_posture_kaggle_project/data/generated/train1_local_run/prepared/train_ood_metadata.csv",
        help="CSV with OOD labels that already has crop_path and class_id columns",
    )
    return parser.parse_args()


def _resolve_checkpoint(run_dir: Path, weights_name: str = "best") -> Path:
    if run_dir.is_file() and run_dir.suffix == ".ckpt":
        return run_dir

    if not run_dir.exists():
        raise FileNotFoundError(f"Torchvision run directory does not exist: {run_dir}")

    best_candidates = sorted(run_dir.glob(f"*{weights_name}*.ckpt"))
    if not best_candidates:
        best_candidates = sorted(run_dir.rglob(f"*{weights_name}*.ckpt"))
    if best_candidates:
        return best_candidates[-1]
    else:
        raise FileNotFoundError(f"Could not find checkpoint with name '{weights_name}' in: {run_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    inf_cfg = cfg.get("inference", {})

    run_dir = Path(args.torchvision_run_dir).resolve()
    ood_csv = Path(args.ood_csv).resolve()
    if not ood_csv.exists():
        raise FileNotFoundError(f"OOD CSV does not exist: {ood_csv}")

    ood_df = pd.read_csv(ood_csv)
    if "crop_path" not in ood_df.columns or "class_id" not in ood_df.columns:
        raise ValueError("OOD CSV must contain 'crop_path' and 'class_id' columns")
    logger.info("Loaded OOD CSV: %d samples from %s", len(ood_df), ood_csv)

    class_file = Path(
        data_cfg.get(
            "class_names_file",
            "/workspace/github/pig_posture_kaggle_project/data/multiview_pig_posture_recognition/pig_posture_classes.txt",
        )
    )
    class_names = load_class_names(
        class_file, fallback_n_classes=int(ood_df["class_id"].max()) + 1
    )

    ckpt_path = _resolve_checkpoint(run_dir, weights_name=args.weights_name)
    logger.info("Using checkpoint: %s", ckpt_path)

    num_classes = len(class_names)
    model = load_torchvision_model_from_checkpoint(
        checkpoint_path=ckpt_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        num_classes=num_classes,
        map_location="cpu",
    )
    model.eval()

    logger.info(
        "Running OOD inference on %d samples ...", len(ood_df)
    )
    metrics = evaluate_on_split(model, ood_df, inf_args=inf_cfg, return_predictions=True)

    out_dir = run_dir / f"ood_validation_{args.weights_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics_ood.json"
    save_metrics_json(
        {
            "top1": metrics["top1"],
            "macro_f1": metrics["macro_f1"],
            "report": metrics["report"],
            "model_path": str(ckpt_path),
        },
        metrics_path,
    )
    logger.info("top1=%.4f  macro_f1=%.4f", metrics["top1"], metrics["macro_f1"])
    logger.info("Saved metrics: %s", metrics_path)

    cm_path, bar_path = save_classification_plots(
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        report=metrics["report"],
        class_names=class_names,
        out_dir=out_dir,
        prefix="ood",
    )
    cm_out = out_dir / "confusion_matrix.png"
    bar_out = out_dir / "per_class_metrics.png"
    cm_path.replace(cm_out)
    bar_path.replace(bar_out)
    logger.info("Saved confusion matrix: %s", cm_out)
    logger.info("Saved per-class bar chart: %s", bar_out)
    logger.info("All OOD validation outputs saved to: %s", out_dir)


if __name__ == "__main__":
    main()
