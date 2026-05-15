from __future__ import annotations

import argparse
from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd

from pig_pipeline.config import ensure_dir, load_config
from pig_pipeline.tracking import RunTracker
from pig_pipeline.training.utills import calibrate_probs, save_metrics_json
from pig_pipeline.training.yolo import (
    collect_val_probs,
    evaluate_on_split,
    load_yolo_model,
    predict_test_probs,
    train_classifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("yolo_cv_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO 5-fold CV + logits ensemble."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_root = ensure_dir(Path(cfg["output"]["root"]) / cfg["output"]["run_name"])
    cv_root = ensure_dir(run_root / "cv_strategy")
    n_folds = int(cfg["split"]["n_folds"])

    test_df = pd.read_csv(run_root / "prepared" / "test_metadata.csv")
    test_probs_all: list[np.ndarray] = []
    fold_scores: list[dict[str, float]] = []

    for fold_idx in range(n_folds):
        _ = pd.read_csv(run_root / "splits" / "cv" / f"fold_{fold_idx}" / "train.csv")
        fold_val_df = pd.read_csv(
            run_root / "splits" / "cv" / f"fold_{fold_idx}" / "val.csv"
        )

        fold_tracker = RunTracker(
            cfg,
            run_name=f"{cfg['output']['run_name']}-cv-fold-{fold_idx}",
            group=cfg["output"]["run_name"],
        )

        model = load_yolo_model(cfg["model"]["weights"])
        train_args = dict(cfg["train"])
        train_args["project"] = str(cv_root / "runs")
        train_args["name"] = f"fold_{fold_idx}"

        best_pt = train_classifier(
            model,
            run_root / "yolo_data" / "cv" / f"fold_{fold_idx}",
            train_args=train_args,
        )
        best_model = load_yolo_model(str(best_pt))

        metrics = evaluate_on_split(
            best_model, fold_val_df, inf_args=dict(cfg["inference"])
        )
        fold_tracker.log(
            {
                "fold/top1": metrics["top1"],
                "fold/macro_f1": metrics["macro_f1"],
                "fold/index": fold_idx,
            }
        )
        fold_scores.append(
            {
                "fold": fold_idx,
                "top1": metrics["top1"],
                "macro_f1": metrics["macro_f1"],
                "model_path": str(best_pt),
            }
        )

        val_probs, y_val_true = collect_val_probs(
            best_model, fold_val_df, inf_args=dict(cfg["inference"])
        )
        test_probs = predict_test_probs(
            best_model, test_df, inf_args=dict(cfg["inference"])
        )
        calibrated_test_probs = calibrate_probs(val_probs, y_val_true, test_probs)
        test_probs_all.append(calibrated_test_probs)

        # Memory clearup (precautionary)
        del val_probs, y_val_true, test_probs
        del best_model, model

        fold_tracker.finish()

    avg_probs = np.mean(np.stack(test_probs_all, axis=0), axis=0)
    pred_classes = np.argmax(avg_probs, axis=1).astype(int)

    submission = pd.DataFrame({"row_id": test_df["row_id"], "class_id": pred_classes})
    submission_path = (
        cv_root / f"submission_cv_ensemble_{cfg['output']['submission_key']}.csv"
    )
    submission.to_csv(submission_path, index=False)

    fold_metrics_path = cv_root / f"fold_metrics_{cfg['output']['submission_key']}.json"
    save_metrics_json(fold_scores, fold_metrics_path)

    logger.info(f"CV submission: {submission_path}")
    logger.info(f"Fold metrics: {fold_metrics_path}")


if __name__ == "__main__":
    main()
