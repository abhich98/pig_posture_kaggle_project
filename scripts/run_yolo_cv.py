from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_pipeline.config import ensure_dir, load_config
from pig_pipeline.tracking import RunTracker
from pig_pipeline.training.yolo import (
    evaluate_on_split,
    load_yolo_model,
    predict_test_logits,
    train_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO 5-fold CV + logits ensemble.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_root = ensure_dir(Path(cfg["output"]["root"]) / cfg["output"]["run_name"])
    cv_root = ensure_dir(run_root / "cv_strategy")
    n_folds = int(cfg["split"]["n_folds"])

    test_df = pd.read_csv(run_root / "prepared" / "test_metadata.csv")
    test_logits_all: list[np.ndarray] = []
    fold_scores: list[dict[str, float]] = []

    aggregate_tracker = RunTracker(cfg, run_name=f"{cfg['output']['run_name']}-cv", group=cfg["output"]["run_name"])

    for fold_idx in range(n_folds):
        fold_train_df = pd.read_csv(run_root / "splits" / "cv" / f"fold_{fold_idx}" / "train.csv")
        fold_val_df = pd.read_csv(run_root / "splits" / "cv" / f"fold_{fold_idx}" / "val.csv")
        _ = fold_train_df

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

        metrics = evaluate_on_split(best_model, fold_val_df)
        fold_tracker.log(
            {
                "fold/top1": metrics["top1"],
                "fold/macro_f1": metrics["macro_f1"],
                "fold/index": fold_idx,
            }
        )
        fold_scores.append({"fold": fold_idx, "top1": metrics["top1"], "macro_f1": metrics["macro_f1"]})

        test_logits = predict_test_logits(best_model, test_df)
        test_logits_all.append(test_logits)
        fold_tracker.finish()

    avg_logits = np.mean(np.stack(test_logits_all, axis=0), axis=0)
    pred_classes = np.argmax(avg_logits, axis=1).astype(int)

    submission = pd.DataFrame({"row_id": test_df["row_id"], "class_id": pred_classes})
    submission_path = cv_root / "submission_cv_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    fold_metrics_path = cv_root / "fold_metrics.json"
    with fold_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(fold_scores, f, indent=2)

    mean_top1 = float(np.mean([f["top1"] for f in fold_scores]))
    mean_macro = float(np.mean([f["macro_f1"] for f in fold_scores]))
    aggregate_tracker.log({"cv/mean_top1": mean_top1, "cv/mean_macro_f1": mean_macro})
    aggregate_tracker.log_file("cv_fold_metrics", fold_metrics_path)
    aggregate_tracker.log_file("cv_submission", submission_path)
    aggregate_tracker.finish()

    print(f"CV submission: {submission_path}")
    print(f"Fold metrics: {fold_metrics_path}")


if __name__ == "__main__":
    main()
