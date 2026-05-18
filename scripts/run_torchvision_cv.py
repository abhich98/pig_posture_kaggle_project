from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pig_pipeline.config import ensure_dir, load_config
from pig_pipeline.tracking import RunTracker
from pig_pipeline.training.torchvision import (
    collect_val_probs,
    evaluate_on_split,
    predict_test_probs,
    train_torchvision_classifier,
)
from pig_pipeline.training.utills import (
    calibrate_probs,
    compute_classification_metrics,
    load_class_names,
    save_classification_plots,
    save_metrics_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("torchvision_cv_training")

# Enable Tensor Core optimization for float32 matmul operations
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run torchvision+Lightning 5-fold CV + calibrated probability ensemble."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


@DeprecationWarning
def _common_inf_args(cfg: dict) -> dict[str, int]:
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    inf_cfg = cfg.get("inference", {})
    return {
        "batch": int(inf_cfg.get("batch", train_cfg.get("batch", 32))),
        "num_workers": int(inf_cfg.get("num_workers", train_cfg.get("workers", 4))),
        "imgsz": int(
            inf_cfg.get("imgsz", data_cfg.get("img_size", train_cfg.get("imgsz", 224)))
        ),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    aug_cfg = cfg.get("augment", {})
    inf_cfg = cfg.get("inference", {})

    run_root = ensure_dir(Path(cfg["output"]["root"]) / cfg["output"]["run_name"])
    cv_root = ensure_dir(run_root / "cv_strategy_torchvision")
    i = 0
    while (cv_root / f"runs-{i:02d}").exists():
        i += 1
    cv_run_root = ensure_dir(cv_root / f"runs-{i:02d}")
    n_folds = int(cfg["split"]["n_folds"])

    test_df = pd.read_csv(run_root / "prepared" / "test_metadata.csv")
    test_probs_all: list[np.ndarray] = []
    fold_scores: list[dict[str, float | int | str | None]] = []
    oof_rows: list[pd.DataFrame] = []

    for fold_idx in range(n_folds):
        fold_train_df = pd.read_csv(
            run_root / "splits" / "cv" / f"fold_{fold_idx}" / "train.csv"
        )
        fold_val_df = pd.read_csv(
            run_root / "splits" / "cv" / f"fold_{fold_idx}" / "val.csv"
        )

        fold_tracker = RunTracker(
            cfg,
            run_name=f"{cfg['output']['run_name']}-torchvision-cv-fold-{fold_idx}",
            group=cfg["output"]["run_name"],
        )

        fold_run_dir = ensure_dir(cv_run_root / f"fold_{fold_idx}")
        artifacts = train_torchvision_classifier(
            train_df=fold_train_df,
            val_df=fold_val_df,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            out_dir=fold_run_dir,
            aug_cfg=aug_cfg,
        )

        metrics = evaluate_on_split(
            artifacts.model, fold_val_df, inf_args=inf_cfg, return_predictions=True
        )

        fold_class_file = Path(
            data_cfg.get(
                "class_names_file",
                "/workspace/github/pig_posture_kaggle_project/data/multiview_pig_posture_recognition/pig_posture_classes.txt",
            )
        )
        class_names = load_class_names(
            fold_class_file, fallback_n_classes=int(max(metrics["y_true"]) + 1)
        )

        fold_metrics_path = cv_run_root / f"fold_{fold_idx}_metrics_torchvision.json"
        save_metrics_json(
            {
                "fold": fold_idx,
                "top1": metrics["top1"],
                "macro_f1": metrics["macro_f1"],
                "report": metrics["report"],
                "model_path": str(artifacts.best_ckpt),
                "best_score": artifacts.best_score,
            },
            fold_metrics_path,
        )

        cm_path, bar_path = save_classification_plots(
            y_true=metrics["y_true"],
            y_pred=metrics["y_pred"],
            report=metrics["report"],
            class_names=class_names,
            out_dir=cv_run_root,
            prefix=f"fold_{fold_idx}",
        )

        fold_tracker.log(
            {
                "fold_torchvision/top1": metrics["top1"],
                "fold_torchvision/macro_f1": metrics["macro_f1"],
                "fold_torchvision/index": fold_idx,
            }
        )
        fold_tracker.log_file(f"fold_{fold_idx}_torchvision_metrics", fold_metrics_path)
        fold_tracker.log_file(f"fold_{fold_idx}_torchvision_cm", cm_path)
        fold_tracker.log_file(f"fold_{fold_idx}_torchvision_per_class", bar_path)

        val_probs, y_val_true = collect_val_probs(
            artifacts.model, fold_val_df, inf_args=inf_cfg
        )
        test_probs = predict_test_probs(artifacts.model, test_df, inf_args=inf_cfg)
        calibrated_test_probs = calibrate_probs(val_probs, y_val_true, test_probs)
        test_probs_all.append(calibrated_test_probs)

        fold_scores.append(
            {
                "fold": fold_idx,
                "top1": metrics["top1"],
                "macro_f1": metrics["macro_f1"],
                "model_path": str(artifacts.best_ckpt),
                "best_score": artifacts.best_score,
            }
        )

        fold_oof = pd.DataFrame(
            {
                "row_id": fold_val_df["row_id"].astype(str).tolist(),
                "class_id_true": metrics["y_true"],
                "class_id_pred": metrics["y_pred"],
                "fold": fold_idx,
            }
        )
        oof_rows.append(fold_oof)

        fold_tracker.finish()

    avg_probs = np.mean(np.stack(test_probs_all, axis=0), axis=0)
    pred_classes = np.argmax(avg_probs, axis=1).astype(int)

    submission = pd.DataFrame({"row_id": test_df["row_id"], "class_id": pred_classes})
    submission_path = (
        cv_root
        / f"submission_cv_torchvision_ensemble_{cfg['output']['submission_key']}.csv"
    )
    submission.to_csv(submission_path, index=False)

    fold_metrics_all_path = (
        cv_root / f"fold_metrics_torchvision_{cfg['output']['submission_key']}.json"
    )
    with fold_metrics_all_path.open("w", encoding="utf-8") as f:
        json.dump(fold_scores, f, indent=2)

    oof_df = pd.concat(oof_rows, axis=0, ignore_index=True)
    oof_path = (
        cv_root / f"oof_predictions_torchvision_{cfg['output']['submission_key']}.csv"
    )
    oof_df.to_csv(oof_path, index=False)

    y_true_all = oof_df["class_id_true"].astype(int).tolist()
    y_pred_all = oof_df["class_id_pred"].astype(int).tolist()
    oof_metrics = compute_classification_metrics(y_true_all, y_pred_all)

    class_file = Path(
        data_cfg.get(
            "class_names_file",
            "/workspace/github/pig_posture_kaggle_project/data/multiview_pig_posture_recognition/pig_posture_classes.txt",
        )
    )
    class_names = load_class_names(
        class_file, fallback_n_classes=int(max(y_true_all) + 1)
    )

    oof_metrics_path = (
        cv_root / f"oof_metrics_torchvision_{cfg['output']['submission_key']}.json"
    )
    save_metrics_json(oof_metrics, oof_metrics_path)

    oof_cm_path, oof_bar_path = save_classification_plots(
        y_true=y_true_all,
        y_pred=y_pred_all,
        report=oof_metrics["report"],
        class_names=class_names,
        out_dir=cv_root,
        prefix="oof",
    )

    final_tracker = RunTracker(
        cfg,
        run_name=f"{cfg['output']['run_name']}-torchvision-cv-ensemble",
        group=cfg["output"]["run_name"],
    )
    final_tracker.log(
        {
            "cv_torchvision/oof_top1": oof_metrics["top1"],
            "cv_torchvision/oof_macro_f1": oof_metrics["macro_f1"],
        }
    )
    final_tracker.log_file("cv_torchvision_fold_metrics", fold_metrics_all_path)
    final_tracker.log_file("cv_torchvision_oof_predictions", oof_path)
    final_tracker.log_file("cv_torchvision_oof_metrics", oof_metrics_path)
    final_tracker.log_file("cv_torchvision_oof_confusion_matrix", oof_cm_path)
    final_tracker.log_file("cv_torchvision_oof_per_class", oof_bar_path)
    final_tracker.log_file("cv_torchvision_submission", submission_path)
    final_tracker.finish()

    logger.info("CV submission: %s", submission_path)
    logger.info("OOF predictions: %s", oof_path)
    logger.info("OOF metrics: %s", oof_metrics_path)


if __name__ == "__main__":
    main()
