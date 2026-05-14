from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

from pig_pipeline.config import ensure_dir, load_config
from pig_pipeline.tracking import RunTracker
from pig_pipeline.training.torchvision import (
    evaluate_on_split,
    predict_test_top1,
    train_torchvision_classifier,
)
from pig_pipeline.training.utills import load_class_names, save_classification_plots, save_metrics_json


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("torchvision_single_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run torchvision+Lightning single-split training and inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    inf_cfg = cfg.get("inference", {})
    aug_cfg = cfg.get("augment", {})

    run_root = ensure_dir(Path(cfg["output"]["root"]) / cfg["output"]["run_name"])
    single_root = ensure_dir(run_root / "single_strategy_torchvision")

    train_df = pd.read_csv(run_root / "splits" / "single" / "train.csv")
    val_df = pd.read_csv(run_root / "splits" / "single" / "val.csv")
    test_df = pd.read_csv(run_root / "prepared" / "test_metadata.csv")

    tracker = RunTracker(cfg, run_name=f"{cfg['output']['run_name']}-torchvision-single")

    train_out_dir = ensure_dir(single_root / "runs")
    artifacts = train_torchvision_classifier(
        train_df=train_df,
        val_df=val_df,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        out_dir=train_out_dir,
        aug_cfg=aug_cfg,
    )
    best_model = artifacts.model

    inf_args = {
        "batch": int(inf_cfg.get("batch", 32)),
        "num_workers": int(inf_cfg.get("num_workers", 4)),
        "imgsz": int(inf_cfg.get("imgsz", 224)),
    }

    metrics = evaluate_on_split(best_model, val_df, inf_args=inf_args, return_predictions=True)
    metrics["model_path"] = str(artifacts.best_ckpt)
    metrics["best_score"] = artifacts.best_score

    class_file = Path(
        data_cfg.get(
            "class_names_file",
            "/workspace/github/pig_posture_kaggle_project/data/multiview_pig_posture_recognition/pig_posture_classes.txt",
        )
    )
    class_names = load_class_names(class_file, fallback_n_classes=int(max(metrics["y_true"]) + 1))

    metrics_path = single_root / f"metrics_single_torchvision_{cfg['output']['submission_key']}.json"
    save_metrics_json(
        {
            "top1": metrics["top1"],
            "macro_f1": metrics["macro_f1"],
            "report": metrics["report"],
            "model_path": metrics["model_path"],
            "best_score": metrics["best_score"],
        },
        metrics_path,
    )

    cm_path, bar_path = save_classification_plots(
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        report=metrics["report"],
        class_names=class_names,
        out_dir=single_root,
        prefix="single",
    )

    submission = predict_test_top1(best_model, test_df, inf_args=inf_args)
    submission_path = single_root / f"submission_single_torchvision_{cfg['output']['submission_key']}.csv"
    submission.to_csv(submission_path, index=False)

    tracker.log({
        "single_torchvision/top1": metrics["top1"],
        "single_torchvision/macro_f1": metrics["macro_f1"],
    })
    tracker.log_file("single_torchvision_metrics", metrics_path)
    tracker.log_file("single_torchvision_confusion_matrix", cm_path)
    tracker.log_file("single_torchvision_per_class", bar_path)
    tracker.log_file("single_torchvision_submission", submission_path)

    logger.info("Best checkpoint: %s", artifacts.best_ckpt)
    logger.info("Validation metrics saved: %s", metrics_path)
    logger.info("Submission saved: %s", submission_path)
    tracker.finish()


if __name__ == "__main__":
    main()
