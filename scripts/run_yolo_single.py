from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from pig_pipeline.config import ensure_dir, load_config
from pig_pipeline.tracking import RunTracker
from pig_pipeline.training.yolo import evaluate_on_split, load_yolo_model, predict_test_top1, train_classifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("yolo_single_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO single-split training and inference.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_root = ensure_dir(Path(cfg["output"]["root"]) / cfg["output"]["run_name"])
    single_root = ensure_dir(run_root / "single_strategy")

    train_df = pd.read_csv(run_root / "splits" / "single" / "train.csv")
    val_df = pd.read_csv(run_root / "splits" / "single" / "val.csv")
    test_df = pd.read_csv(run_root / "prepared" / "test_metadata.csv")

    tracker = RunTracker(cfg, run_name=f"{cfg['output']['run_name']}-single")

    model = load_yolo_model(cfg["model"]["weights"])
    train_args = dict(cfg["train"])
    train_args["project"] = str(single_root / "runs")
    train_args["name"] = "single"

    best_pt = train_classifier(model, run_root / "yolo_data" / "single", train_args=train_args)
    best_model = load_yolo_model(str(best_pt))

    metrics = evaluate_on_split(best_model, val_df, inf_args=dict(cfg["inference"]))
    metrics["model_path"] = str(best_pt)
    tracker.log({"single/top1": metrics["top1"], "single/macro_f1": metrics["macro_f1"]})

    submission = predict_test_top1(best_model, test_df, inf_args=dict(cfg["inference"]))
    submission_path = single_root / f"submission_single_{cfg['output']['submission_key']}.csv"
    submission.to_csv(submission_path, index=False)

    metrics_path = single_root / f"metrics_single_{cfg['output']['submission_key']}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    tracker.log_file("single_metrics", metrics_path)
    tracker.log_file("single_submission", submission_path)

    logger.info(f"Best checkpoint: {best_pt}")
    logger.info(f"Submission: {submission_path}")
    tracker.finish()


if __name__ == "__main__":
    main()
