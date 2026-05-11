from __future__ import annotations

import argparse
from pathlib import Path
import logging

import pandas as pd

from pig_pipeline.config import ensure_dir, load_config
from pig_pipeline.data.prepare import build_crop_metadata, materialize_yolo_classification_dir
from pig_pipeline.data.splits import build_cv_splits, build_single_split
from pig_pipeline.tracking import RunTracker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare crop dataset and split manifests.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    out_cfg = cfg["output"]

    run_root = ensure_dir(Path(out_cfg["root"]) / out_cfg["run_name"])
    tracker = RunTracker(cfg, run_name=f"{out_cfg['run_name']}-prepare", group=out_cfg["run_name"])
    prep_root = ensure_dir(run_root / "prepared")
    crops_train = ensure_dir(prep_root / "crops" / "train")
    crops_test = ensure_dir(prep_root / "crops" / "test")

    train_meta_path = prep_root / "train_metadata.csv"
    test_meta_path = prep_root / "test_metadata.csv"

    train_df = build_crop_metadata(
        csv_path=data_cfg["train_csv"],
        image_dir=data_cfg["train_image_dir"],
        crops_root=crops_train,
        output_csv_path=train_meta_path,
        img_size=int(data_cfg["img_size"]),
        pad=int(data_cfg["pad"]),
        is_train=True,
        n_jobs=int(data_cfg.get("prep_n_jobs", 1)),
    )
    test_df = build_crop_metadata(
        csv_path=data_cfg["test_csv"],
        image_dir=data_cfg["test_image_dir"],
        crops_root=crops_test,
        output_csv_path=test_meta_path,
        img_size=int(data_cfg["img_size"]),
        pad=int(data_cfg["pad"]),
        is_train=False,
        n_jobs=int(data_cfg.get("prep_n_jobs", 1)),
    )

    split_root = ensure_dir(run_root / "splits")

    single_train_df, single_val_df = build_single_split(
        train_df,
        val_ratio=float(split_cfg["single_val_ratio"]),
        seed=int(split_cfg["seed"]),
        out_dir=split_root / "single",
    )

    yolo_single_root = ensure_dir(run_root / "yolo_data" / "single")
    materialize_yolo_classification_dir(single_train_df, yolo_single_root / "train", n_jobs=int(data_cfg.get("prep_n_jobs", 1)))
    materialize_yolo_classification_dir(single_val_df, yolo_single_root / "val", n_jobs=int(data_cfg.get("prep_n_jobs", 1)))

    folds = build_cv_splits(
        train_df,
        n_folds=int(split_cfg["n_folds"]),
        seed=int(split_cfg["seed"]),
        out_dir=split_root / "cv",
    )
    yolo_cv_root = ensure_dir(run_root / "yolo_data" / "cv")
    for fold_train_df, fold_val_df, fold_idx in folds:
        fold_root = ensure_dir(yolo_cv_root / f"fold_{fold_idx}")
        materialize_yolo_classification_dir(fold_train_df, fold_root / "train", n_jobs=int(data_cfg.get("prep_n_jobs", 1)))
        materialize_yolo_classification_dir(fold_val_df, fold_root / "val", n_jobs=int(data_cfg.get("prep_n_jobs", 1)))

    summary = pd.DataFrame(
        {
            "split": ["train_rows", "test_rows", "single_train", "single_val", "folds"],
            "count": [len(train_df), len(test_df), len(single_train_df), len(single_val_df), len(folds)],
        }
    )
    summary_path = run_root / "prepare_summary.csv"
    summary.to_csv(summary_path, index=False)

    tracker.log(
        {
            "prepare/train_rows": len(train_df),
            "prepare/test_rows": len(test_df),
            "prepare/single_train": len(single_train_df),
            "prepare/single_val": len(single_val_df),
            "prepare/n_folds": len(folds),
        }
    )
    tracker.log_file("prepare_summary", summary_path)
    tracker.finish()
    logger.info(f"Prepared data at: {run_root}")


if __name__ == "__main__":
    main()
