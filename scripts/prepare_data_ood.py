from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import random
import shutil
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from pig_pipeline.config import ensure_dir, load_config
from pig_pipeline.data.prepare import build_crop_metadata
from pig_pipeline.tracking import RunTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("prepare_data_ood")


REQUIRED_COLUMNS = ["row_id", "image_id", "width", "height", "bbox", "class_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare in_dist/OOD crop dataset with image-level OOD split, fixed OOD val, "
            "and oversampling for train distribution control."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
        return
    except OSError:
        pass

    try:
        shutil.copy2(src, dst)
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing source crop file: {src}") from e


def _materialize_yolo_classification_dir_with_replicas(
    split_df: pd.DataFrame,
    out_root: str | Path,
) -> None:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seen_dsts: set[Path] = set()
    for row in split_df.itertuples(index=False):
        class_id = int(getattr(row, "class_id"))
        src = Path(str(getattr(row, "crop_path")))

        rep_idx = int(getattr(row, "replica_idx", 0))
        if rep_idx > 0:
            dst_name = f"{src.stem}__rep{rep_idx:03d}{src.suffix}"
        else:
            dst_name = src.name

        dst = out_root / str(class_id) / dst_name
        if dst in seen_dsts:
            raise ValueError(f"Duplicate destination path during materialization: {dst}")
        seen_dsts.add(dst)

        _safe_link_or_copy(src, dst)


def _dominant_class_per_image(df: pd.DataFrame) -> pd.Series:
    def dominant(series: pd.Series) -> int:
        mode_vals = series.mode(dropna=True)
        if mode_vals.empty:
            return int(series.iloc[0])
        return int(mode_vals.iloc[0])

    return df.groupby("image_id")["class_id"].apply(dominant)


def _split_ood_by_image(
    ood_df: pd.DataFrame,
    val_target_postures: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ood_df.empty:
        raise ValueError("OOD dataframe is empty")

    image_counts = ood_df.groupby("image_id").size().rename("n_postures")
    image_dom_class = _dominant_class_per_image(ood_df).rename("dom_class")
    image_df = pd.concat([image_counts, image_dom_class], axis=1).reset_index()

    n_total_postures = int(len(ood_df))
    target = max(1, int(val_target_postures))
    val_ratio = min(0.95, max(0.05, target / n_total_postures))

    stratify = image_df["dom_class"]
    if stratify.nunique() < 2 or stratify.value_counts().min() < 2:
        stratify = None

    train_img, val_img = train_test_split(
        image_df["image_id"],
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    train_images = set(train_img.tolist())
    val_images = set(val_img.tolist())

    ood_train_df = ood_df[ood_df["image_id"].isin(train_images)].copy()
    ood_val_df = ood_df[ood_df["image_id"].isin(val_images)].copy()

    if ood_train_df.empty or ood_val_df.empty:
        raise ValueError(
            "OOD image split produced empty train or val set. Adjust val_target_postures."
        )

    return (
        ood_train_df.reset_index(drop=True),
        ood_val_df.reset_index(drop=True),
    )


def _compute_repeat_factor(
    n_in_dist: int,
    n_ood_base: int,
    target_ood_share: float,
) -> int:
    if n_ood_base <= 0:
        raise ValueError("Cannot oversample because base OOD train size is zero")
    if not (0.0 < target_ood_share < 1.0):
        raise ValueError("target_ood_share must be in (0, 1)")

    required = (target_ood_share * n_in_dist) / (n_ood_base * (1.0 - target_ood_share))
    return max(1, int(math.ceil(required)))


def _expand_replicas(df: pd.DataFrame, repeats: int) -> pd.DataFrame:
    if repeats <= 1:
        out = df.copy()
        out["replica_idx"] = 0
        return out.reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for rep_idx in range(repeats):
        part = df.copy()
        part["replica_idx"] = rep_idx
        parts.append(part)
    return pd.concat(parts, axis=0, ignore_index=True)


def _image_level_cv_split(
    in_dist_df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    image_dom = _dominant_class_per_image(in_dist_df)
    image_ids = image_dom.index.to_numpy()
    y = image_dom.to_numpy()

    if len(image_ids) < n_folds:
        raise ValueError(
            f"Not enough in_dist images ({len(image_ids)}) for n_folds={n_folds}"
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_ids, y)):
        train_images = set(image_ids[train_idx].tolist())
        val_images = set(image_ids[val_idx].tolist())

        fold_train = in_dist_df[in_dist_df["image_id"].isin(train_images)].copy()
        fold_val = in_dist_df[in_dist_df["image_id"].isin(val_images)].copy()
        folds.append((fold_train.reset_index(drop=True), fold_val.reset_index(drop=True), fold_idx))

    return folds


def _summarize_split(df: pd.DataFrame, name: str) -> dict[str, Any]:
    n_rows = int(len(df))
    n_images = int(df["image_id"].nunique()) if n_rows else 0
    n_ood = int((df["domain"] == "ood").sum()) if "domain" in df.columns else 0
    ood_share = (n_ood / n_rows) if n_rows > 0 else 0.0
    return {
        "split": name,
        "rows": n_rows,
        "images": n_images,
        "ood_rows": n_ood,
        "ood_share": ood_share,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    out_cfg = cfg["output"]

    # Defaults are aligned to the user's in_dist/OOD plan.
    in_dist_csv = Path(data_cfg.get("in_dist_train_csv", data_cfg["train_csv"]))
    in_dist_image_dir = Path(
        data_cfg.get("in_dist_train_image_dir", data_cfg["train_image_dir"])
    )
    ood_csv = Path(
        data_cfg.get(
            "ood_train_csv",
            "/workspace/github/pig_posture_kaggle_project/data/generated/train2_unique_labels.csv",
        )
    )
    ood_image_dir = Path(
        data_cfg.get(
            "ood_train_image_dir",
            "/workspace/github/pig_posture_kaggle_project/data/multiview_pig_posture_recognition/train2_images",
        )
    )

    ood_val_target_postures = int(split_cfg.get("ood_val_target_postures", 250))
    ood_train_target_share = float(split_cfg.get("ood_train_target_share", 0.225))
    seed = int(split_cfg["seed"])
    n_folds = int(split_cfg["n_folds"])

    run_root = ensure_dir(Path(out_cfg["root"]).resolve() / out_cfg["run_name"])
    tracker = RunTracker(
        cfg,
        run_name=f"{out_cfg['run_name']}-prepare-ood",
        group=out_cfg["run_name"],
    )

    prep_root = ensure_dir(run_root / "prepared")
    crops_in_dist = ensure_dir(prep_root / "crops" / "in_dist")
    crops_ood = ensure_dir(prep_root / "crops" / "ood")
    crops_test = ensure_dir(prep_root / "crops" / "test")

    in_dist_meta_path = prep_root / "in_dist_metadata.csv"
    ood_meta_path = prep_root / "ood_metadata.csv"
    test_meta_path = prep_root / "test_metadata.csv"

    logger.info("Preparing in_dist crops from: %s", in_dist_csv)
    in_dist_df = build_crop_metadata(
        csv_path=in_dist_csv,
        image_dir=in_dist_image_dir,
        crops_root=crops_in_dist,
        output_csv_path=in_dist_meta_path,
        img_size=int(data_cfg["img_size"]),
        pad=int(data_cfg["pad"]),
        keep_aspect_ratio=bool(data_cfg.get("keep_aspect_ratio", True)),
        is_train=True,
        n_jobs=int(data_cfg.get("prep_n_jobs", 1)),
    )
    _require_columns(in_dist_df, "in_dist metadata")
    in_dist_df["domain"] = "in_dist"

    logger.info("Preparing OOD crops from: %s", ood_csv)
    ood_df = build_crop_metadata(
        csv_path=ood_csv,
        image_dir=ood_image_dir,
        crops_root=crops_ood,
        output_csv_path=ood_meta_path,
        img_size=int(data_cfg["img_size"]),
        pad=int(data_cfg["pad"]),
        keep_aspect_ratio=bool(data_cfg.get("keep_aspect_ratio", True)),
        is_train=True,
        n_jobs=int(data_cfg.get("prep_n_jobs", 1)),
    )
    _require_columns(ood_df, "ood metadata")
    ood_df["domain"] = "ood"

    logger.info("Preparing test crops from: %s", data_cfg["test_csv"])
    test_df = build_crop_metadata(
        csv_path=data_cfg["test_csv"],
        image_dir=data_cfg["test_image_dir"],
        crops_root=crops_test,
        output_csv_path=test_meta_path,
        img_size=int(data_cfg["img_size"]),
        pad=int(data_cfg["pad"]),
        keep_aspect_ratio=bool(data_cfg.get("keep_aspect_ratio", True)),
        is_train=False,
        n_jobs=int(data_cfg.get("prep_n_jobs", 1)),
    )

    ood_train_df, ood_val_df = _split_ood_by_image(
        ood_df=ood_df,
        val_target_postures=ood_val_target_postures,
        seed=seed,
    )
    logger.info(
        "OOD image split complete. train rows=%d, val rows=%d (target approx=%d)",
        len(ood_train_df),
        len(ood_val_df),
        ood_val_target_postures,
    )

    split_root = ensure_dir(run_root / "splits")

    # Single split: train = in_dist + oversampled ood_train, val = fixed ood_val.
    single_train_base = pd.concat([in_dist_df, ood_train_df], axis=0, ignore_index=True)
    single_val_df = ood_val_df.copy().reset_index(drop=True)

    single_repeats = _compute_repeat_factor(
        n_in_dist=len(in_dist_df),
        n_ood_base=len(ood_train_df),
        target_ood_share=ood_train_target_share,
    )
    ood_single_expanded = _expand_replicas(ood_train_df, repeats=single_repeats)

    single_train_df = pd.concat(
        [in_dist_df.assign(replica_idx=0), ood_single_expanded],
        axis=0,
        ignore_index=True,
    )

    single_split_dir = ensure_dir(split_root / "single")
    single_train_df.to_csv(single_split_dir / "train.csv", index=False)
    single_val_df.to_csv(single_split_dir / "val.csv", index=False)

    yolo_single_root = ensure_dir(run_root / "yolo_data" / "single")
    _materialize_yolo_classification_dir_with_replicas(single_train_df, yolo_single_root / "train")
    _materialize_yolo_classification_dir_with_replicas(single_val_df.assign(replica_idx=0), yolo_single_root / "val")

    # CV folds: image-level stratified split on in_dist; every fold uses the same OOD val set.
    folds = _image_level_cv_split(in_dist_df=in_dist_df, n_folds=n_folds, seed=seed)
    cv_root = ensure_dir(split_root / "cv")
    yolo_cv_root = ensure_dir(run_root / "yolo_data" / "cv")

    cv_summaries: list[dict[str, Any]] = []
    for fold_in_train, _fold_in_val_unused, fold_idx in folds:
        fold_train_repeats = _compute_repeat_factor(
            n_in_dist=len(fold_in_train),
            n_ood_base=len(ood_train_df),
            target_ood_share=ood_train_target_share,
        )
        fold_ood_expanded = _expand_replicas(ood_train_df, repeats=fold_train_repeats)

        fold_train_df = pd.concat(
            [fold_in_train.assign(replica_idx=0), fold_ood_expanded],
            axis=0,
            ignore_index=True,
        )
        fold_val_df = ood_val_df.copy().assign(replica_idx=0).reset_index(drop=True)

        fold_dir = ensure_dir(cv_root / f"fold_{fold_idx}")
        fold_train_df.to_csv(fold_dir / "train.csv", index=False)
        fold_val_df.to_csv(fold_dir / "val.csv", index=False)

        yolo_fold_root = ensure_dir(yolo_cv_root / f"fold_{fold_idx}")
        _materialize_yolo_classification_dir_with_replicas(fold_train_df, yolo_fold_root / "train")
        _materialize_yolo_classification_dir_with_replicas(fold_val_df, yolo_fold_root / "val")

        cv_summaries.append(_summarize_split(fold_train_df, f"cv_fold_{fold_idx}_train"))
        cv_summaries.append(_summarize_split(fold_val_df, f"cv_fold_{fold_idx}_val"))

    summary_rows = [
        _summarize_split(in_dist_df, "in_dist_all"),
        _summarize_split(ood_df, "ood_all"),
        _summarize_split(ood_train_df, "ood_train_base"),
        _summarize_split(ood_val_df, "ood_val_fixed"),
        _summarize_split(single_train_base, "single_train_base"),
        _summarize_split(single_train_df, "single_train_oversampled"),
        _summarize_split(single_val_df, "single_val"),
    ] + cv_summaries

    summary_df = pd.DataFrame(summary_rows)
    summary_path = run_root / "prepare_ood_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save useful class/domain distributions for sanity checks.
    dist_dir = ensure_dir(run_root / "splits" / "distributions")
    single_train_df.groupby(["domain", "class_id"]).size().rename("count").reset_index().to_csv(
        dist_dir / "single_train_domain_class_counts.csv", index=False
    )
    single_val_df.groupby(["domain", "class_id"]).size().rename("count").reset_index().to_csv(
        dist_dir / "single_val_domain_class_counts.csv", index=False
    )

    # Persist fixed OOD validation identities to make fold consistency explicit.
    ood_val_images_path = split_root / "ood_val_images.txt"
    val_images = sorted(ood_val_df["image_id"].unique().tolist())
    ood_val_images_path.write_text("\n".join(val_images) + "\n", encoding="utf-8")

    tracker.log(
        {
            "prepare_ood/in_dist_rows": len(in_dist_df),
            "prepare_ood/ood_rows": len(ood_df),
            "prepare_ood/ood_train_rows": len(ood_train_df),
            "prepare_ood/ood_val_rows": len(ood_val_df),
            "prepare_ood/single_train_rows": len(single_train_df),
            "prepare_ood/single_val_rows": len(single_val_df),
            "prepare_ood/single_ood_repeat_factor": single_repeats,
            "prepare_ood/n_folds": n_folds,
            "prepare_ood/test_rows": len(test_df),
        }
    )
    tracker.log_file("prepare_ood_summary", summary_path)
    tracker.log_file("prepare_ood_ood_val_images", ood_val_images_path)
    tracker.finish()

    logger.info("Prepared OOD-aware data at: %s", run_root)


if __name__ == "__main__":
    random.seed(0)
    main()
