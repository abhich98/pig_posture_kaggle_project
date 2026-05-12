from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

import cv2
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

from pig_pipeline.data.schema import REQUIRED_TEST_COLUMNS, REQUIRED_TRAIN_COLUMNS, parse_bbox_string


def _safe_crop(img: np.ndarray, x: float, y: float, w: float, h: float, pad: int) -> np.ndarray:
    """
    Safely crop an image with padding.

    Args:
        img (np.ndarray): The input image, assumed to be in RGB format.
        x (float): The x-coordinate of the top-left corner of the bounding box.
        y (float): The y-coordinate of the top-left corner of the bounding box.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
        pad (int): The padding to apply around the bounding box.

    Returns:
        np.ndarray: The cropped image.
    """
    height, width = img.shape[:2]
    x1 = int(max(0, np.floor(x) - pad))
    y1 = int(max(0, np.floor(y) - pad))
    x2 = int(min(width, np.ceil(x + w) + pad))
    y2 = int(min(height, np.ceil(y + h) + pad))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((32, 32, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _validate_columns(df: pd.DataFrame, is_train: bool) -> None:
    required = REQUIRED_TRAIN_COLUMNS if is_train else REQUIRED_TEST_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def build_crop_metadata(
    csv_path: str | Path,
    image_dir: str | Path,
    crops_root: str | Path,
    output_csv_path: str | Path,
    img_size: int = 224,
    pad: int = 10,
    is_train: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)
    crops_root = Path(crops_root)
    output_csv_path = Path(output_csv_path)

    crops_root.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    _validate_columns(df, is_train=is_train)

    def process_group(image_id: str, group_df: pd.DataFrame) -> list[dict[str, Any]]:
        image_path = image_dir / str(image_id)
        image = _read_rgb(image_path)
        group_rows: list[dict[str, Any]] = []

        for _, row in group_df.iterrows():
            bbox = parse_bbox_string(str(row["bbox"]))
            crop = _safe_crop(image, bbox.x, bbox.y, bbox.w, bbox.h, pad=pad)
            crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)

            row_id = str(row["row_id"])
            class_id = int(row["class_id"]) if is_train else -1

            crop_name = f"{row_id}.png"
            crop_path = crops_root / crop_name
            ok = cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Failed to write crop to {crop_path}")

            group_rows.append(
                {
                    "row_id": row_id,
                    "image_id": str(row["image_id"]),
                    "class_id": class_id,
                    "bbox": str(row["bbox"]),
                    "crop_path": str(crop_path),
                }
            )
        return group_rows

    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0. Use 1 for single-thread or -1 for all CPUs.")

    grouped = list(df.groupby("image_id", sort=False))
    results = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
        delayed(process_group)(str(image_id), group_df)
        for image_id, group_df in tqdm(grouped, total=len(grouped), desc=f"CropsParallel:{csv_path.name}")
    )
    rows: list[dict[str, Any]] = [row for group_rows in results for row in group_rows]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv_path, index=False)
    return out_df


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


def materialize_yolo_classification_dir(
    split_df: pd.DataFrame,
    out_root: str | Path,
    n_jobs: int = 1,
) -> None:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0. Use 1 for single-thread or -1 for all CPUs.")

    records: list[tuple[int, Path, Path]] = []
    seen_dsts: set[Path] = set()
    for row in split_df.itertuples(index=False):
        class_id = int(getattr(row, "class_id"))
        src = Path(str(getattr(row, "crop_path")))
        dst = out_root / str(class_id) / src.name
        if dst in seen_dsts:
            raise ValueError(f"Duplicate destination path detected during materialization: {dst}")
        seen_dsts.add(dst)
        records.append((class_id, src, dst))

    Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
        delayed(_safe_link_or_copy)(src, dst)
        for _, src, dst in tqdm(records, total=len(records), desc=f"Materialize:{out_root.name}")
    )
