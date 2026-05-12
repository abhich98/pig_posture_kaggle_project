from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from pig_pipeline.training.metrics import macro_f1, per_class_report


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("yolo_training")


def load_yolo_model(weights: str) -> YOLO:
    try:
        return YOLO(weights)
    except Exception:
        logger.warning(f"Failed to load model with weights '{weights}', falling back to 'yolov8n-cls.pt'.")
        if weights != "yolov8n-cls.pt":
            return YOLO("yolov8n-cls.pt")
        raise


def train_classifier(
    model: YOLO,
    dataset_dir: str | Path,
    train_args: dict[str, Any],
) -> Path:
    results = model.train(data=str(dataset_dir), **train_args)
    _ = results
    return Path(model.trainer.best)


def _predict_single(model: YOLO, image_path: str | Path) -> tuple[int, np.ndarray]:
    out = model.predict(source=str(image_path), verbose=False)
    top1 = int(out[0].probs.top1)
    probs = out[0].probs.data.detach().cpu().numpy()
    return top1, probs


def evaluate_on_split(model: YOLO, val_df: pd.DataFrame) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validate"):
        pred, _ = _predict_single(model, row["crop_path"])
        y_true.append(int(row["class_id"]))
        y_pred.append(pred)

    top1 = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro = macro_f1(y_true, y_pred)
    report = per_class_report(y_true, y_pred)
    return {
        "top1": top1,
        "macro_f1": macro,
        "report": report,
    }


def predict_test_top1(model: YOLO, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[str, int]] = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predict test"):
        pred, _ = _predict_single(model, row["crop_path"])
        rows.append((str(row["row_id"]), int(pred)))
    return pd.DataFrame(rows, columns=["row_id", "class_id"])


def predict_test_probs(model: YOLO, test_df: pd.DataFrame) -> np.ndarray:
    """Return softmax probabilities for every test sample, shape (n_samples, n_classes)."""
    probs_list: list[np.ndarray] = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predict probs"):
        _, probs = _predict_single(model, row["crop_path"])
        probs_list.append(probs)
    return np.stack(probs_list, axis=0)


def collect_val_probs(model: YOLO, val_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Collect softmax probabilities and true labels on the validation split.

    Returns
    -------
    probs : np.ndarray, shape (n_samples, n_classes)
    y_true : np.ndarray, shape (n_samples,)
    """
    probs_list: list[np.ndarray] = []
    y_true: list[int] = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Collect val probs"):
        _, probs = _predict_single(model, row["crop_path"])
        probs_list.append(probs)
        y_true.append(int(row["class_id"]))
    return np.stack(probs_list, axis=0), np.array(y_true)


def calibrate_probs(
    val_probs: np.ndarray,
    y_true: np.ndarray,
    test_probs: np.ndarray,
) -> np.ndarray:
    """Fit per-fold temperature scaling on validation data and return calibrated test probs.

    Parameters
    ----------
    val_probs : np.ndarray, shape (n_val, n_classes) — softmax probs on validation set.
    y_true    : np.ndarray, shape (n_val,)            — integer ground-truth labels.
    test_probs: np.ndarray, shape (n_test, n_classes) — softmax probs on test set.

    Returns
    -------
    np.ndarray, shape (n_test, n_classes) — calibrated probabilities.
    """
    from netcal.scaling import TemperatureScaling

    calibrator = TemperatureScaling()
    calibrator.fit(val_probs, y_true, tensorboard=False)
    return calibrator.transform(test_probs, mean_estimate=True)
