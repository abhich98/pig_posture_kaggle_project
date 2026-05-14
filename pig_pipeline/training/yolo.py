from __future__ import annotations

import gc
from copy import copy
from pathlib import Path
from typing import Any
import logging

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.classify.val import ClassificationValidator

from pig_pipeline.training.metrics import macro_f1
from pig_pipeline.training.utills import calibrate_probs, compute_classification_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("yolo_training")


def flush_inference_memory_2() -> None:
    """Release Python and CUDA cached memory between large inference phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

def flush_inference_memory() -> None:
    """Release Python and CUDA cached memory when multiple GPUs are in use."""
    gc.collect()
    if not torch.cuda.is_available():
        return

    device_count = torch.cuda.device_count()
    if device_count <= 1:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        return

    current_device = torch.cuda.current_device()
    try:
        for idx in range(device_count):
            torch.cuda.set_device(idx)
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    finally:
        torch.cuda.set_device(current_device)


class F1ClassificationValidator(ClassificationValidator):
    """Ultralytics classification validator that tracks macro F1 and uses it as fitness."""

    def get_desc(self) -> str:
        return ("%22s" + "%11s" * 3) % ("classes", "top1_acc", "top5_acc", "macro_f1")

    def get_stats(self) -> dict[str, float]:
        stats = super().get_stats()

        targets = np.concatenate([t.numpy().reshape(-1) for t in self.targets]).astype(int)
        preds_topk = np.concatenate([p.numpy() for p in self.pred], axis=0)
        preds_top1 = preds_topk[:, 0].astype(int)

        macro_f1_value = macro_f1(targets.tolist(), preds_top1.tolist())
        stats["metrics/macro_f1"] = macro_f1_value
        stats["fitness"] = macro_f1_value
        return stats

    def print_results(self) -> None:
        """Print results with macro_f1 added to standard metrics."""
        # Let parent print the standard top1/top5 accuracy
        super().print_results()
        # Then log our custom macro_f1 metric
        results = self.get_stats()
        f1 = results.get("metrics/macro_f1", 0.0)
        LOGGER = logging.getLogger("yolo_training")
        LOGGER.info(f"macro_f1: {f1:.3g}")


class F1ClassificationTrainer(ClassificationTrainer):
    """Ultralytics classification trainer that selects best model by macro F1."""

    def get_validator(self):
        self.loss_names = ["loss"]
        return F1ClassificationValidator(
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


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
    resolved_train_args = dict(train_args)
    use_f1_selection = bool(resolved_train_args.pop("use_f1_selection", False))

    if use_f1_selection:
        results = model.train(trainer=F1ClassificationTrainer, data=str(dataset_dir), **resolved_train_args)
    else:
        results = model.train(data=str(dataset_dir), **resolved_train_args)

    _ = results
    return Path(model.trainer.best)


def _predict_single(model: YOLO, image_path: str | Path) -> tuple[int, np.ndarray]:
    out = model.predict(source=str(image_path), verbose=False)
    top1 = int(out[0].probs.top1)
    probs = out[0].probs.data.detach().cpu().numpy()
    return top1, probs


def _predict_batch(
    model: YOLO,
    image_paths: list[str | Path],
    chunk: int = 1000,
    imgsz: int = 320,
    batch: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched GPU inference over a list of image paths.

    Returns
    -------
    top1  : np.ndarray, shape (n,)           — top-1 class index per image.
    probs : np.ndarray, shape (n, n_classes) — softmax probabilities per image.
    """
    # paths = [str(p) for p in image_paths]
    # top1_list: list[int] = []
    # probs_list: list[np.ndarray] = []
    # for r in model.predict(source=paths, verbose=False, batch=batch, stream=True):
    #     top1_list.append(int(r.probs.top1))
    #     probs_list.append(r.probs.data.detach().cpu().numpy())
    # return np.array(top1_list), np.stack(probs_list, axis=0)
    paths = [str(p) for p in image_paths]
    top1_list = []
    probs_list = []
    
    # 1. Process in smaller "chunks" to prevent massive preprocessing allocation
    for i in range(0, len(paths), chunk):
        chunk_paths = paths[i : i + chunk]
        
        # Use torch.no_grad() to ensure no gradient history is saved
        with torch.no_grad():
            for r in model.predict(source=chunk_paths, verbose=False, imgsz=imgsz, batch=batch, stream=True):
                top1_list.append(int(r.probs.top1))
                probs_list.append(r.probs.data.detach().cpu().numpy())
        
        # 2. Periodic Memory Clearing
        flush_inference_memory()
        gc.collect()

    return np.array(top1_list), np.stack(probs_list, axis=0)


def evaluate_on_split(model: YOLO, val_df: pd.DataFrame, inf_args: dict[str, Any]) -> dict[str, Any]:
    paths = val_df["crop_path"].tolist()
    y_true = val_df["class_id"].astype(int).tolist()

    logger.info(f"Evaluating {len(paths)} validation samples (batch={inf_args.get('batch', 64)})...")
    y_pred_arr, _ = _predict_batch(model, paths, **inf_args)
    y_pred = y_pred_arr.tolist()

    return compute_classification_metrics(y_true, y_pred)


def predict_test_top1(model: YOLO, test_df: pd.DataFrame, inf_args: dict[str, Any]) -> pd.DataFrame:
    paths = test_df["crop_path"].tolist()
    logger.info(f"Predicting top-1 for {len(paths)} test samples (batch={inf_args.get('batch', 64)})...")
    top1_arr, _ = _predict_batch(model, paths, **inf_args)
    return pd.DataFrame({"row_id": test_df["row_id"].astype(str).tolist(), "class_id": top1_arr.tolist()})


def predict_test_probs(model: YOLO, test_df: pd.DataFrame, inf_args: dict[str, Any]) -> np.ndarray:
    """Return softmax probabilities for every test sample, shape (n_samples, n_classes)."""
    paths = test_df["crop_path"].tolist()
    logger.info(f"Predicting probs for {len(paths)} test samples (batch={inf_args.get('batch', 64)})...")
    _, probs = _predict_batch(model, paths, **inf_args)
    return probs


def collect_val_probs(model: YOLO, val_df: pd.DataFrame, inf_args: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Collect softmax probabilities and true labels on the validation split.

    Returns
    -------
    probs : np.ndarray, shape (n_samples, n_classes)
    y_true : np.ndarray, shape (n_samples,)
    """
    paths = val_df["crop_path"].tolist()
    y_true = val_df["class_id"].astype(int).to_numpy()
    logger.info(f"Collecting val probs for {len(paths)} samples (batch={inf_args.get('batch', 64)})...")
    _, probs = _predict_batch(model, paths, **inf_args)
    return probs, y_true
