from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from pig_pipeline.training.metrics import macro_f1
from pig_pipeline.training.utills import (
    flush_torch_memory,
    compute_classification_metrics,
)

logger = logging.getLogger("torchvision_training")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CropClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms.Compose,
        with_labels: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(str(row["crop_path"]))
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        x = self.transform(image)

        if self.with_labels:
            y = int(row["class_id"])
            return x, y

        return x


class CropPathDataset(Dataset):
    def __init__(
        self, image_paths: list[str | Path], transform: transforms.Compose
    ) -> None:
        self.image_paths = [Path(str(p)) for p in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image)


class CropDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        train_transform: transforms.Compose,
        eval_transform: transforms.Compose,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self.train_ds: CropClassificationDataset | None = None
        self.val_ds: CropClassificationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        self.train_ds = CropClassificationDataset(
            self.train_df, self.train_transform, with_labels=True
        )
        self.val_ds = CropClassificationDataset(
            self.val_df, self.eval_transform, with_labels=True
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )


def _build_backbone(
    model_name: str, num_classes: int, pretrained: bool, dropout: float
) -> nn.Module:
    name = model_name.lower().replace("-", "_")

    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            nn.Dropout(p=float(dropout), inplace=False),
            nn.Linear(in_features, num_classes),
        )
        return model

    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[0] = nn.Dropout(p=float(dropout), inplace=True)
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=float(dropout)), nn.Linear(in_features, num_classes)
        )
        return model

    raise ValueError(
        f"Unsupported model_name='{model_name}'. Supported: convnext_tiny, efficientnet_v2_s, resnet50"
    )


class LitImageClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
        lr: float,
        weight_decay: float,
        label_smoothing: float,
        max_epochs: int,
        lr_strategy: str = "single",
        lr_backbone_factor: float = 0.1,
        lr_layer_decay: float = 0.3,
        freeze_backbone_epochs: int = 0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_name = str(model_name).lower().replace("-", "_")

        self.model = _build_backbone(
            model_name=model_name,
            num_classes=int(num_classes),
            pretrained=bool(pretrained),
            dropout=float(dropout),
        )

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.label_smoothing = float(label_smoothing)
        self.max_epochs = int(max_epochs)
        self.lr_strategy = str(lr_strategy).strip().lower().replace("-", "_")
        self.lr_backbone_factor = float(lr_backbone_factor)
        self.lr_layer_decay = float(lr_layer_decay)
        self.freeze_backbone_epochs = int(freeze_backbone_epochs)

        if self.lr_strategy not in {"single", "backbone_head", "layerwise"}:
            raise ValueError(
                "train.lr_strategy must be one of: single, backbone_head, layerwise"
            )
        if self.lr_backbone_factor <= 0.0:
            raise ValueError("train.lr_backbone_factor must be > 0")
        if not (0.0 < self.lr_layer_decay <= 1.0):
            raise ValueError("train.lr_layer_decay must be in (0, 1]")
        if self.freeze_backbone_epochs < 0:
            raise ValueError("train.freeze_backbone_epochs must be >= 0")

        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor, persistent=True)
        else:
            self.class_weights = None

        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._backbone_frozen = False

        if self.freeze_backbone_epochs > 0:
            self._set_backbone_trainable(False)
            logger.info(
                "Backbone frozen for first %d epoch(s)",
                self.freeze_backbone_epochs,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _ = batch_idx
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits,
            y,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _ = batch_idx
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(
            logits, y, weight=self.class_weights, label_smoothing=self.label_smoothing
        )
        preds = torch.argmax(logits, dim=1)

        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_targets:
            return

        y_pred = torch.cat(self._val_preds).numpy()
        y_true = torch.cat(self._val_targets).numpy()
        self._val_preds.clear()
        self._val_targets.clear()

        top1 = float(np.mean(y_pred == y_true))
        macro = macro_f1(y_true.tolist(), y_pred.tolist())

        self.log(
            "val_top1",
            top1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            "val_macro_f1",
            macro,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

    def on_train_epoch_start(self) -> None:
        should_freeze = self.current_epoch < self.freeze_backbone_epochs
        if should_freeze and not self._backbone_frozen:
            self._set_backbone_trainable(False)
            logger.info("Backbone re-frozen at epoch %d", self.current_epoch)
        elif not should_freeze and self._backbone_frozen:
            self._set_backbone_trainable(True)
            logger.info("Backbone unfrozen at epoch %d", self.current_epoch)

    def _is_head_param_name(self, name: str) -> bool:
        if self.model_name in {"convnext_tiny", "efficientnet_v2_s"}:
            return name.startswith("model.classifier")
        if self.model_name == "resnet50":
            return name.startswith("model.fc")
        return False

    def _set_backbone_trainable(self, trainable: bool) -> None:
        for name, param in self.named_parameters():
            if not name.startswith("model."):
                continue
            if self._is_head_param_name(name):
                continue
            param.requires_grad = trainable
        self._backbone_frozen = not trainable

    @staticmethod
    def _match_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
        return any(name.startswith(prefix) for prefix in prefixes)

    def _build_param_groups(self) -> list[dict[str, Any]]:
        named_params = list(self.named_parameters())

        if self.lr_strategy == "single":
            return [
                {
                    "name": "all",
                    "params": [p for _, p in named_params],
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            ]

        head_params = [
            p for n, p in named_params if n.startswith("model.") and self._is_head_param_name(n)
        ]
        backbone_params = [
            p for n, p in named_params if n.startswith("model.") and not self._is_head_param_name(n)
        ]
        non_model_params = [p for n, p in named_params if not n.startswith("model.")]

        if self.lr_strategy == "backbone_head":
            groups: list[dict[str, Any]] = []
            if head_params:
                groups.append(
                    {
                        "name": "head",
                        "params": head_params,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
            if backbone_params:
                groups.append(
                    {
                        "name": "backbone",
                        "params": backbone_params,
                        "lr": self.lr * self.lr_backbone_factor,
                        "weight_decay": self.weight_decay,
                    }
                )
            if non_model_params:
                groups.append(
                    {
                        "name": "misc",
                        "params": non_model_params,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
            return groups

        # Layer-wise discriminative LR
        if self.model_name in {"convnext_tiny", "efficientnet_v2_s"}:
            early_prefixes = (
                "model.features.0",
                "model.features.1",
                "model.features.2",
                "model.features.3",
            )
            mid_prefixes = ("model.features.4", "model.features.5")
            late_prefixes = ("model.features.6", "model.features.7")
        elif self.model_name == "resnet50":
            early_prefixes = (
                "model.conv1",
                "model.bn1",
                "model.layer1",
                "model.layer2",
            )
            mid_prefixes = ("model.layer3",)
            late_prefixes = ("model.layer4",)
        else:
            # Safe fallback for unsupported future backbones.
            return [
                {
                    "name": "head",
                    "params": head_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "name": "backbone",
                    "params": backbone_params,
                    "lr": self.lr * self.lr_backbone_factor,
                    "weight_decay": self.weight_decay,
                },
            ]

        early_params = [
            p
            for n, p in named_params
            if n.startswith("model.")
            and not self._is_head_param_name(n)
            and self._match_prefix(n, early_prefixes)
        ]
        mid_params = [
            p
            for n, p in named_params
            if n.startswith("model.")
            and not self._is_head_param_name(n)
            and self._match_prefix(n, mid_prefixes)
        ]
        late_params = [
            p
            for n, p in named_params
            if n.startswith("model.")
            and not self._is_head_param_name(n)
            and self._match_prefix(n, late_prefixes)
        ]

        assigned_ids = {id(p) for p in early_params + mid_params + late_params + head_params}
        other_backbone_params = [
            p
            for n, p in named_params
            if n.startswith("model.") and not self._is_head_param_name(n) and id(p) not in assigned_ids
        ]

        d = self.lr_layer_decay
        groups = []
        if head_params:
            groups.append(
                {
                    "name": "head",
                    "params": head_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            )
        if late_params:
            groups.append(
                {
                    "name": "backbone_late",
                    "params": late_params,
                    "lr": self.lr * d,
                    "weight_decay": self.weight_decay,
                }
            )
        if mid_params:
            groups.append(
                {
                    "name": "backbone_mid",
                    "params": mid_params,
                    "lr": self.lr * (d**2),
                    "weight_decay": self.weight_decay,
                }
            )
        if early_params:
            groups.append(
                {
                    "name": "backbone_early",
                    "params": early_params,
                    "lr": self.lr * (d**3),
                    "weight_decay": self.weight_decay,
                }
            )
        if other_backbone_params:
            groups.append(
                {
                    "name": "backbone_other",
                    "params": other_backbone_params,
                    "lr": self.lr * (d**2),
                    "weight_decay": self.weight_decay,
                }
            )
        if non_model_params:
            groups.append(
                {
                    "name": "misc",
                    "params": non_model_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            )
        return groups

    def configure_optimizers(self):
        param_groups = self._build_param_groups()
        logger.info(
            "Optimizer LR strategy=%s, groups=%s",
            self.lr_strategy,
            [
                {
                    "name": g.get("name", "group"),
                    "lr": g["lr"],
                    "n_params": sum(p.numel() for p in g["params"]),
                }
                for g in param_groups
            ],
        )
        optimizer = AdamW(param_groups)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, self.max_epochs), eta_min=self.lr * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


@dataclass
class TorchvisionTrainArtifacts:
    best_ckpt: Path
    best_score: float | None
    model: LitImageClassifier


def _precision_from_config(value: Any) -> str | int:
    if value is None:
        return "16-mixed" if torch.cuda.is_available() else "32-true"

    if isinstance(value, int):
        return value

    text = str(value).strip().lower()
    if text in {"16", "16-mixed", "fp16", "float16"}:
        return "16-mixed"
    if text in {"bf16", "bf16-mixed"}:
        return "bf16-mixed"
    if text in {"32", "32-true", "fp32", "float32"}:
        return "32-true"

    return str(value)


class RandomGamma:
    """Randomly apply gamma correction to mimic exposure response changes."""

    def __init__(
        self,
        p: float = 0.0,
        gamma_min: float = 0.75,
        gamma_max: float = 1.35,
        gain: float = 1.0,
    ) -> None:
        self.p = float(p)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.gain = float(gain)
        if self.p < 0.0 or self.p > 1.0:
            raise ValueError("augment.gamma_p must be in [0, 1]")
        if self.gamma_min <= 0 or self.gamma_max <= 0:
            raise ValueError("augment.gamma_min and augment.gamma_max must be > 0")
        if self.gamma_min > self.gamma_max:
            raise ValueError("augment.gamma_min must be <= augment.gamma_max")

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0.0 or torch.rand(1).item() >= self.p:
            return image
        gamma = float(torch.empty(1).uniform_(self.gamma_min, self.gamma_max).item())
        return TF.adjust_gamma(image, gamma=gamma, gain=self.gain)


def build_transforms(
    img_size: int, aug_cfg: dict[str, Any] | None = None
) -> tuple[transforms.Compose, transforms.Compose]:
    aug_cfg = aug_cfg or {}
    img_size = int(img_size)
    affine_shear = float(aug_cfg.get("affine_shear", 5.0))

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=img_size,
                scale=(
                    float(aug_cfg.get("scale_min", 0.75)),
                    float(aug_cfg.get("scale_max", 1.0)),
                ),
                ratio=(
                    float(aug_cfg.get("ratio_min", 0.85)),
                    float(aug_cfg.get("ratio_max", 1.2)),
                ),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=float(aug_cfg.get("affine_degrees", 10.0)),
                        translate=(
                            float(aug_cfg.get("affine_translate", 0.1)),
                            float(aug_cfg.get("affine_translate", 0.1)),
                        ),
                        scale=(
                            float(aug_cfg.get("affine_scale_min", 0.9)),
                            float(aug_cfg.get("affine_scale_max", 1.1)),
                        ),
                        shear=(-affine_shear, affine_shear),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                ],
                p=float(aug_cfg.get("affine_p", 0.7)),
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=float(aug_cfg.get("brightness", 0.45)),
                        contrast=float(aug_cfg.get("contrast", 0.4)),
                        saturation=float(aug_cfg.get("saturation", 0.15)),
                        hue=float(aug_cfg.get("hue", 0.03)),
                    )
                ],
                p=float(aug_cfg.get("color_p", 0.8)),
            ),
            RandomGamma(
                p=float(aug_cfg.get("gamma_p", 0.2)),
                gamma_min=float(aug_cfg.get("gamma_min", 0.75)),
                gamma_max=float(aug_cfg.get("gamma_max", 1.35)),
            ),
            transforms.RandomAutocontrast(p=float(aug_cfg.get("autocontrast_p", 0.12))),
            transforms.RandomAdjustSharpness(
                sharpness_factor=float(aug_cfg.get("sharpness_factor", 1.8)),
                p=float(aug_cfg.get("sharpness_p", 0.12)),
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=float(aug_cfg.get("blur_p", 0.12)),
            ),
            transforms.RandomPerspective(
                distortion_scale=float(aug_cfg.get("perspective", 0.18)),
                p=float(aug_cfg.get("perspective_p", 0.25)),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=float(aug_cfg.get("erasing_p", 0.18)),
                scale=(
                    float(aug_cfg.get("erase_scale_min", 0.02)),
                    float(aug_cfg.get("erase_scale_max", 0.12)),
                ),
                ratio=(
                    float(aug_cfg.get("erase_ratio_min", 0.5)),
                    float(aug_cfg.get("erase_ratio_max", 2.0)),
                ),
            ),
        ]
    )

    resize_size = int(round(img_size * float(aug_cfg.get("eval_resize_ratio", 1.14))))
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (resize_size, resize_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def load_torchvision_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    num_classes: int,
    map_location: str | torch.device = "cpu",
) -> LitImageClassifier:
    return LitImageClassifier.load_from_checkpoint(
        str(checkpoint_path),
        map_location=map_location,
        model_name=str(model_cfg.get("name", "convnext_tiny")),
        num_classes=int(num_classes),
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.2)),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        max_epochs=int(train_cfg.get("epochs", 30)),
        lr_strategy=str(train_cfg.get("lr_strategy", "single")),
        lr_backbone_factor=float(train_cfg.get("lr_backbone_factor", 0.1)),
        lr_layer_decay=float(train_cfg.get("lr_layer_decay", 0.3)),
        freeze_backbone_epochs=int(train_cfg.get("freeze_backbone_epochs", 0)),
    )


def train_torchvision_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    out_dir: str | Path,
    aug_cfg: dict[str, Any] | None = None,
) -> TorchvisionTrainArtifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "class_id" not in train_df.columns:
        raise ValueError("train_df must include class_id")

    num_classes = int(max(train_df["class_id"].max(), val_df["class_id"].max()) + 1)

    cls_pw = float(train_cfg.get("cls_pw", 0.0))
    class_weights: list[float] | None = None
    if cls_pw > 0.0:
        class_counts = train_df["class_id"].astype(int).value_counts().sort_index()
        counts = np.ones(num_classes, dtype=np.float32)
        for class_id, count in class_counts.items():
            counts[int(class_id)] = float(count)

        # YOLO-like behavior: inverse-frequency class weighting with power control.
        # cls_pw=0.0 -> no weighting, cls_pw=1.0 -> full inverse-frequency.
        inv_freq = 1.0 / counts
        weights = np.power(inv_freq, cls_pw)
        weights = weights / max(float(weights.mean()), 1e-12)
        class_weights = weights.astype(np.float32).tolist()
        logger.info(
            "Using class-weighted CE loss with cls_pw=%.3f, weights=%s",
            cls_pw,
            class_weights,
        )

    pl.seed_everything(int(train_cfg.get("seed", 42)), workers=True)

    img_size = int(data_cfg.get("img_size", 224))
    train_transform, eval_transform = build_transforms(
        img_size=img_size, aug_cfg=aug_cfg
    )

    datamodule = CropDataModule(
        train_df=train_df,
        val_df=val_df,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=int(train_cfg.get("batch", 32)),
        num_workers=int(train_cfg.get("workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
    )

    lightning_model = LitImageClassifier(
        model_name=str(model_cfg.get("name", "convnext_tiny")),
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        max_epochs=int(train_cfg.get("epochs", 30)),
        lr_strategy=str(train_cfg.get("lr_strategy", "single")),
        lr_backbone_factor=float(train_cfg.get("lr_backbone_factor", 0.1)),
        lr_layer_decay=float(train_cfg.get("lr_layer_decay", 0.3)),
        freeze_backbone_epochs=int(train_cfg.get("freeze_backbone_epochs", 0)),
        class_weights=class_weights,
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(out_dir),
        filename="best-epoch{epoch:02d}-f1{val_macro_f1:.4f}",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    checkpoint_every_n_epochs = int(train_cfg.get("checkpoint_every_n_epochs", 0))
    periodic_checkpoint_cb: pl.callbacks.ModelCheckpoint | None = None
    if checkpoint_every_n_epochs > 0:
        periodic_checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(out_dir),
            filename="epoch{epoch:02d}",
            save_top_k=-1,
            every_n_epochs=checkpoint_every_n_epochs,
            save_last=False,
        )

    callbacks: list[pl.Callback] = [checkpoint_cb]
    if periodic_checkpoint_cb is not None:
        callbacks.append(periodic_checkpoint_cb)
    patience = int(train_cfg.get("early_stopping_patience", 0))
    if patience > 0:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                mode="max",
                patience=patience,
                min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
            )
        )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        default_root_dir=str(out_dir),
        max_epochs=int(train_cfg.get("epochs", 30)),
        accelerator=accelerator,
        devices="auto",
        precision=_precision_from_config(train_cfg.get("precision", None)),
        accumulate_grad_batches=int(train_cfg.get("accumulate_grad_batches", 1)),
        deterministic=bool(train_cfg.get("deterministic", False)),
        callbacks=callbacks,
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 20)),
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(lightning_model, datamodule=datamodule)

    best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    if not best_path:
        raise RuntimeError("No checkpoint path found after training")

    best_score = None
    if checkpoint_cb.best_model_score is not None:
        best_score = float(checkpoint_cb.best_model_score.detach().cpu().item())

    best_model = load_torchvision_model_from_checkpoint(
        checkpoint_path=best_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        num_classes=num_classes,
        map_location="cpu",
    )
    best_model.eval()

    return TorchvisionTrainArtifacts(
        best_ckpt=Path(best_path), best_score=best_score, model=best_model
    )


def _predict_batch(
    model: LitImageClassifier,
    image_paths: list[str | Path],
    batch: int = 64,
    workers: int = 4,
    imgsz: int = 224,
) -> tuple[np.ndarray, np.ndarray]:

    _, eval_transform = build_transforms(img_size=imgsz, aug_cfg={})
    dataset = CropPathDataset(image_paths=image_paths, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

    flush_torch_memory()

    return np.concatenate(all_preds, axis=0), np.concatenate(all_probs, axis=0)


def evaluate_on_split(
    model: LitImageClassifier,
    val_df: pd.DataFrame,
    inf_args: dict[str, Any],
    return_predictions: bool = False,
) -> dict[str, Any]:
    paths = val_df["crop_path"].tolist()
    y_true = val_df["class_id"].astype(int).tolist()

    logger.info("Evaluating %d validation samples...", len(paths))
    y_pred_arr, _ = _predict_batch(model, paths, **inf_args)
    y_pred = y_pred_arr.tolist()

    return compute_classification_metrics(
        y_true, y_pred, return_predictions=return_predictions
    )


def predict_test_top1(
    model: LitImageClassifier, test_df: pd.DataFrame, inf_args: dict[str, Any]
) -> pd.DataFrame:
    paths = test_df["crop_path"].tolist()
    logger.info("Predicting top-1 for %d test samples...", len(paths))
    top1_arr, _ = _predict_batch(model, paths, **inf_args)
    return pd.DataFrame(
        {
            "row_id": test_df["row_id"].astype(str).tolist(),
            "class_id": top1_arr.tolist(),
        }
    )


def predict_test_probs(
    model: LitImageClassifier, test_df: pd.DataFrame, inf_args: dict[str, Any]
) -> np.ndarray:
    paths = test_df["crop_path"].tolist()
    logger.info("Predicting probabilities for %d test samples...", len(paths))
    _, probs = _predict_batch(model, paths, **inf_args)
    return probs


def collect_val_probs(
    model: LitImageClassifier, val_df: pd.DataFrame, inf_args: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    paths = val_df["crop_path"].tolist()
    y_true = val_df["class_id"].astype(int).to_numpy()
    logger.info("Collecting val probabilities for %d samples...", len(paths))
    _, probs = _predict_batch(model, paths, **inf_args)
    return probs, y_true
