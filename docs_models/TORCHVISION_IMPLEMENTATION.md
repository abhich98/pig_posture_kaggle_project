# Torchvision Model Implementation

This document summarizes how the torchvision + PyTorch Lightning pipeline is implemented in this repository.

## Scope

- Backbone options: `convnext_tiny`, `efficientnet_v2_s`, `resnet50`
- Task: multi-class posture classification from prepared crop images
- Training entry points:
  - `scripts/run_torchvision_single.py`
  - `scripts/run_torchvision_cv.py`
  - `scripts/run_torchvision_ood_val.py`

## Data Flow

1. Data preparation writes split CSVs and crop paths.
2. Training scripts load split files from generated run folders.
3. `CropClassificationDataset` reads `crop_path` and `class_id`.
4. `CropDataModule` builds train/validation dataloaders.
5. Lightning module trains and validates with macro-F1-based checkpoint selection.
6. Best checkpoint is reloaded for final validation and test prediction.

## Model Construction

Implemented in `pig_pipeline/training/torchvision.py`.

- `convnext_tiny`
  - Pretrained weights: `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
  - Head replacement: keep classifier norm/flatten, then `Dropout`, then `Linear(num_classes)`

- `efficientnet_v2_s`
  - Pretrained weights: `EfficientNet_V2_S_Weights.IMAGENET1K_V1`
  - Head replacement: classifier `Dropout` + `Linear(num_classes)`

- `resnet50`
  - Pretrained weights: `ResNet50_Weights.IMAGENET1K_V2`
  - Head replacement: `Dropout` + `Linear(num_classes)`

## Loss and Metrics

- Loss: cross-entropy with optional:
  - label smoothing (`train.label_smoothing`)
  - class weights via `cls_pw` (inverse-frequency weighting power)
- Logged metrics (epoch-level):
  - `train_loss`
  - `val_loss`
  - `val_top1`
  - `val_macro_f1`
- Checkpoint monitor: `val_macro_f1` (mode `max`)

## Augmentation and Evaluation Transform

Train transform includes:

- random resized crop
- random affine (rotation/translate/scale/shear) for camera viewpoint shift
- color jitter (with probability)
- random gamma for exposure change
- random autocontrast and random sharpness adjustment
- gaussian blur (with probability)
- random perspective
- normalization (ImageNet mean/std)
- random erasing

Validation/test transform:

- resize to `round(img_size * eval_resize_ratio)`
- center crop to `img_size`
- normalization (ImageNet mean/std)

## Precision and Hardware

- Precision configurable via `train.precision`
  - recommended default on GPU: `16-mixed`
- Tensor Core optimization enabled in runner scripts via:
  - `torch.set_float32_matmul_precision("high")`

## LR Strategy and Freezing (New)

Implemented in `LitImageClassifier.configure_optimizers()` with config-controlled behavior.

### 1) Single LR

- `train.lr_strategy: single`
- One optimizer group for all trainable params.

### 2) Backbone vs Head LR

- `train.lr_strategy: backbone_head`
- Head LR: `train.lr`
- Backbone LR: `train.lr * train.lr_backbone_factor`

### 3) Layer-wise LR

- `train.lr_strategy: layerwise`
- Head LR: `train.lr`
- Backbone stages are split into early/mid/late groups.
- Effective LR:
  - late: `lr * d`
  - mid: `lr * d^2`
  - early: `lr * d^3`
  - where `d = train.lr_layer_decay`

### Freeze Backbone Warmup

- `train.freeze_backbone_epochs: N`
- For first `N` epochs, backbone params are frozen (`requires_grad=False`), head trains only.
- Backbone is automatically unfrozen after warmup.

## Recommended Starting Presets

### Stable baseline

```yaml
train:
  lr_strategy: "single"
  freeze_backbone_epochs: 0
```

### Strong transfer-learning baseline

```yaml
train:
  lr_strategy: "backbone_head"
  lr: 0.0003
  lr_backbone_factor: 0.1
  freeze_backbone_epochs: 2
```

### Aggressive fine-tuning

```yaml
train:
  lr_strategy: "layerwise"
  lr: 0.0003
  lr_layer_decay: 0.3
  freeze_backbone_epochs: 2
```

## Main Config Keys

From `configs/torchvision_train_base.yaml`:

- Model:
  - `model.name`
  - `model.pretrained`
  - `model.dropout`
- Train:
  - `train.epochs`, `train.batch`, `train.workers`
  - `train.lr`, `train.weight_decay`
  - `train.label_smoothing`
  - `train.cls_pw`
  - `train.precision`
  - `train.accumulate_grad_batches`
  - `train.early_stopping_patience`, `train.early_stopping_min_delta`
  - `train.lr_strategy`
  - `train.lr_backbone_factor`
  - `train.lr_layer_decay`
  - `train.freeze_backbone_epochs`
- Inference:
  - `inference.batch`, `inference.workers`, `inference.imgsz`

## Outputs

### Single strategy

- best/last checkpoints in run folder
- metrics JSON
- confusion matrix and per-class plot
- submission CSV (`row_id`, `class_id`)

### CV strategy

- per-fold checkpoints and metrics
- fold-level plots
- OOF predictions and OOF metrics
- calibrated ensemble submission CSV

## Notes

- Current validation aggregation logs are epoch-level and designed for model selection via macro-F1.
- If training with multiple GPUs and strict globally-synchronized validation metrics is needed, metric synchronization behavior should be verified for the specific strategy run.
