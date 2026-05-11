# Pig Posture Classification Pipeline

This repository now includes a reusable data pipeline for posture classification experiments, starting with two YOLO strategies:

1. Train2 single split baseline.
2. Train2 5-fold cross-validation with logits averaging ensemble.

The pipeline is designed so data preparation and split manifests can be reused by future non-YOLO PyTorch models.

## Implemented Structure

- `pig_pipeline/data`: bbox parsing, crop generation, split creation.
- `pig_pipeline/training`: YOLO train/eval/inference helpers and metrics.
- `pig_pipeline/tracking.py`: W&B run tracking utility.
- `scripts/prepare_data.py`: prepare crops, split files, and YOLO-ready folders.
- `scripts/run_yolo_single.py`: run single-split training and submission.
- `scripts/run_yolo_cv.py`: run CV training and logits-ensemble submission.
- `configs/`: YAML configs for Kaggle and local runs.

## Quick Start

1. Install dependencies (uv):

```bash
uv sync
```

2. Edit local config file paths:

- `configs/yolo_train2_local_example.yaml`

3. Login to W&B once (optional but recommended):

```bash
wandb login
```

4. Prepare data and manifests:

```bash
python scripts/prepare_data.py --config configs/yolo_train2_local_example.yaml
```

5. Run single split strategy:

```bash
python scripts/run_yolo_single.py --config configs/yolo_train2_local_example.yaml
```

6. Run 5-fold CV strategy:

```bash
python scripts/run_yolo_cv.py --config configs/yolo_train2_local_example.yaml
```

## Outputs

All generated files are placed under:

`outputs/<run_name>/`

Including:

- Prepared crop metadata.
- Split CSVs.
- YOLO train/val directory views.
- Strategy submissions and metrics.

## Notes

- Model selection metric is Macro-F1 on validation (Top1 is also tracked).
- CV ensemble prediction uses average logits then argmax.
- Split strategy is row-level stratified, as currently specified.
