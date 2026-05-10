# Multi-view Pig Posture Recognition — Dataset Reference

Source: https://www.kaggle.com/competitions/multi-view-pig-posture-recognition/data
License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

---

## Dataset Description

A Computer Vision Challenge in Precision Livestock Farming.

The dataset provides multi-view images of pigs in farm pens, annotated with bounding boxes and posture class labels. The goal is instance-level posture classification.

---

## Dataset Challenges: Generalization Across Cameras and Animals

The dataset is intentionally designed to evaluate model generalization under realistic farm conditions. The primary challenge arises from distribution shifts between training and testing data, including:

- Changes in camera viewpoints
- Scene geometry
- Illumination
- Time period
- Animal appearance

**Training Set 1 (T1)** — Most challenging setting. Models trained on T1 have never seen the camera viewpoints or pig batches used in the test set, requiring robust cross-view and cross-animal generalization.

**Training Set 2 (T2)** — Relaxes the constraint by adding a small number of images (~4.5% of the test set size) captured from the same camera viewpoints as the test set. No test-set pigs are included. Provides limited camera-view context without introducing subject overlap.

The performance gap between T1 and T2 highlights the difficulty of viewpoint generalization and emphasizes the importance of learning posture representations robust to camera-dependent variations.

---

## Dataset Folder and File Structure

```
multiview_pig_posture_recognition/
├── train1_images/      # Training Set 1 images (3,090 images)
│   ├── pen1_orb_cam1_20250108_085204.jpg
│   └── ...
├── train2_images/      # Training Set 2 images (3,150 images)
│   ├── pen1_orb_cam1_20250108_085204.jpg
│   └── ...
├── test_images/        # Test images (1,350 images)
│   ├── pen1_tur_cam1_20250920_174649.jpg
│   └── ...
├── train1.csv          # Training Set 1 annotations (bbox + posture labels)
├── train2.csv          # Training Set 2 annotations (bbox + posture labels)
├── test.csv            # Test annotations (bbox only, no labels)
├── sample_submission.csv
└── pig_posture_classes.txt   # Mapping of class IDs to posture names
```

**Total:** 7,595 files, ~2 GB, jpg + csv + txt

---

## Image Filename Pattern

```
pen-number_camera-type_camera-number_date_time.jpg
```

Example: `pen1_orb_cam1_20250108_085204.jpg`

| Component       | Description                               | Example           |
|-----------------|-------------------------------------------|-------------------|
| pen-number      | Pig pen identifier                        | pen1, pen2        |
| camera-type     | Camera model (orb = Orbbec, tur = Turret) | orb, tur          |
| camera-number   | Camera index                              | cam1, cam2        |
| date_time       | Capture timestamp (YYYYMMDD_HHMMSS)       | 20250108_085204   |
| file-extension  | Image file format                         | .jpg              |

Camera IDs present in the dataset: `tur_cam1`, `tur_cam2`, `orb_cam1`, `orb_cam2`

---

## Training Annotations (`train1.csv`, `train2.csv`)

Each row = one pig instance (one bounding box) in one image.

| Column       | Description                                         |
|--------------|-----------------------------------------------------|
| row_id       | Unique identifier for each pig instance             |
| image_id     | Image filename (e.g., pen1_orb_cam1_20250108_085204.jpg) |
| width        | Original image width in pixels                      |
| height       | Original image height in pixels                     |
| bbox         | Bounding box: `[xmin, ymin, w, h]` in absolute pixels |
| class_id     | Posture class ID (0–4)                              |

### Example row

```
row_id,image_id,width,height,bbox,class_id
train_pen1_orb_cam1_20250108_085204_0000,pen1_orb_cam1_20250108_085204.jpg,1920,1080,"[967.5,331.5,463.0,447.0]",3
```

### row_id breakdown

```
train_pen1_orb_cam1_20250108_085204_0000
 ↑     ↑    ↑    ↑   ↑        ↑      ↑
split  pen  cam  num date     time   instance_id
[0]   [1]  [2] [3]  [4]      [5]    [6]
```

- `split`: train or test
- `pen_id`: e.g. pen1
- `cam_id`: e.g. orb_cam1 (parts[2] + "_" + parts[3])
- `date`: YYYYMMDD
- `time`: HHMMSS
- `instance_id`: zero-padded sequential ID per image

---

## Test Set (`test.csv`)

Same format as training CSVs but **no `class_id` column**. Bounding boxes are provided; participants predict posture class for each `row_id`.

---

## Pig Posture Classes

Defined in `pig_posture_classes.txt`. Class ID = line index (0-based).

| class_id | Posture Name         |
|----------|----------------------|
| 0        | Lateral_lying_left   |
| 1        | Lateral_lying_right  |
| 2        | Sitting              |
| 3        | Standing             |
| 4        | Sternal_lying        |

Note: Class `10` appears in some annotation tools as `UNKNOWN` (not an official competition class).

---

## Evaluation Metric

**Macro-averaged F1 score** across the 5 posture classes.

- A prediction is correct if the predicted `class_id` matches the ground-truth label for that `row_id`.
- Macro-F1 gives equal weight to all classes, making it robust to class imbalance.

---

## Submission Format

File: `sample_submission.csv`

| Column   | Description                             |
|----------|-----------------------------------------|
| row_id   | Unique pig instance ID from test.csv    |
| class_id | Predicted posture class (integer 0–4)   |

Rules:
- All `row_id` values must exactly match those in `test.csv`
- Each `row_id` must appear exactly once

### Example

```
row_id,class_id
test_pen1_tur_cam1_20250920_174649_0000,0
test_pen1_tur_cam1_20250920_174649_0001,1
```

---

## Notes

- Bounding boxes are axis-aligned, in absolute pixel units, **not normalized**
- Multiple pigs may appear in a single image
- Participants do not predict bounding boxes — task is instance-level posture classification only
- Train and test sets come from different time periods and different camera setups
