That is actually a very reasonable setup.

A 2GB dataset with ~5000 images is small enough that:

* Colab GPU is sufficient for Phase 1
* training iteration will be fast
* experimentation becomes easier
* you can focus more on architecture + engineering quality instead of infrastructure complexity

Using Colab initially is a good decision. Later, you can migrate the exact same pipeline to AWS SageMaker.

---

# What Phase 1 Should Achieve

Phase 1 is not “training a model”.

It is about building:

* a clean research pipeline
* reproducible experiments
* modular code
* scalable architecture

At the end of Phase 1, you should have:

✅ dataset understanding
✅ proper train/validation strategy
✅ reusable dataloaders
✅ augmentation pipeline
✅ baseline model
✅ experiment tracking
✅ metrics evaluation
✅ clean project structure
✅ ability to migrate to AWS later with minimal changes

---

# Important Mindset

Do NOT think:

> “How do I get best Kaggle score quickly?”

Think:

> “How do I design a production-grade ML experimentation framework?”

That mindset changes everything.

---

# Suggested Phase 1 Architecture

You should structure Phase 1 into:

```text id="aq6qmq"
1. Dataset Understanding
2. Data Pipeline
3. Augmentation Pipeline
4. Baseline Training
5. Experiment Tracking
6. Evaluation Framework
7. Configuration Management
8. Reproducibility
```

Let us go deep into each.

---

# 1. Dataset Understanding (VERY IMPORTANT)

Most beginners rush through this.

You should spend serious time here.

---

## Objectives

Understand:

* label distribution
* camera views
* image quality
* temporal relationships
* occlusions
* class imbalance
* synchronization between views

---

# Questions You Must Answer

## A. How many camera views exist?

Example:

* top view
* side view
* front view

Are all views always available?

This matters enormously.

---

## B. Are frames synchronized?

Example:

```text id="9zl77l"
frame_001_cam1
frame_001_cam2
frame_001_cam3
```

Are they captured at same timestamp?

If yes:

* multi-view fusion becomes meaningful

---

## C. Is this actually video data?

Even if stored as images.

You need to determine:

* whether neighboring frames are correlated
* whether temporal modeling is possible

---

## D. Class Distribution

Critical.

You should visualize:

* posture frequencies
* imbalance severity

Example:

* standing = 60%
* lying = 30%
* sitting = 10%

This affects:

* loss function
* augmentation
* sampling strategy

---

# Deliverables

You should create:

* EDA notebook
* visualizations
* sample grids
* class histograms

This becomes portfolio material later.

---

# 2. Data Pipeline (CORE ENGINEERING)

This is where good ML engineers differentiate themselves.

---

# Goal

Create a robust reusable PyTorch dataset system.

---

# What Your Dataset Class Should Handle

## A. Multi-view loading

Your dataset class should return:

```python
{
    "images": {
        "cam1": tensor,
        "cam2": tensor,
        "cam3": tensor
    },
    "label": label
}
```

NOT:

* separate loaders
* messy code
* hardcoded paths

---

# B. Missing View Handling

Real systems have:

* broken cameras
* corrupted images

You should prepare for this early.

Possible strategies:

* zero tensors
* learned embeddings
* view masking

This later becomes a strong interview discussion.

---

# C. Transform Pipeline

Use:

* Albumentations

Not torchvision transforms.

Albumentations is industry-standard for CV augmentation.

---

# D. Efficient Loading

Even with 5000 images:

* optimize early
* build habits

Use:

* pin_memory=True
* num_workers
* prefetching

---

# Recommended Dataset Design

```text id="0r5c9u"
src/
 ├── datasets/
 │    ├── pig_dataset.py
 │    ├── transforms.py
 │    └── samplers.py
```

---

# 3. Augmentation Pipeline

This is extremely important in CV.

Especially for small datasets.

---

# Objective

Improve:

* generalization
* robustness
* camera invariance

---

# Recommended Augmentations

## Safe Augmentations

These are likely appropriate:

* horizontal flip
* brightness/contrast
* gaussian noise
* blur
* slight rotation
* resize/crop

---

# Dangerous Augmentations

Avoid excessive:

* rotation
* perspective transforms

because posture semantics may break.

---

# Very Important: Multi-View Consistency

Suppose:

* cam1 flipped
* cam2 not flipped

Then views become inconsistent.

You need synchronized augmentations.

This is an advanced but important concept.

---

# Best Practice

Apply same geometric transform across all views.

But photometric transforms can vary.

Example:

```text id="y3tjsr"
Same:
- crop
- rotate
- flip

Different:
- brightness
- noise
```

This mimics real cameras.

---

# 4. Baseline Training Pipeline

DO NOT overcomplicate initially.

---

# Your First Goal

Train a single-view classifier.

NOT multi-view.

Why?

Because you first need:

* stable training
* debugging
* metrics baseline

---

# Recommended Baseline

Use:

* EfficientNet-B0
  OR
* ResNet18

Why not huge models?

* small dataset
* faster iteration
* easier debugging

---

# Training Stack

I strongly recommend:

| Component | Tool              |
| --------- | ----------------- |
| Framework | PyTorch           |
| Trainer   | PyTorch Lightning |
| Logging   | W&B               |
| Configs   | Hydra             |
| Metrics   | torchmetrics      |

This is close to real industry stacks.

---

# Why PyTorch Lightning?

Without Lightning:

* training loop clutter
* harder reproducibility
* messy scaling later

Lightning gives:

* cleaner code
* logging
* checkpointing
* GPU abstraction

---

# Baseline Objective

Establish:

* reproducible metric
* confusion matrix
* overfitting behavior

---

# 5. Experiment Tracking (VERY IMPORTANT)

Many candidates skip this.

Huge mistake.

---

# Use Weights & Biases

Use:

* hyperparameter logging
* metric tracking
* artifact storage

Track:

* learning rate
* augmentations
* architecture
* accuracy
* F1
* confusion matrix

---

# Why This Matters

Interviewers LOVE hearing:

> “I tracked experiments systematically and versioned datasets/models.”

This sounds production-oriented.

---

# 6. Evaluation Framework

This is often weak in portfolio projects.

---

# You Need More Than Accuracy

Because posture datasets are often imbalanced.

Use:

* macro F1
* per-class accuracy
* confusion matrix

---

# Important Questions

Which postures are confused?

Example:

* sitting vs lying
* standing vs transitioning

This is where actual ML insight appears.

---

# Strong Addition

Visualize:

* false positives
* false negatives

This is extremely valuable.

---

# 7. Configuration Management

Very important.

Do NOT hardcode:

* learning rate
* paths
* batch size

Use Hydra.

---

# Example

```yaml
model:
  name: efficientnet_b0

training:
  batch_size: 32
  lr: 1e-4

dataset:
  image_size: 224
```

This later makes SageMaker migration easy.

---

# 8. Reproducibility

Critical.

Set:

* random seeds
* deterministic behavior
* versioned dependencies

Create:

* requirements.txt
  OR
* poetry

---

# Recommended Initial Colab Workflow

## Stage 1 — Exploration

Notebook:

```text id="0g6zw9"
01_eda.ipynb
```

---

## Stage 2 — Dataset Validation

Notebook:

```text id="d2x06s"
02_dataset_debug.ipynb
```

Visualize:

* batches
* augmentations
* labels

---

## Stage 3 — Training

Notebook:

```text id="1t5txe"
03_baseline_training.ipynb
```

---

## Stage 4 — Refactor Into Scripts

VERY IMPORTANT.

Eventually move to:

```text id="w4q6xw"
train.py
evaluate.py
predict.py
```

Notebook-only projects look weak professionally.

---

# Recommended Initial Repository Structure

```text id="w9x0pp"
pig-posture-project/
│
├── notebooks/
│
├── configs/
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── utils/
│   └── visualization/
│
├── scripts/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

# Important Design Decision

## Should You Use Multi-View Immediately?

My recommendation:

NO.

---

# Correct Progression

## Stage 1

Single image classifier

## Stage 2

Single-view temporal model

## Stage 3

Multi-view fusion

Why?

Because debugging multi-view systems is much harder.

---

# Your Immediate Next Tasks

I would suggest your next concrete steps are:

---

## Task 1

Download dataset locally into Colab

---

## Task 2

Perform EDA:

* class distribution
* view distribution
* visualize samples

---

## Task 3

Design dataset abstraction

---

## Task 4

Implement baseline dataloader

---

## Task 5

Train single-view baseline

---

# Important Advice About Colab

Use:

* Google Drive mount
* persistent checkpoints
* periodic artifact saving

Colab sessions die frequently.

---

# One More Important Thing

Even in Phase 1:

* use Docker eventually
* keep paths configurable
* avoid notebook spaghetti code

Future-you will thank you when migrating to AWS.

---

# Suggested First Technical Stack

I recommend:

| Purpose       | Tool           |
| ------------- | -------------- |
| DL Framework  | PyTorch        |
| Trainer       | Lightning      |
| Tracking      | W&B            |
| Augmentation  | Albumentations |
| Configs       | Hydra          |
| Metrics       | torchmetrics   |
| Visualization | matplotlib     |
| Data handling | pandas         |

This is a modern, employable stack.

---

The next thing we can do is:

1. Design the exact repository architecture
2. Design the dataset abstraction for this competition
3. Discuss baseline model choices deeply
4. Discuss multi-view fusion architectures
5. Discuss Colab → AWS migration strategy
6. Design the experiment tracking setup
7. Create the exact implementation roadmap for Week 1

I would recommend starting with dataset abstraction next, because that determines everything downstream.
