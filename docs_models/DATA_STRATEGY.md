This is a classic "Domain Shift" problem combined with severe class imbalance. If your test data is entirely "Angle B," but your training data is 95% "Angle A," a standard random split will fail. The model will optimize for Angle A, and your validation scores will be high, but the model will crash and burn on the actual test set.

To succeed, you need to treat your 60 "Angle B" images as your most valuable resource.

### 1. The "Golden Rule" Split: Angle-Based Validation

**Do not do a random split.** If you put some "Angle B" images in training and some in validation randomly, you won't know if the model actually learned to generalize or if it just memorized those specific 60 images.

* **The Strategy:** Use a **Stratified Group Split** based on View Angle.
* **Validation Set:** Put 20–30 of your "Angle B" images exclusively in your Validation set.
* **Training Set:** Use all "Angle A" images + the remaining 30–40 "Angle B" images.
* **Why?** This forces the validation score to reflect "Angle B" performance. If the model fails on Angle B during training, your validation mAP will stay low, alerting you that the model hasn't generalized yet.

---

### 2. Handling Class Imbalance (Oversampling)

With only 60 images of Angle B against thousands of Angle A, the model will treat Angle B as "noise."

* **The Fix:** **Oversample Angle B.** In your `train_fold.txt`, list the paths to those 60 images multiple times (e.g., repeat them 10x or 20x).
* **Goal:** You want the model to see an "Angle B" image at least as often as an "Angle A" image in every batch.

---

### 3. Domain Adaptation: The "Pseudo-Labeling" Loop

Since you have a test dataset (Angle B) but no labels, you can use an iterative process to "teach" the model the new domain:

1. **Step 1:** Train your 5-fold ensemble using the Angle-Based split mentioned above.
2. **Step 2 (Inference):** Run your ensemble on the **unlabeled test data**.
3. **Step 3 (Filtering):** Identify images where all 5 models agree with very high confidence (e.g., > 0.90).
4. **Step 4 (Pseudo-Labels):** Take those high-confidence predictions and treat them as "Ground Truth" labels.
5. **Step 5 (Retrain):** Add these new "pseudo-labeled" Angle B images into your training set and retrain the models.

---

### 4. Augmentation Strategy

Since the difference is a "View Angle," standard flips and crops aren't enough. You need to simulate the geometric distortion of Angle B.

* **Perspective Transforms:** Use heavy perspective/shear augmentations in your YOLO config to "warp" Angle A images so they mathematically look more like Angle B.
* **Mixup/Mosaic:** Ensure these are turned ON, as they help the model focus on object features rather than background context (which changes with view angles).

### Summary of your Data Plan

| Component | Action |
| --- | --- |
| **Train Set** | All Angle A + 40 Angle B (Repeated 15x in the .txt list). |
| **Val Set** | 20 Angle B images ONLY. |
| **Augmentation** | High `degrees`, `shear`, and `perspective` settings. |
| **Ensemble** | Use **WBF (Weighted Boxes Fusion)** to merge predictions on test data. |

### Suggested Python Tool: `imgaug` or `Albumentations`

While YOLO has built-in transforms, using **Albumentations** to specifically apply `Perspective` or `Affine` transforms to your "Angle A" images can help bridge the gap between views more effectively than the default YOLO pipeline.

**Which ensemble voting method are you planning to use—Weighted Boxes Fusion (WBF) or a simple NMS on the combined results?**