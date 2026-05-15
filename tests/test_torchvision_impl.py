"""Tests for the torchvision training pipeline.

Covers:
    1. Loss and metrics: cross-entropy, label-smoothing, val metric accumulation.
    2. Image transforms: eval transform output shape/dtype/normalization range.
    3. LR strategy param groups: correct group counts, LR values, and backbone freeze.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from pig_pipeline.training.torchvision import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LitImageClassifier,
    build_transforms,
)
from pig_pipeline.training.utills import compute_classification_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classifier(
    num_classes: int = 4,
    model_name: str = "efficientnet_v2_s",
    lr: float = 1e-3,
    lr_strategy: str = "single",
    lr_backbone_factor: float = 0.1,
    lr_layer_decay: float = 0.3,
    freeze_backbone_epochs: int = 0,
    label_smoothing: float = 0.0,
) -> LitImageClassifier:
    return LitImageClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=0.0,
        lr=lr,
        weight_decay=1e-4,
        label_smoothing=label_smoothing,
        max_epochs=5,
        lr_strategy=lr_strategy,
        lr_backbone_factor=lr_backbone_factor,
        lr_layer_decay=lr_layer_decay,
        freeze_backbone_epochs=freeze_backbone_epochs,
    )


# ---------------------------------------------------------------------------
# 1. Loss and metrics
# ---------------------------------------------------------------------------

class TestLossAndMetrics:
    """Verify that the cross-entropy loss and val-epoch metric helpers are correct."""

    def test_training_step_loss_is_finite(self):
        """Loss from training_step must be a finite scalar tensor."""
        model = _make_classifier(num_classes=4)
        model.eval()

        batch_size, num_classes = 8, 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, num_classes, (batch_size,))

        loss = model.training_step((x, y), batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0, "Loss must be a scalar"
        assert math.isfinite(loss.item()), "Loss must be finite"
        assert loss.item() > 0.0, "Cross-entropy loss must be positive"

    def test_label_smoothing_increases_loss(self):
        """Label smoothing must increase loss when the model predicts with high confidence.

        With perfect logits F.cross_entropy gives ~0 without smoothing, but smoothing
        pulls predictions toward uniform → loss = ε * log(K) / (K-1) > 0.
        """
        num_classes = 4
        eps = 0.2
        # Perfect logits: very high score for each sample's true class
        batch_size = 16
        y = torch.arange(batch_size) % num_classes
        logits = torch.full((batch_size, num_classes), -100.0)
        logits[torch.arange(batch_size), y] = 100.0  # confident correct prediction

        loss_plain = F.cross_entropy(logits, y, label_smoothing=0.0).item()
        loss_smooth = F.cross_entropy(logits, y, label_smoothing=eps).item()

        assert loss_smooth > loss_plain, (
            f"Smoothed loss ({loss_smooth:.4f}) should exceed plain loss ({loss_plain:.4f}) "
            "when the model predicts with perfect confidence"
        )

    def test_val_metrics_perfect_predictions(self):
        """When predictions == targets, top1=1.0 and macro_f1=1.0."""
        num_classes = 5
        n_samples = 50
        labels = list(range(num_classes)) * (n_samples // num_classes)

        metrics = compute_classification_metrics(labels, labels)
        assert metrics["top1"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)

    def test_val_metrics_worst_case(self):
        """When every prediction is wrong, top1=0.0."""
        y_true = [0, 1, 2, 3]
        y_pred = [1, 2, 3, 0]

        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["top1"] == pytest.approx(0.0)
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    def test_val_epoch_accumulation(self):
        """on_validation_epoch_end correctly computes metrics from accumulated batches."""
        num_classes = 3
        model = _make_classifier(num_classes=num_classes)

        # Simulate two validation batches with perfect predictions
        for _ in range(2):
            preds = torch.tensor([0, 1, 2])
            targets = torch.tensor([0, 1, 2])
            model._val_preds.append(preds)
            model._val_targets.append(targets)

        # Patch self.log to capture logged values
        logged: dict[str, float] = {}
        model.log = lambda key, val, **_: logged.update({key: val})

        model.on_validation_epoch_end()

        assert logged["val_top1"] == pytest.approx(1.0)
        assert logged["val_macro_f1"] == pytest.approx(1.0)
        # Buffers must be cleared after epoch end
        assert len(model._val_preds) == 0
        assert len(model._val_targets) == 0


# ---------------------------------------------------------------------------
# 2. Image transforms
# ---------------------------------------------------------------------------

class TestImageTransforms:
    """Verify that build_transforms returns correct pipelines."""

    @pytest.fixture(scope="class")
    def transforms_224(self):
        return build_transforms(img_size=224)

    @pytest.fixture(scope="class")
    def sample_image(self):
        """A deterministic 300×300 RGB PIL image."""
        arr = np.random.default_rng(0).integers(0, 256, (300, 300, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_eval_output_shape(self, transforms_224, sample_image):
        """Eval transform must produce a (3, 224, 224) tensor."""
        _, eval_tf = transforms_224
        tensor = eval_tf(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_eval_output_dtype(self, transforms_224, sample_image):
        """Eval transform must produce a float32 tensor."""
        _, eval_tf = transforms_224
        tensor = eval_tf(sample_image)
        assert tensor.dtype == torch.float32

    def test_eval_normalization_range(self, transforms_224, sample_image):
        """After ImageNet normalization the per-channel range must be plausible (not [0,1])."""
        _, eval_tf = transforms_224
        tensor = eval_tf(sample_image)  # shape (3, H, W)

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        # Invert normalization and check pixel range
        unnorm = tensor * std + mean
        assert float(unnorm.min()) >= -0.05, "Unnormalised pixels should be ≥ 0"
        assert float(unnorm.max()) <= 1.05, "Unnormalised pixels should be ≤ 1"

        # The tensor itself must NOT be in [0, 1] due to normalization
        assert float(tensor.min()) < 0.0, "Normalised tensor should have negative values"

    def test_train_transform_output_shape(self, transforms_224, sample_image):
        """Train transform must produce a (3, 224, 224) tensor."""
        train_tf, _ = transforms_224
        tensor = train_tf(sample_image)
        assert tensor.shape == (3, 224, 224)

    def test_eval_transform_is_deterministic(self, transforms_224, sample_image):
        """Eval transform must return identical tensors on repeated calls (no random ops)."""
        _, eval_tf = transforms_224
        t1 = eval_tf(sample_image)
        t2 = eval_tf(sample_image)
        assert torch.allclose(t1, t2), "Eval transform must be deterministic"

    def test_eval_resize_ratio_respected(self):
        """Custom eval_resize_ratio must change the intermediate resize size."""
        img_size = 224
        ratio = 1.5
        _, eval_tf = build_transforms(img_size=img_size, aug_cfg={"eval_resize_ratio": ratio})

        # The first step is a Resize; its size should be round(224 * 1.5) = 336
        resize_op = eval_tf.transforms[0]
        expected = int(round(img_size * ratio))
        assert resize_op.size in ([expected, expected], (expected, expected)), (
            f"Expected resize to {expected}×{expected}, got {resize_op.size}"
        )


# ---------------------------------------------------------------------------
# 3. LR strategy param groups
# ---------------------------------------------------------------------------

class TestLRStrategyParamGroups:
    """Verify that _build_param_groups constructs the correct groups for each strategy."""

    @pytest.mark.parametrize("model_name", ["efficientnet_v2_s", "convnext_tiny", "resnet50"])
    def test_single_strategy_one_group(self, model_name):
        model = _make_classifier(model_name=model_name, lr_strategy="single")
        groups = model._build_param_groups()
        assert len(groups) == 1
        assert groups[0]["lr"] == pytest.approx(model.lr)

    @pytest.mark.parametrize("model_name", ["efficientnet_v2_s", "convnext_tiny", "resnet50"])
    def test_single_strategy_all_params_covered(self, model_name):
        model = _make_classifier(model_name=model_name, lr_strategy="single")
        groups = model._build_param_groups()
        group_param_ids = {id(p) for g in groups for p in g["params"]}
        all_param_ids = {id(p) for p in model.parameters()}
        assert group_param_ids == all_param_ids, "single strategy must cover all params"

    @pytest.mark.parametrize("model_name", ["efficientnet_v2_s", "convnext_tiny", "resnet50"])
    def test_backbone_head_strategy_lr_values(self, model_name):
        lr = 1e-3
        factor = 0.1
        model = _make_classifier(
            model_name=model_name,
            lr_strategy="backbone_head",
            lr=lr,
            lr_backbone_factor=factor,
        )
        groups = model._build_param_groups()
        group_by_name = {g["name"]: g for g in groups}

        assert "head" in group_by_name, "backbone_head must have a 'head' group"
        assert "backbone" in group_by_name, "backbone_head must have a 'backbone' group"
        assert group_by_name["head"]["lr"] == pytest.approx(lr)
        assert group_by_name["backbone"]["lr"] == pytest.approx(lr * factor)

    @pytest.mark.parametrize("model_name", ["efficientnet_v2_s", "convnext_tiny", "resnet50"])
    def test_backbone_head_strategy_no_param_overlap(self, model_name):
        model = _make_classifier(model_name=model_name, lr_strategy="backbone_head")
        groups = model._build_param_groups()
        seen_ids: set[int] = set()
        for g in groups:
            for p in g["params"]:
                assert id(p) not in seen_ids, f"Param {id(p)} appears in multiple groups"
                seen_ids.add(id(p))

    @pytest.mark.parametrize("model_name", ["efficientnet_v2_s", "convnext_tiny"])
    def test_layerwise_strategy_lr_ordering(self, model_name):
        """head LR > late > mid > early (for decay < 1)."""
        lr, d = 1e-3, 0.3
        model = _make_classifier(
            model_name=model_name,
            lr_strategy="layerwise",
            lr=lr,
            lr_layer_decay=d,
        )
        groups = model._build_param_groups()
        by_name = {g["name"]: g["lr"] for g in groups}

        assert by_name.get("head", 0.0) == pytest.approx(lr)
        assert by_name.get("backbone_late", 0.0) == pytest.approx(lr * d)
        assert by_name.get("backbone_mid", 0.0) == pytest.approx(lr * d**2)
        assert by_name.get("backbone_early", 0.0) == pytest.approx(lr * d**3)

        assert by_name["head"] > by_name["backbone_late"]
        assert by_name["backbone_late"] > by_name["backbone_mid"]
        assert by_name["backbone_mid"] > by_name["backbone_early"]

    def test_freeze_backbone_epochs_disables_backbone_grad(self):
        """With freeze_backbone_epochs > 0, backbone params must have requires_grad=False."""
        model = _make_classifier(
            model_name="efficientnet_v2_s",
            freeze_backbone_epochs=3,
        )
        for name, param in model.named_parameters():
            if model._is_head_param_name(name):
                assert param.requires_grad, f"Head param {name} should be trainable"
            elif name.startswith("model."):
                assert not param.requires_grad, f"Backbone param {name} should be frozen"

    def test_freeze_unfreeze_via_epoch_hook(self):
        """on_train_epoch_start must unfreeze backbone once current_epoch >= freeze_backbone_epochs.

        We mock the `current_epoch` property because Lightning reads it from the trainer.
        """
        from unittest.mock import PropertyMock, patch

        model = _make_classifier(
            model_name="efficientnet_v2_s",
            freeze_backbone_epochs=2,
        )

        def backbone_frozen(m: LitImageClassifier) -> bool:
            return all(
                not p.requires_grad
                for n, p in m.named_parameters()
                if n.startswith("model.") and not m._is_head_param_name(n)
            )

        # Before unfreeze threshold: epoch 0
        with patch.object(
            type(model), "current_epoch", new_callable=PropertyMock, return_value=0
        ):
            model.on_train_epoch_start()
            assert backbone_frozen(model), "Backbone should remain frozen before freeze_backbone_epochs"

        # At unfreeze threshold: epoch 2
        with patch.object(
            type(model), "current_epoch", new_callable=PropertyMock, return_value=2
        ):
            model.on_train_epoch_start()
            assert not backbone_frozen(model), "Backbone should be unfrozen at freeze_backbone_epochs"

    def test_invalid_lr_strategy_raises(self):
        with pytest.raises(ValueError, match="lr_strategy"):
            _make_classifier(lr_strategy="cosine_warmup_unknown")
