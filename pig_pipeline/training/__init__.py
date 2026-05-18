"""Training and evaluation utilities."""

__all__ = [
	"F1ClassificationTrainer",
	"F1ClassificationValidator",
	"LitImageClassifier",
	"train_torchvision_classifier",
	"calibrate_probs",
	"compute_classification_metrics",
]


def __getattr__(name: str):
	if name in {"F1ClassificationTrainer", "F1ClassificationValidator"}:
		from .yolo import F1ClassificationTrainer, F1ClassificationValidator

		exports = {
			"F1ClassificationTrainer": F1ClassificationTrainer,
			"F1ClassificationValidator": F1ClassificationValidator,
		}
		return exports[name]
	if name in {"LitImageClassifier", "train_torchvision_classifier"}:
		from .torchvision import LitImageClassifier, train_torchvision_classifier

		exports = {
			"LitImageClassifier": LitImageClassifier,
			"train_torchvision_classifier": train_torchvision_classifier,
		}
		return exports[name]
	if name in {"calibrate_probs", "compute_classification_metrics"}:
		from .utills import calibrate_probs, compute_classification_metrics

		exports = {
			"calibrate_probs": calibrate_probs,
			"compute_classification_metrics": compute_classification_metrics,
		}
		return exports[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
