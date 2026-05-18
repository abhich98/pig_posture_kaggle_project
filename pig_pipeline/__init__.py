"""Reusable pipeline utilities for pig posture experiments."""

__version__ = "0.1.0"

__all__ = ["data", "training"]


def __getattr__(name: str):
	if name == "data":
		from . import data as data_module

		return data_module
	if name == "training":
		from . import training as training_module

		return training_module
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")