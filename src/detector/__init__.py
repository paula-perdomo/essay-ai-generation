"""Detector (Discriminator) model and evaluation routines."""

from .model import EssayClassifier
from .trainer import DetectorTrainer
from .evaluate import DetectorEvaluator

__all__ = ["EssayClassifier", "DetectorTrainer", "DetectorEvaluator"]
