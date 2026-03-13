"""Out-of-Distribution Detection for Explainable AI.

This package provides state-of-the-art OOD detection methods for robust AI systems.
"""

__version__ = "0.1.0"
__author__ = "XAI Research Team"
__email__ = "research@example.com"

from .methods import MahalanobisOODDetector, EnergyOODDetector
from .eval import evaluate_ood_detection
from .data import load_iris_dataset, load_synthetic_dataset
from .viz import plot_ood_scores, plot_roc_curve, plot_method_comparison
from .utils import set_seed, get_device

__all__ = [
    "MahalanobisOODDetector",
    "EnergyOODDetector", 
    "evaluate_ood_detection",
    "load_iris_dataset",
    "load_synthetic_dataset",
    "plot_ood_scores",
    "plot_roc_curve", 
    "plot_method_comparison",
    "set_seed",
    "get_device",
]
