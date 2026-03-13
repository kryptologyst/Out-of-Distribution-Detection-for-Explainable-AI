"""Base class and interface for OOD detection methods."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseOODDetector(ABC):
    """Abstract base class for OOD detection methods."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the detector.
        
        Args:
            **kwargs: Additional parameters for the detector.
        """
        self.is_fitted = False
        self.params = kwargs
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseOODDetector":
        """Fit the OOD detector to training data.
        
        Args:
            X: Training features.
            y: Training labels (optional).
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute OOD scores for samples.
        
        Args:
            X: Features to score.
            
        Returns:
            OOD scores (higher = more likely OOD).
        """
        pass
    
    def predict_ood(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict whether samples are OOD.
        
        Args:
            X: Features to predict.
            threshold: OOD threshold. If None, uses median of training scores.
            
        Returns:
            Boolean array indicating OOD samples.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction.")
        
        scores = self.score_samples(X)
        
        if threshold is None:
            # Use median of training scores as threshold
            threshold = np.median(self.training_scores_)
        
        return scores > threshold
    
    def get_params(self) -> Dict[str, Any]:
        """Get detector parameters.
        
        Returns:
            Dictionary of parameters.
        """
        return self.params.copy()
    
    def set_params(self, **params: Any) -> "BaseOODDetector":
        """Set detector parameters.
        
        Args:
            **params: Parameters to set.
            
        Returns:
            Self for method chaining.
        """
        self.params.update(params)
        return self
