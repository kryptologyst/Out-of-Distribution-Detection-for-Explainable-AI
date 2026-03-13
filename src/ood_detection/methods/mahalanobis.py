"""Mahalanobis distance-based OOD detection."""

from typing import Optional

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv

from .base import BaseOODDetector


class MahalanobisOODDetector(BaseOODDetector):
    """OOD detection using Mahalanobis distance.
    
    This method computes the Mahalanobis distance between test samples and the
    training data distribution. Samples with high Mahalanobis distances are
    considered out-of-distribution.
    """
    
    def __init__(self, regularization: float = 1e-6, **kwargs) -> None:
        """Initialize Mahalanobis OOD detector.
        
        Args:
            regularization: Regularization term for covariance matrix inversion.
            **kwargs: Additional parameters.
        """
        super().__init__(**kwargs)
        self.regularization = regularization
        self.mean_ = None
        self.cov_ = None
        self.inv_cov_ = None
        self.training_scores_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MahalanobisOODDetector":
        """Fit the Mahalanobis detector.
        
        Args:
            X: Training features.
            y: Training labels (ignored).
            
        Returns:
            Self for method chaining.
        """
        # Compute mean and covariance
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X.T)
        
        # Add regularization to avoid singular matrix
        self.cov_ += np.eye(self.cov_.shape[0]) * self.regularization
        
        # Compute inverse covariance matrix
        self.inv_cov_ = pinv(self.cov_)
        
        # Set fitted flag before computing training scores
        self.is_fitted = True
        
        # Compute training scores for threshold estimation
        self.training_scores_ = self.score_samples(X)
        
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance scores.
        
        Args:
            X: Features to score.
            
        Returns:
            Mahalanobis distance scores.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring.")
        
        scores = []
        for x in X:
            try:
                dist = mahalanobis(x, self.mean_, self.inv_cov_)
                scores.append(dist)
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to Euclidean distance if Mahalanobis fails
                dist = np.linalg.norm(x - self.mean_)
                scores.append(dist)
        
        return np.array(scores)
    
    def get_params(self) -> dict:
        """Get detector parameters.
        
        Returns:
            Dictionary of parameters including regularization.
        """
        params = super().get_params()
        params["regularization"] = self.regularization
        return params
