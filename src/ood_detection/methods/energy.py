"""Energy-based OOD detection using neural networks."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseOODDetector


class EnergyOODDetector(BaseOODDetector):
    """OOD detection using energy-based scoring.
    
    This method trains a neural network and uses the energy (negative log-sum-exp)
    of the logits as OOD scores. Higher energy indicates more likely OOD samples.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100, 50),
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        temperature: float = 1.0,
        **kwargs
    ) -> None:
        """Initialize Energy OOD detector.
        
        Args:
            hidden_layer_sizes: Hidden layer sizes for MLP.
            learning_rate: Learning rate for training.
            max_iter: Maximum iterations for training.
            temperature: Temperature scaling for energy computation.
            **kwargs: Additional parameters.
        """
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.temperature = temperature
        
        self.model_ = None
        self.scaler_ = None
        self.training_scores_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "EnergyOODDetector":
        """Fit the Energy detector.
        
        Args:
            X: Training features.
            y: Training labels (required for this method).
            
        Returns:
            Self for method chaining.
        """
        if y is None:
            raise ValueError("Labels are required for Energy OOD detection.")
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Train MLP classifier
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42,
        )
        self.model_.fit(X_scaled, y)
        
        # Set fitted flag before computing training scores
        self.is_fitted = True
        
        # Compute training scores
        self.training_scores_ = self.score_samples(X)
        
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute energy-based OOD scores.
        
        Args:
            X: Features to score.
            
        Returns:
            Energy scores (higher = more likely OOD).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring.")
        
        # Scale features
        X_scaled = self.scaler_.transform(X)
        
        # Get logits from the model
        logits = self.model_.predict_proba(X_scaled)
        
        # Compute energy scores
        # Energy = -log(sum(exp(logits / temperature)))
        energy_scores = -np.log(np.sum(np.exp(logits / self.temperature), axis=1))
        
        return energy_scores
    
    def get_params(self) -> dict:
        """Get detector parameters.
        
        Returns:
            Dictionary of parameters.
        """
        params = super().get_params()
        params.update({
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "temperature": self.temperature,
        })
        return params


class PyTorchEnergyOODDetector(BaseOODDetector):
    """PyTorch-based Energy OOD detector for more control."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [100, 50],
        learning_rate: float = 0.001,
        epochs: int = 100,
        temperature: float = 1.0,
        device: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize PyTorch Energy detector.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dims: Hidden layer dimensions.
            learning_rate: Learning rate for training.
            epochs: Number of training epochs.
            temperature: Temperature scaling for energy computation.
            device: Device to use ('cuda', 'mps', 'cpu').
            **kwargs: Additional parameters.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.temperature = temperature
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model_ = None
        self.scaler_ = None
        self.training_scores_ = None
    
    def _build_model(self, n_classes: int) -> nn.Module:
        """Build the neural network model.
        
        Args:
            n_classes: Number of output classes.
            
        Returns:
            PyTorch model.
        """
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        
        return nn.Sequential(*layers).to(self.device)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PyTorchEnergyOODDetector":
        """Fit the PyTorch Energy detector.
        
        Args:
            X: Training features.
            y: Training labels (required).
            
        Returns:
            Self for method chaining.
        """
        if y is None:
            raise ValueError("Labels are required for Energy OOD detection.")
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Build model
        n_classes = len(np.unique(y))
        self.model_ = self._build_model(n_classes)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model_(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Compute training scores
        self.training_scores_ = self.score_samples(X)
        
        self.is_fitted = True
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute energy-based OOD scores.
        
        Args:
            X: Features to score.
            
        Returns:
            Energy scores (higher = more likely OOD).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring.")
        
        # Scale features
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get logits
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
        
        # Compute energy scores
        energy_scores = -torch.logsumexp(logits / self.temperature, dim=1)
        
        return energy_scores.cpu().numpy()
    
    def get_params(self) -> dict:
        """Get detector parameters.
        
        Returns:
            Dictionary of parameters.
        """
        params = super().get_params()
        params.update({
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "temperature": self.temperature,
            "device": str(self.device),
        })
        return params
