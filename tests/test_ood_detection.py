"""Tests for OOD detection methods."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.ood_detection.data import load_iris_dataset, load_synthetic_dataset, preprocess_data
from src.ood_detection.methods import MahalanobisOODDetector, EnergyOODDetector
from src.ood_detection.eval import evaluate_ood_detection
from src.ood_detection.utils import set_seed


class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_iris_dataset(self):
        """Test Iris dataset loading."""
        X_train, X_test, y_train, y_test, X_ood = load_iris_dataset(
            test_size=0.3, ood_size=50, random_state=42
        )
        
        assert X_train.shape[1] == 4  # 4 features
        assert len(np.unique(y_train)) == 3  # 3 classes
        assert X_ood.shape[1] == 4
        assert len(X_ood) == 50
    
    def test_load_synthetic_dataset(self):
        """Test synthetic dataset loading."""
        X_train, X_test, y_train, y_test, X_ood = load_synthetic_dataset(
            n_samples=200, n_features=4, n_classes=3, 
            test_size=0.3, ood_size=50, random_state=42
        )
        
        assert X_train.shape[1] == 4
        assert len(np.unique(y_train)) == 3
        assert X_ood.shape[1] == 4
        assert len(X_ood) == 50
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        X_train = np.random.randn(100, 4)
        X_test = np.random.randn(30, 4)
        X_ood = np.random.randn(50, 4)
        
        X_train_scaled, X_test_scaled, X_ood_scaled, scaler = preprocess_data(
            X_train, X_test, X_ood
        )
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert X_ood_scaled.shape == X_ood.shape
        
        # Check that training data is standardized
        assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-10)


class TestMahalanobisDetector:
    """Test Mahalanobis OOD detector."""
    
    def test_fit_and_score(self):
        """Test fitting and scoring."""
        # Generate test data
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(30, 4)
        
        detector = MahalanobisOODDetector(regularization=1e-6)
        detector.fit(X_train, y_train)
        
        scores = detector.score_samples(X_test)
        
        assert len(scores) == len(X_test)
        assert np.all(scores >= 0)  # Mahalanobis distance should be non-negative
    
    def test_predict_ood(self):
        """Test OOD prediction."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(30, 4)
        
        detector = MahalanobisOODDetector()
        detector.fit(X_train, y_train)
        
        predictions = detector.predict_ood(X_test)
        
        assert len(predictions) == len(X_test)
        assert np.all(np.isin(predictions, [True, False]))
    
    def test_parameters(self):
        """Test parameter handling."""
        detector = MahalanobisOODDetector(regularization=1e-4)
        
        params = detector.get_params()
        assert params["regularization"] == 1e-4
        
        detector.set_params(regularization=1e-3)
        assert detector.regularization == 1e-3


class TestEnergyDetector:
    """Test Energy OOD detector."""
    
    def test_fit_and_score(self):
        """Test fitting and scoring."""
        # Generate test data
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(30, 4)
        
        detector = EnergyOODDetector(
            hidden_layer_sizes=(50, 25),
            max_iter=100
        )
        detector.fit(X_train, y_train)
        
        scores = detector.score_samples(X_test)
        
        assert len(scores) == len(X_test)
        assert np.all(np.isfinite(scores))  # Scores should be finite
    
    def test_predict_ood(self):
        """Test OOD prediction."""
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(30, 4)
        
        detector = EnergyOODDetector(max_iter=100)
        detector.fit(X_train, y_train)
        
        predictions = detector.predict_ood(X_test)
        
        assert len(predictions) == len(X_test)
        assert np.all(np.isin(predictions, [True, False]))
    
    def test_parameters(self):
        """Test parameter handling."""
        detector = EnergyOODDetector(
            hidden_layer_sizes=(100, 50),
            learning_rate=0.01,
            temperature=2.0
        )
        
        params = detector.get_params()
        assert params["hidden_layer_sizes"] == (100, 50)
        assert params["learning_rate"] == 0.01
        assert params["temperature"] == 2.0


class TestEvaluation:
    """Test evaluation functions."""
    
    def test_evaluate_ood_detection(self):
        """Test OOD detection evaluation."""
        # Create mock scores
        in_dist_scores = np.random.randn(100)  # Lower scores for in-distribution
        ood_scores = np.random.randn(50) + 2  # Higher scores for OOD
        
        metrics = evaluate_ood_detection(in_dist_scores, ood_scores)
        
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "fpr_at_95_tpr" in metrics
        assert "detection_error" in metrics
        
        # AUROC should be reasonable (not perfect due to randomness)
        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["auprc"] <= 1
        assert 0 <= metrics["fpr_at_95_tpr"] <= 1
        assert 0 <= metrics["detection_error"] <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_iris(self):
        """Test end-to-end pipeline with Iris dataset."""
        set_seed(42)
        
        # Load data
        X_train, X_test, y_train, y_test, X_ood = load_iris_dataset(
            test_size=0.3, ood_size=50, random_state=42
        )
        
        # Preprocess
        X_train_scaled, X_test_scaled, X_ood_scaled, _ = preprocess_data(
            X_train, X_test, X_ood
        )
        
        # Test Mahalanobis detector
        detector = MahalanobisOODDetector()
        detector.fit(X_train_scaled, y_train)
        
        in_dist_scores = detector.score_samples(X_test_scaled)
        ood_scores = detector.score_samples(X_ood_scaled)
        
        # Evaluate
        metrics = evaluate_ood_detection(in_dist_scores, ood_scores)
        
        # Should achieve reasonable performance
        assert metrics["auroc"] > 0.5  # Better than random
    
    def test_end_to_end_synthetic(self):
        """Test end-to-end pipeline with synthetic dataset."""
        set_seed(42)
        
        # Load data
        X_train, X_test, y_train, y_test, X_ood = load_synthetic_dataset(
            n_samples=200, test_size=0.3, ood_size=50, random_state=42
        )
        
        # Preprocess
        X_train_scaled, X_test_scaled, X_ood_scaled, _ = preprocess_data(
            X_train, X_test, X_ood
        )
        
        # Test Energy detector
        detector = EnergyOODDetector(max_iter=100)
        detector.fit(X_train_scaled, y_train)
        
        in_dist_scores = detector.score_samples(X_test_scaled)
        ood_scores = detector.score_samples(X_ood_scaled)
        
        # Evaluate
        metrics = evaluate_ood_detection(in_dist_scores, ood_scores)
        
        # Should achieve reasonable performance
        assert metrics["auroc"] > 0.5  # Better than random


if __name__ == "__main__":
    pytest.main([__file__])
