"""Data loading and preprocessing utilities for OOD detection."""

import os
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_iris_dataset(
    test_size: float = 0.3,
    ood_size: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare Iris dataset for OOD detection.
    
    Args:
        test_size: Fraction of data to use for testing.
        ood_size: Number of OOD samples to generate.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_ood).
    """
    # Load Iris dataset
    data = load_iris()
    X, y = data.data, data.target
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Generate OOD data by sampling from different distribution
    np.random.seed(random_state)
    X_ood = np.random.uniform(
        low=np.min(X_train) - 1.0,
        high=np.max(X_train) + 1.0,
        size=(ood_size, X_train.shape[1])
    )
    
    return X_train, X_test, y_train, y_test, X_ood


def load_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 4,
    n_classes: int = 3,
    test_size: float = 0.3,
    ood_size: int = 200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic dataset for OOD detection.
    
    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        n_classes: Number of classes.
        test_size: Fraction of data to use for testing.
        ood_size: Number of OOD samples to generate.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_ood).
    """
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state,
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Generate OOD data with different distribution
    np.random.seed(random_state + 1)
    X_ood = np.random.normal(
        loc=np.mean(X_train) + 2.0,
        scale=np.std(X_train) * 1.5,
        size=(ood_size, X_train.shape[1])
    )
    
    return X_train, X_test, y_train, y_test, X_ood


def preprocess_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_ood: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Preprocess data with standardization.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        X_ood: OOD features.
        scaler: Optional pre-fitted scaler.
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, X_ood_scaled, scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    X_ood_scaled = scaler.transform(X_ood)
    
    return X_train_scaled, X_test_scaled, X_ood_scaled, scaler


def get_dataset_metadata(dataset_name: str) -> Dict[str, Union[str, List[str]]]:
    """Get metadata for a dataset.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Dictionary containing dataset metadata.
    """
    metadata = {
        "iris": {
            "description": "Iris flower classification dataset",
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "target": "species",
            "classes": ["setosa", "versicolor", "virginica"],
            "feature_types": ["continuous"] * 4,
            "sensitive_attributes": [],
            "monotonicity_constraints": {},
        },
        "synthetic": {
            "description": "Synthetic classification dataset",
            "features": [f"feature_{i}" for i in range(4)],
            "target": "class",
            "classes": ["class_0", "class_1", "class_2"],
            "feature_types": ["continuous"] * 4,
            "sensitive_attributes": [],
            "monotonicity_constraints": {},
        },
    }
    
    return metadata.get(dataset_name, {})


def save_dataset_info(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_ood: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_dir: str,
) -> None:
    """Save dataset information and samples.
    
    Args:
        dataset_name: Name of the dataset.
        X_train: Training features.
        X_test: Test features.
        X_ood: OOD features.
        y_train: Training labels.
        y_test: Test labels.
        save_dir: Directory to save information.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metadata
    metadata = get_dataset_metadata(dataset_name)
    metadata.update({
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_ood_samples": len(X_ood),
        "n_features": X_train.shape[1],
        "n_classes": len(np.unique(y_train)),
    })
    
    import json
    
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save data samples
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "X_ood.npy"), X_ood)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
