"""Utility functions for reproducible experiments and device management."""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration object.
    """
    from omegaconf import OmegaConf
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object.
        save_path: Path to save configuration.
    """
    from omegaconf import OmegaConf
    
    OmegaConf.save(config, save_path)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments.
        experiment_name: Name of the experiment.
        
    Returns:
        Path to created experiment directory.
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir


def log_metrics(metrics: Dict[str, Any], log_file: str) -> None:
    """Log metrics to file.
    
    Args:
        metrics: Dictionary of metrics to log.
        log_file: Path to log file.
    """
    import json
    
    with open(log_file, "w") as f:
        json.dump(metrics, f, indent=2)


def ensure_dir(path: str) -> None:
    """Ensure directory exists.
    
    Args:
        path: Directory path.
    """
    os.makedirs(path, exist_ok=True)
