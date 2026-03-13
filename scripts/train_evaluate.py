"""Main training and evaluation script for OOD detection methods."""

import argparse
import os
from typing import Dict, List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.ood_detection.data import load_iris_dataset, load_synthetic_dataset, preprocess_data
from src.ood_detection.methods import MahalanobisOODDetector, EnergyOODDetector
from src.ood_detection.eval import evaluate_ood_detection, create_evaluation_report, compare_methods
from src.ood_detection.viz import (
    plot_ood_scores,
    plot_roc_curve,
    plot_method_comparison,
    create_summary_plot,
)
from src.ood_detection.utils import set_seed, get_device, create_experiment_dir, log_metrics


def load_dataset(dataset_name: str, **kwargs) -> tuple:
    """Load dataset based on name.
    
    Args:
        dataset_name: Name of the dataset to load.
        **kwargs: Additional arguments for dataset loading.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_ood).
    """
    if dataset_name == "iris":
        return load_iris_dataset(**kwargs)
    elif dataset_name == "synthetic":
        return load_synthetic_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_ood_detection_experiment(
    config: DictConfig,
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """Run OOD detection experiment with multiple methods.
    
    Args:
        config: Configuration object.
        output_dir: Output directory for results.
        
    Returns:
        Dictionary of method results.
    """
    # Set random seed
    set_seed(config.seed)
    
    # Load dataset
    print(f"Loading {config.dataset.name} dataset...")
    X_train, X_test, y_train, y_test, X_ood = load_dataset(
        config.dataset.name,
        **config.dataset.params
    )
    
    # Preprocess data
    X_train_scaled, X_test_scaled, X_ood_scaled, scaler = preprocess_data(
        X_train, X_test, X_ood
    )
    
    # Initialize methods
    methods = {}
    if config.methods.mahalanobis.enabled:
        methods["Mahalanobis"] = MahalanobisOODDetector(
            **config.methods.mahalanobis.params
        )
    
    if config.methods.energy.enabled:
        methods["Energy"] = EnergyOODDetector(
            **config.methods.energy.params
        )
    
    # Train and evaluate methods
    method_results = {}
    
    for method_name, detector in methods.items():
        print(f"\nTraining {method_name} detector...")
        
        # Train detector
        detector.fit(X_train_scaled, y_train)
        
        # Compute scores
        in_dist_scores = detector.score_samples(X_test_scaled)
        ood_scores = detector.score_samples(X_ood_scaled)
        
        # Evaluate performance
        metrics = evaluate_ood_detection(in_dist_scores, ood_scores)
        method_results[method_name] = metrics
        
        # Create visualizations
        print(f"Creating visualizations for {method_name}...")
        
        # Score distribution plot
        fig_scores = plot_ood_scores(in_dist_scores, ood_scores, method_name)
        fig_scores.savefig(
            os.path.join(output_dir, f"{method_name.lower()}_scores.png"),
            dpi=300,
            bbox_inches="tight"
        )
        fig_scores.close()
        
        # ROC curve
        fig_roc = plot_roc_curve(in_dist_scores, ood_scores, method_name)
        fig_roc.savefig(
            os.path.join(output_dir, f"{method_name.lower()}_roc.png"),
            dpi=300,
            bbox_inches="tight"
        )
        fig_roc.close()
        
        # Print results
        report = create_evaluation_report(method_name, metrics)
        print(report)
        
        # Save individual results
        log_metrics(metrics, os.path.join(output_dir, f"{method_name.lower()}_metrics.json"))
    
    # Create comparison plots
    if len(method_results) > 1:
        print("\nCreating comparison plots...")
        
        # Method comparison plot
        fig_comparison = plot_method_comparison(method_results)
        fig_comparison.savefig(
            os.path.join(output_dir, "method_comparison.png"),
            dpi=300,
            bbox_inches="tight"
        )
        fig_comparison.close()
        
        # Summary plot
        fig_summary = create_summary_plot(method_results)
        fig_summary.savefig(
            os.path.join(output_dir, "summary_comparison.png"),
            dpi=300,
            bbox_inches="tight"
        )
        fig_summary.close()
        
        # Print comparison
        comparison = compare_methods(method_results)
        print("\n=== Method Comparison ===")
        print(f"Primary metric: {comparison['primary_metric']}")
        print("\nRankings:")
        for i, (method, score) in enumerate(comparison['rankings'], 1):
            print(f"  {i}. {method}: {score:.4f}")
        
        print(f"\nStatistics:")
        for stat, value in comparison['statistics'].items():
            print(f"  {stat}: {value:.4f}")
    
    # Save overall results
    log_metrics(method_results, os.path.join(output_dir, "all_results.json"))
    
    return method_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="OOD Detection Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mahalanobis", "energy", "all"],
        default="all",
        help="OOD detection method to run"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["iris", "synthetic"],
        default="iris",
        help="Dataset to use"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
    else:
        # Create default configuration
        config = OmegaConf.create({
            "seed": 42,
            "dataset": {
                "name": args.dataset,
                "params": {
                    "test_size": 0.3,
                    "ood_size": 100,
                    "random_state": 42,
                }
            },
            "methods": {
                "mahalanobis": {
                    "enabled": args.method in ["mahalanobis", "all"],
                    "params": {"regularization": 1e-6}
                },
                "energy": {
                    "enabled": args.method in ["energy", "all"],
                    "params": {
                        "hidden_layer_sizes": (100, 50),
                        "learning_rate": 0.001,
                        "max_iter": 1000,
                        "temperature": 1.0,
                    }
                }
            }
        })
    
    # Create output directory
    exp_dir = create_experiment_dir(args.output_dir, f"ood_detection_{args.dataset}")
    
    # Save configuration
    OmegaConf.save(config, os.path.join(exp_dir, "config.yaml"))
    
    print(f"Starting OOD detection experiment...")
    print(f"Output directory: {exp_dir}")
    print(f"Configuration: {config}")
    
    # Run experiment
    try:
        results = run_ood_detection_experiment(config, exp_dir)
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {exp_dir}")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
