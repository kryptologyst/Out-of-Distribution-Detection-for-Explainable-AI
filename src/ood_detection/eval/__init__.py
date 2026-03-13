"""Evaluation metrics and utilities for OOD detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def evaluate_ood_detection(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate OOD detection performance.
    
    Args:
        in_dist_scores: OOD scores for in-distribution samples.
        ood_scores: OOD scores for out-of-distribution samples.
        metrics: List of metrics to compute. If None, computes all available.
        
    Returns:
        Dictionary of metric names and values.
    """
    if metrics is None:
        metrics = ["auroc", "auprc", "fpr_at_95_tpr", "detection_error"]
    
    # Create binary labels: 0 for in-distribution, 1 for OOD
    y_true = np.concatenate([
        np.zeros(len(in_dist_scores)),  # In-distribution samples
        np.ones(len(ood_scores))        # OOD samples
    ])
    
    # Combine scores
    y_scores = np.concatenate([in_dist_scores, ood_scores])
    
    results = {}
    
    if "auroc" in metrics:
        results["auroc"] = roc_auc_score(y_true, y_scores)
    
    if "auprc" in metrics:
        results["auprc"] = average_precision_score(y_true, y_scores)
    
    if "fpr_at_95_tpr" in metrics:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        # Find threshold where TPR = 0.95
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            results["fpr_at_95_tpr"] = fpr[idx[0]]
        else:
            results["fpr_at_95_tpr"] = 1.0
    
    if "detection_error" in metrics:
        # Detection error = 0.5 * (FPR + FNR) at optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        detection_errors = 0.5 * (fpr + fnr)
        results["detection_error"] = np.min(detection_errors)
    
    return results


def compute_calibration_metrics(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, float]:
    """Compute calibration metrics.
    
    Args:
        predictions: Predicted probabilities.
        true_labels: True binary labels.
        num_bins: Number of bins for calibration.
        
    Returns:
        Dictionary of calibration metrics.
    """
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Maximum Calibration Error (MCE)
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return {
        "ece": ece,
        "mce": mce,
    }


def compute_robustness_metrics(
    clean_scores: np.ndarray,
    adversarial_scores: np.ndarray,
    clean_labels: np.ndarray,
    adversarial_labels: np.ndarray,
) -> Dict[str, float]:
    """Compute robustness metrics.
    
    Args:
        clean_scores: OOD scores on clean data.
        adversarial_scores: OOD scores on adversarial data.
        clean_labels: True labels for clean data.
        adversarial_labels: True labels for adversarial data.
        
    Returns:
        Dictionary of robustness metrics.
    """
    # Score stability
    score_correlation = np.corrcoef(clean_scores, adversarial_scores)[0, 1]
    
    # Detection consistency
    clean_ood_pred = clean_scores > np.median(clean_scores)
    adv_ood_pred = adversarial_scores > np.median(adversarial_scores)
    detection_consistency = (clean_ood_pred == adv_ood_pred).mean()
    
    # Performance degradation
    clean_performance = evaluate_ood_detection(
        clean_scores[clean_labels == 0],
        clean_scores[clean_labels == 1]
    )["auroc"]
    
    adv_performance = evaluate_ood_detection(
        adversarial_scores[adversarial_labels == 0],
        adversarial_scores[adversarial_labels == 1]
    )["auroc"]
    
    performance_degradation = clean_performance - adv_performance
    
    return {
        "score_correlation": score_correlation,
        "detection_consistency": detection_consistency,
        "performance_degradation": performance_degradation,
    }


def create_evaluation_report(
    method_name: str,
    metrics: Dict[str, float],
    additional_info: Optional[Dict[str, any]] = None,
) -> str:
    """Create a formatted evaluation report.
    
    Args:
        method_name: Name of the OOD detection method.
        metrics: Dictionary of computed metrics.
        additional_info: Additional information to include.
        
    Returns:
        Formatted report string.
    """
    report = f"=== OOD Detection Evaluation Report ===\n"
    report += f"Method: {method_name}\n\n"
    
    report += "Performance Metrics:\n"
    for metric, value in metrics.items():
        if isinstance(value, float):
            report += f"  {metric}: {value:.4f}\n"
        else:
            report += f"  {metric}: {value}\n"
    
    if additional_info:
        report += "\nAdditional Information:\n"
        for key, value in additional_info.items():
            report += f"  {key}: {value}\n"
    
    return report


def compare_methods(
    method_results: Dict[str, Dict[str, float]],
    primary_metric: str = "auroc",
) -> Dict[str, any]:
    """Compare multiple OOD detection methods.
    
    Args:
        method_results: Dictionary mapping method names to their metrics.
        primary_metric: Primary metric for ranking.
        
    Returns:
        Comparison results including rankings and statistics.
    """
    if not method_results:
        return {}
    
    # Extract primary metric values
    primary_values = {
        method: results.get(primary_metric, 0.0)
        for method, results in method_results.items()
    }
    
    # Rank methods by primary metric
    ranked_methods = sorted(
        primary_values.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Compute statistics
    values = list(primary_values.values())
    stats = {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "median": np.median(values),
    }
    
    return {
        "rankings": ranked_methods,
        "statistics": stats,
        "primary_metric": primary_metric,
    }
