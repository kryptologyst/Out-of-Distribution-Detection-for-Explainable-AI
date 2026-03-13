"""Visualization utilities for OOD detection results."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def plot_ood_scores(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str = "OOD Detection",
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot distribution of OOD scores for in-distribution and OOD samples.
    
    Args:
        in_dist_scores: OOD scores for in-distribution samples.
        ood_scores: OOD scores for out-of-distribution samples.
        method_name: Name of the detection method.
        bins: Number of histogram bins.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(
        in_dist_scores,
        bins=bins,
        alpha=0.7,
        label="In-Distribution",
        color="blue",
        density=True,
    )
    ax.hist(
        ood_scores,
        bins=bins,
        alpha=0.7,
        label="Out-of-Distribution",
        color="red",
        density=True,
    )
    
    ax.set_xlabel("OOD Score")
    ax.set_ylabel("Density")
    ax.set_title(f"OOD Score Distribution - {method_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_roc_curve(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str = "OOD Detection",
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """Plot ROC curve for OOD detection.
    
    Args:
        in_dist_scores: OOD scores for in-distribution samples.
        ood_scores: OOD scores for out-of-distribution samples.
        method_name: Name of the detection method.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    from sklearn.metrics import roc_curve, auc
    
    # Create binary labels and scores
    y_true = np.concatenate([
        np.zeros(len(in_dist_scores)),
        np.ones(len(ood_scores))
    ])
    y_scores = np.concatenate([in_dist_scores, ood_scores])
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {method_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_precision_recall_curve(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str = "OOD Detection",
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """Plot precision-recall curve for OOD detection.
    
    Args:
        in_dist_scores: OOD scores for in-distribution samples.
        ood_scores: OOD scores for out-of-distribution samples.
        method_name: Name of the detection method.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Create binary labels and scores
    y_true = np.concatenate([
        np.zeros(len(in_dist_scores)),
        np.ones(len(ood_scores))
    ])
    y_scores = np.concatenate([in_dist_scores, ood_scores])
    
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {avg_precision:.3f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {method_name}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_method_comparison(
    method_results: Dict[str, Dict[str, float]],
    metric: str = "auroc",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot comparison of multiple OOD detection methods.
    
    Args:
        method_results: Dictionary mapping method names to their metrics.
        metric: Metric to compare.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    methods = list(method_results.keys())
    values = [method_results[method].get(metric, 0.0) for method in methods]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(methods, values, color="skyblue", edgecolor="navy", alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    
    ax.set_ylabel(metric.upper())
    ax.set_title(f"OOD Detection Method Comparison - {metric.upper()}")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")
    
    # Rotate x-axis labels if needed
    if len(max(methods, key=len)) > 10:
        plt.xticks(rotation=45, ha="right")
    
    return fig


def plot_calibration_diagram(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    method_name: str = "Calibration",
    num_bins: int = 10,
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    """Plot calibration diagram (reliability diagram).
    
    Args:
        predictions: Predicted probabilities.
        true_labels: True binary labels.
        method_name: Name of the method.
        num_bins: Number of bins for calibration.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(prop_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot calibration curve
    ax.plot(bin_confidences, bin_accuracies, "o-", label="Calibration curve")
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    
    # Add bin counts as bar heights
    ax2 = ax.twinx()
    ax2.bar(bin_centers, bin_counts, alpha=0.3, color="gray", label="Bin counts")
    
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Fraction of samples")
    ax.set_title(f"Calibration Diagram - {method_name}")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    return fig


def create_summary_plot(
    method_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["auroc", "auprc", "fpr_at_95_tpr"],
    figsize: Tuple[int, int] = (15, 5),
) -> Figure:
    """Create a summary plot with multiple metrics.
    
    Args:
        method_results: Dictionary mapping method names to their metrics.
        metrics: List of metrics to plot.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    methods = list(method_results.keys())
    
    for i, metric in enumerate(metrics):
        values = [method_results[method].get(metric, 0.0) for method in methods]
        
        bars = axes[i].bar(methods, values, color="lightcoral", edgecolor="darkred", alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
        
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f"{metric.upper()}")
        axes[i].set_ylim([0, 1])
        axes[i].grid(True, alpha=0.3, axis="y")
        
        if len(max(methods, key=len)) > 10:
            axes[i].tick_params(axis="x", rotation=45)
    
    fig.suptitle("OOD Detection Method Comparison", fontsize=16)
    plt.tight_layout()
    
    return fig
