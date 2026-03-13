"""Streamlit demo for OOD detection."""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ood_detection.data import load_iris_dataset, load_synthetic_dataset, preprocess_data
from ood_detection.methods import MahalanobisOODDetector, EnergyOODDetector
from ood_detection.eval import evaluate_ood_detection
from ood_detection.viz import plot_ood_scores, plot_roc_curve, plot_precision_recall_curve
from ood_detection.utils import set_seed


# Page configuration
st.set_page_config(
    page_title="OOD Detection Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_demo_data(dataset_name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset for demo."""
    if dataset_name == "iris":
        return load_iris_dataset(**kwargs)
    elif dataset_name == "synthetic":
        return load_synthetic_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_sample_input() -> np.ndarray:
    """Create sample input for testing."""
    return np.array([[5.1, 3.5, 1.4, 0.2]])  # Sample iris data


def main():
    """Main demo function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Out-of-Distribution Detection Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p>This demo is for research and educational purposes only. OOD detection methods may be unstable 
    and should not be used for regulated decisions without human review. Results may vary across 
    different random seeds and datasets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["iris", "synthetic"],
        help="Choose the dataset for OOD detection"
    )
    
    # Method selection
    methods = st.sidebar.multiselect(
        "Select OOD Detection Methods",
        ["Mahalanobis", "Energy"],
        default=["Mahalanobis", "Energy"],
        help="Choose which methods to compare"
    )
    
    # Parameters
    st.sidebar.subheader("Parameters")
    
    test_size = st.sidebar.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Fraction of data used for testing"
    )
    
    ood_size = st.sidebar.slider(
        "OOD Sample Size",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Number of out-of-distribution samples to generate"
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000,
        value=42,
        help="Random seed for reproducibility"
    )
    
    # Load data
    with st.spinner("Loading data..."):
        set_seed(random_seed)
        
        try:
            X_train, X_test, y_train, y_test, X_ood = load_demo_data(
                dataset_name,
                test_size=test_size,
                ood_size=ood_size,
                random_state=random_seed
            )
            
            # Preprocess data
            X_train_scaled, X_test_scaled, X_ood_scaled, scaler = preprocess_data(
                X_train, X_test, X_ood
            )
            
            st.success(f"✅ Data loaded successfully!")
            st.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}, OOD samples: {len(X_ood)}")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Detection", "📈 Evaluation", "🧪 Interactive"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            
            # Create summary statistics
            stats_data = {
                "Metric": ["Training Samples", "Test Samples", "OOD Samples", "Features", "Classes"],
                "Value": [
                    len(X_train),
                    len(X_test),
                    len(X_ood),
                    X_train.shape[1],
                    len(np.unique(y_train))
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.subheader("Feature Distribution")
            
            # Plot feature distributions
            fig, ax = plt.subplots(figsize=(8, 6))
            
            feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
            if dataset_name == "iris":
                feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
            
            for i, feature_name in enumerate(feature_names[:X_train.shape[1]]):
                ax.hist(X_train[:, i], alpha=0.7, label=f"{feature_name} (Train)", bins=20)
                ax.hist(X_ood[:, i], alpha=0.7, label=f"{feature_name} (OOD)", bins=20)
            
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Feature Distributions")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    with tab2:
        st.header("OOD Detection Results")
        
        if not methods:
            st.warning("Please select at least one method in the sidebar.")
            return
        
        # Train and evaluate methods
        method_results = {}
        
        for method_name in methods:
            with st.spinner(f"Training {method_name} detector..."):
                try:
                    # Initialize detector
                    if method_name == "Mahalanobis":
                        detector = MahalanobisOODDetector(regularization=1e-6)
                    elif method_name == "Energy":
                        detector = EnergyOODDetector(
                            hidden_layer_sizes=(100, 50),
                            learning_rate=0.001,
                            max_iter=1000
                        )
                    else:
                        continue
                    
                    # Train detector
                    detector.fit(X_train_scaled, y_train)
                    
                    # Compute scores
                    in_dist_scores = detector.score_samples(X_test_scaled)
                    ood_scores = detector.score_samples(X_ood_scaled)
                    
                    # Evaluate performance
                    metrics = evaluate_ood_detection(in_dist_scores, ood_scores)
                    method_results[method_name] = {
                        "metrics": metrics,
                        "in_dist_scores": in_dist_scores,
                        "ood_scores": ood_scores
                    }
                    
                    st.success(f"✅ {method_name} trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error training {method_name}: {e}")
                    continue
        
        if not method_results:
            st.error("No methods were successfully trained.")
            return
        
        # Display results
        st.subheader("Performance Metrics")
        
        # Create metrics comparison table
        metrics_data = []
        for method_name, results in method_results.items():
            row = {"Method": method_name}
            row.update(results["metrics"])
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Visualizations")
        
        for method_name, results in method_results.items():
            st.write(f"**{method_name} Results:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution plot
                fig_scores = plot_ood_scores(
                    results["in_dist_scores"],
                    results["ood_scores"],
                    method_name
                )
                st.pyplot(fig_scores)
            
            with col2:
                # ROC curve
                fig_roc = plot_roc_curve(
                    results["in_dist_scores"],
                    results["ood_scores"],
                    method_name
                )
                st.pyplot(fig_roc)
    
    with tab3:
        st.header("Detailed Evaluation")
        
        if not method_results:
            st.warning("Please run detection methods first.")
            return
        
        # Method comparison
        st.subheader("Method Comparison")
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods_list = list(method_results.keys())
        auroc_values = [method_results[method]["metrics"]["auroc"] for method in methods_list]
        
        bars = ax.bar(methods_list, auroc_values, color="skyblue", edgecolor="navy", alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, auroc_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01,
                   f"{value:.3f}", ha="center", va="bottom")
        
        ax.set_ylabel("AUROC")
        ax.set_title("OOD Detection Method Comparison")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")
        
        st.pyplot(fig)
        
        # Detailed metrics
        st.subheader("Detailed Metrics")
        
        for method_name, results in method_results.items():
            with st.expander(f"{method_name} Details"):
                metrics = results["metrics"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("AUROC", f"{metrics['auroc']:.4f}")
                with col2:
                    st.metric("AUPRC", f"{metrics['auprc']:.4f}")
                with col3:
                    st.metric("FPR@95% TPR", f"{metrics['fpr_at_95_tpr']:.4f}")
                with col4:
                    st.metric("Detection Error", f"{metrics['detection_error']:.4f}")
    
    with tab4:
        st.header("Interactive Testing")
        
        st.subheader("Test Custom Samples")
        
        # Sample input interface
        st.write("Enter custom feature values to test OOD detection:")
        
        if dataset_name == "iris":
            feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
            default_values = [5.1, 3.5, 1.4, 0.2]
        else:
            feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
            default_values = [0.0, 0.0, 0.0, 0.0]
        
        # Create input fields
        input_values = []
        cols = st.columns(len(feature_names))
        
        for i, (col, feature_name, default_val) in enumerate(zip(cols, feature_names, default_values)):
            with col:
                value = st.number_input(
                    feature_name,
                    value=default_val,
                    step=0.1,
                    key=f"input_{i}"
                )
                input_values.append(value)
        
        # Test button
        if st.button("🔍 Test OOD Detection", type="primary"):
            if not method_results:
                st.warning("Please train methods first in the Detection tab.")
            else:
                # Prepare input
                test_sample = np.array([input_values]).reshape(1, -1)
                test_sample_scaled = scaler.transform(test_sample)
                
                # Test with each method
                st.subheader("Results")
                
                for method_name, results in method_results.items():
                    # Get detector (we need to retrain or store it)
                    if method_name == "Mahalanobis":
                        detector = MahalanobisOODDetector(regularization=1e-6)
                    elif method_name == "Energy":
                        detector = EnergyOODDetector(
                            hidden_layer_sizes=(100, 50),
                            learning_rate=0.001,
                            max_iter=1000
                        )
                    
                    detector.fit(X_train_scaled, y_train)
                    
                    # Compute score
                    score = detector.score_samples(test_sample_scaled)[0]
                    
                    # Determine if OOD
                    threshold = np.median(results["in_dist_scores"])
                    is_ood = score > threshold
                    
                    # Display result
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Method", method_name)
                    with col2:
                        st.metric("OOD Score", f"{score:.4f}")
                    with col3:
                        status = "🚨 OOD" if is_ood else "✅ In-Distribution"
                        st.metric("Prediction", status)
                    
                    # Progress bar for score
                    progress_value = min(score / (threshold * 2), 1.0)
                    st.progress(progress_value)
                    st.caption(f"Threshold: {threshold:.4f}")
                    
                    st.divider()


if __name__ == "__main__":
    main()
