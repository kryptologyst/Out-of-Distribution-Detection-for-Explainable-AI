# Out-of-Distribution Detection for Explainable AI

## DISCLAIMER

**IMPORTANT**: This project is for research and educational purposes only. XAI outputs may be unstable or misleading and should not be used for regulated decisions without human review. The methods implemented here are not guaranteed to be reliable in production environments.

## Overview

This project implements state-of-the-art Out-of-Distribution (OOD) detection methods for explainable AI systems. OOD detection is crucial for robust AI systems as it identifies when models encounter data significantly different from their training distribution, enabling safe handling of uncertain predictions.

## Features

- **Multiple OOD Detection Methods**: Energy-based, Mahalanobis distance, uncertainty-based, and adversarial approaches
- **Comprehensive Evaluation**: AUROC/AUPRC metrics, calibration analysis, robustness testing
- **Interactive Demo**: Streamlit-based interface for exploring OOD detection capabilities
- **Production-Ready**: Type hints, comprehensive testing, reproducible results

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.ood_detection.methods import EnergyOODDetector
from src.ood_detection.data import load_iris_dataset

# Load data
X_train, X_test, X_ood = load_iris_dataset()

# Initialize detector
detector = EnergyOODDetector()

# Train and evaluate
detector.fit(X_train)
scores = detector.score_samples(X_test)
ood_scores = detector.score_samples(X_ood)

# Evaluate performance
from src.ood_detection.eval import evaluate_ood_detection
metrics = evaluate_ood_detection(scores, ood_scores)
print(f"AUROC: {metrics['auroc']:.3f}")
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Dataset Schema

The project uses the Iris dataset with the following metadata:

- **Features**: sepal_length, sepal_width, petal_length, petal_width (all continuous)
- **Target**: species (categorical: setosa, versicolor, virginica)
- **Sensitive Attributes**: None
- **Monotonicity Constraints**: None

## Training and Evaluation

### Command Line Interface

```bash
# Train and evaluate all methods
python scripts/train_evaluate.py --config configs/default.yaml

# Run specific method
python scripts/train_evaluate.py --method energy --dataset iris
```

### Configuration

See `configs/default.yaml` for configuration options including:
- Model parameters
- Evaluation metrics
- Visualization settings
- Random seeds

## Methods Implemented

1. **Energy-Based Detection**: Uses energy scores from neural network outputs
2. **Mahalanobis Distance**: Statistical distance-based detection
3. **Uncertainty-Based**: Leverages model uncertainty estimates
4. **Adversarial Detection**: Uses adversarial examples for OOD detection

## Evaluation Metrics

- **AUROC**: Area Under ROC Curve for OOD detection
- **AUPRC**: Area Under Precision-Recall Curve
- **Calibration**: Expected Calibration Error (ECE)
- **Robustness**: Performance under adversarial attacks

## Limitations

- Explanations may be unstable across different random seeds
- Performance may vary significantly across different datasets
- Methods assume certain distributional properties that may not hold in practice
- Not suitable for high-stakes decisions without human oversight

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
# Out-of-Distribution-Detection-for-Explainable-AI
