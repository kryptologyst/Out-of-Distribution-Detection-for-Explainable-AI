#!/usr/bin/env python3
"""Quick test script to verify the OOD detection implementation."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        from ood_detection.data import load_iris_dataset, preprocess_data
        from ood_detection.methods import MahalanobisOODDetector, EnergyOODDetector
        from ood_detection.eval import evaluate_ood_detection
        from ood_detection.utils import set_seed
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic OOD detection functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from ood_detection.data import load_iris_dataset
        from ood_detection.methods import MahalanobisOODDetector
        from ood_detection.eval import evaluate_ood_detection
        from ood_detection.utils import set_seed
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Load data
        X_train, X_test, y_train, y_test, X_ood = load_iris_dataset(
            test_size=0.3, ood_size=50, random_state=42
        )
        
        # Train detector
        detector = MahalanobisOODDetector()
        detector.fit(X_train, y_train)
        
        # Compute scores
        in_dist_scores = detector.score_samples(X_test)
        ood_scores = detector.score_samples(X_ood)
        
        # Evaluate
        metrics = evaluate_ood_detection(in_dist_scores, ood_scores)
        
        print(f"✅ Basic functionality test passed")
        print(f"   AUROC: {metrics['auroc']:.3f}")
        print(f"   AUPRC: {metrics['auprc']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_energy_detector():
    """Test Energy OOD detector."""
    print("\nTesting Energy detector...")
    
    try:
        from ood_detection.data import load_iris_dataset
        from ood_detection.methods import EnergyOODDetector
        from ood_detection.eval import evaluate_ood_detection
        from ood_detection.utils import set_seed
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Load data
        X_train, X_test, y_train, y_test, X_ood = load_iris_dataset(
            test_size=0.3, ood_size=50, random_state=42
        )
        
        # Train detector
        detector = EnergyOODDetector(max_iter=100)  # Reduced for speed
        detector.fit(X_train, y_train)
        
        # Compute scores
        in_dist_scores = detector.score_samples(X_test)
        ood_scores = detector.score_samples(X_ood)
        
        # Evaluate
        metrics = evaluate_ood_detection(in_dist_scores, ood_scores)
        
        print(f"✅ Energy detector test passed")
        print(f"   AUROC: {metrics['auroc']:.3f}")
        print(f"   AUPRC: {metrics['auprc']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Energy detector test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 OOD Detection Implementation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_energy_detector,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
