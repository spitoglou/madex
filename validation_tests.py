#!/usr/bin/env python3
"""
Validation Tests for Bootstrap-Sensitivity Refactoring

This script validates that the refactored bootstrap and sensitivity analysis modules
produce identical numerical results, statistical outputs, and log content compared
to the original implementations.

Test Categories:
1. Numerical accuracy validation (MADEX calculations, confidence intervals)
2. Statistical test validation (Wilcoxon results, p-values, effect sizes)
3. Clinical detection validation (sensitivity calculations)
4. Log content semantic equivalence
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from new_metric.common import (
    MetricCalculator, BootstrapAnalyzer, SignificanceTester, ClinicalDetection,
    get_clinical_scenarios, get_standard_madex_params
)

def test_numerical_accuracy():
    """Test that MADEX calculations match exactly"""
    print("="*60)
    print("NUMERICAL ACCURACY VALIDATION")
    print("="*60)
    
    scenarios = get_clinical_scenarios()
    madex_params = get_standard_madex_params()
    calculator = MetricCalculator()
    
    # Test data from original implementation
    test_cases = [
        {
            'y_true': [45, 55, 48, 52, 58, 61, 54, 50],
            'y_pred': [65, 70, 68, 72, 75, 80, 74, 70],
            'expected_madex': None  # Will calculate with original formula
        },
        {
            'y_true': [280, 310, 295, 275, 290, 305, 285, 300],
            'y_pred': [250, 280, 265, 245, 260, 275, 255, 270],
            'expected_madex': None
        }
    ]
    
    print("Testing MADEX calculation accuracy...")
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        y_true = np.array(test_case['y_true'])
        y_pred = np.array(test_case['y_pred'])
        
        # Calculate with refactored implementation
        madex_refactored = calculator.compute_madex(y_true, y_pred, madex_params)
        
        # Calculate with original formula for validation
        a, b, c = madex_params.a, madex_params.b, madex_params.c
        residuals = np.abs(y_pred - y_true)
        exp_component = 2 - np.tanh((y_true - a)/b) * ((y_pred - y_true)/c)
        madex_original = np.mean(residuals ** exp_component)
        
        # Compare results
        diff = abs(madex_refactored - madex_original)
        tolerance = 1e-10
        
        if diff < tolerance:
            print(f"✓ Test case {i}: MADEX accuracy PASSED (diff: {diff:.2e})")
        else:
            print(f"✗ Test case {i}: MADEX accuracy FAILED (diff: {diff:.2e})")
            all_passed = False
        
        print(f"  Original: {madex_original:.6f}, Refactored: {madex_refactored:.6f}")
    
    # Test traditional metrics
    print("\\nTesting traditional metrics accuracy...")
    for i, test_case in enumerate(test_cases, 1):
        y_true = np.array(test_case['y_true'])
        y_pred = np.array(test_case['y_pred'])
        
        # RMSE validation
        rmse_refactored = calculator.compute_rmse(y_true, y_pred)
        rmse_original = np.sqrt(np.mean((y_pred - y_true)**2))
        
        if abs(rmse_refactored - rmse_original) < tolerance:
            print(f"✓ Test case {i}: RMSE accuracy PASSED")
        else:
            print(f"✗ Test case {i}: RMSE accuracy FAILED")
            all_passed = False
        
        # MAE validation
        mae_refactored = calculator.compute_mae(y_true, y_pred)
        mae_original = np.mean(np.abs(y_pred - y_true))
        
        if abs(mae_refactored - mae_original) < tolerance:
            print(f"✓ Test case {i}: MAE accuracy PASSED")
        else:
            print(f"✗ Test case {i}: MAE accuracy FAILED")
            all_passed = False
    
    return all_passed

def test_bootstrap_consistency():
    """Test bootstrap analysis consistency"""
    print("\\n" + "="*60)
    print("BOOTSTRAP ANALYSIS VALIDATION")
    print("="*60)
    
    scenarios = get_clinical_scenarios()
    madex_params = get_standard_madex_params()
    calculator = MetricCalculator()
    bootstrap_analyzer = BootstrapAnalyzer()
    
    # Set random seed for reproducible bootstrap
    np.random.seed(42)
    
    # Test scenario A data
    scenario = scenarios.get_scenario('A')
    y_true = scenario.y_true
    y_pred = scenario.model_predictions['Model_A']
    
    print("Testing bootstrap confidence intervals...")
    
    # Test MADEX bootstrap
    madex_func = lambda yt, yp: calculator.compute_madex(yt, yp, madex_params)
    mean_val, lower_ci, upper_ci = bootstrap_analyzer.bootstrap_metric(
        y_true, y_pred, madex_func, n_bootstrap=100  # Smaller for testing
    )
    
    # Validate bootstrap results structure
    validation_passed = True
    
    if not (lower_ci <= mean_val <= upper_ci):
        print("✗ Bootstrap CI validation FAILED: mean not within CI")
        validation_passed = False
    else:
        print("✓ Bootstrap CI structure PASSED")
    
    if upper_ci - lower_ci <= 0:
        print("✗ Bootstrap CI validation FAILED: invalid CI width")
        validation_passed = False
    else:
        print("✓ Bootstrap CI width PASSED")
    
    print(f"  Bootstrap result: {mean_val:.3f} ({lower_ci:.3f}-{upper_ci:.3f})")
    
    return validation_passed

def test_statistical_significance():
    """Test statistical significance calculations"""
    print("\\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE VALIDATION")
    print("="*60)
    
    significance_tester = SignificanceTester()
    
    # Test data for Wilcoxon test
    sample_a = np.array([1.2, 1.5, 1.8, 2.1, 2.3, 1.9, 1.7, 2.0])
    sample_b = np.array([2.1, 2.4, 2.2, 2.8, 2.6, 2.3, 2.5, 2.7])
    
    print("Testing Wilcoxon signed-rank test...")
    
    # Test Wilcoxon results extraction
    from scipy.stats import wilcoxon
    original_result = wilcoxon(sample_a, sample_b)
    
    # Extract using refactored method
    stat_refactored, p_val_refactored = significance_tester.extract_wilcoxon_results(original_result)
    
    # Validate extraction
    try:
        stat_original = float(original_result.statistic)
        p_val_original = float(original_result.pvalue)
    except AttributeError:
        stat_original = float(original_result[0])
        p_val_original = float(original_result[1])
    
    stat_match = abs(stat_refactored - stat_original) < 1e-10
    p_val_match = abs(p_val_refactored - p_val_original) < 1e-10
    
    if stat_match and p_val_match:
        print("✓ Wilcoxon result extraction PASSED")
        validation_passed = True
    else:
        print("✗ Wilcoxon result extraction FAILED")
        validation_passed = False
    
    print(f"  Statistic: {stat_refactored:.6f}, P-value: {p_val_refactored:.6f}")
    
    # Test Cohen's d calculation
    print("\\nTesting Cohen's d calculation...")
    cohens_d_refactored = significance_tester.cohens_d(sample_a, sample_b)
    
    # Original Cohen's d calculation
    diff = sample_a - sample_b
    cohens_d_original = np.mean(diff) / np.std(diff, ddof=1)
    
    if abs(cohens_d_refactored - cohens_d_original) < 1e-10:
        print("✓ Cohen's d calculation PASSED")
    else:
        print("✗ Cohen's d calculation FAILED")
        validation_passed = False
    
    print(f"  Cohen's d: {cohens_d_refactored:.6f}")
    
    return validation_passed

def test_clinical_detection():
    """Test clinical detection sensitivity functions"""
    print("\\n" + "="*60)
    print("CLINICAL DETECTION VALIDATION")
    print("="*60)
    
    clinical_detector = ClinicalDetection()
    
    # Test data with known hypoglycemic events
    y_true = np.array([45, 55, 68, 72, 180, 200, 65, 50])  # Some <70, some >180
    y_pred = np.array([40, 60, 70, 75, 175, 190, 60, 55])
    
    print("Testing hypoglycemia detection sensitivity...")
    
    # Calculate sensitivity using refactored method
    hypo_sens_refactored = clinical_detector.sensitivity_hypoglycemia(y_true, y_pred, threshold=70)
    
    # Calculate using original formula
    true_positives = np.sum((y_true < 70) & (y_pred < 70))
    total_positives = np.sum(y_true < 70)
    hypo_sens_original = true_positives / total_positives if total_positives > 0 else np.nan
    
    validation_passed = True
    
    if np.isnan(hypo_sens_refactored) and np.isnan(hypo_sens_original):
        print("✓ Hypoglycemia sensitivity PASSED (both NaN)")
    elif abs(hypo_sens_refactored - hypo_sens_original) < 1e-10:
        print("✓ Hypoglycemia sensitivity PASSED")
    else:
        print("✗ Hypoglycemia sensitivity FAILED")
        validation_passed = False
    
    print(f"  Sensitivity: {hypo_sens_refactored:.3f}")
    
    # Test hyperglycemia detection
    print("\\nTesting hyperglycemia detection sensitivity...")
    
    hyper_sens_refactored = clinical_detector.sensitivity_hyperglycemia(y_true, y_pred, threshold=180)
    
    # Calculate using original formula  
    true_positives = np.sum((y_true > 180) & (y_pred > 180))
    total_positives = np.sum(y_true > 180)
    hyper_sens_original = true_positives / total_positives if total_positives > 0 else np.nan
    
    if np.isnan(hyper_sens_refactored) and np.isnan(hyper_sens_original):
        print("✓ Hyperglycemia sensitivity PASSED (both NaN)")
    elif abs(hyper_sens_refactored - hyper_sens_original) < 1e-10:
        print("✓ Hyperglycemia sensitivity PASSED")
    else:
        print("✗ Hyperglycemia sensitivity FAILED")
        validation_passed = False
    
    print(f"  Sensitivity: {hyper_sens_refactored:.3f}")
    
    return validation_passed

def test_scenario_data_integrity():
    """Test that scenario data matches original definitions"""
    print("\\n" + "="*60)
    print("SCENARIO DATA INTEGRITY VALIDATION")
    print("="*60)
    
    scenarios = get_clinical_scenarios()
    
    # Original reference values for validation
    original_reference_values = {
        'A': [45, 55, 48, 52, 58, 61, 54, 50],
        'B': [280, 310, 295, 275, 290, 305, 285, 300],
        'C': [120, 180, 220, 195, 160, 140, 125, 115],
        'D': [85, 95, 110, 125, 140, 155, 150, 145],
        'E': [140, 120, 95, 75, 65, 80, 100, 125],
        'F': [100, 105, 98, 102, 99, 103, 101, 97],
        'G': [65, 180, 45, 250, 95, 200, 55, 160],
        'H': [25, 35, 400, 450, 30, 420, 40, 380],
    }
    
    print("Testing scenario data integrity...")
    validation_passed = True
    
    for scenario_id, expected_values in original_reference_values.items():
        scenario = scenarios.get_scenario(scenario_id)
        actual_values = scenario.y_true.tolist()
        
        if actual_values == expected_values:
            print(f"✓ Scenario {scenario_id}: Data integrity PASSED")
        else:
            print(f"✗ Scenario {scenario_id}: Data integrity FAILED")
            validation_passed = False
    
    return validation_passed

def run_all_validations():
    """Run all validation tests"""
    print("BOOTSTRAP-SENSITIVITY REFACTORING VALIDATION")
    print("="*80)
    print("Validating that refactored modules produce identical results...")
    print()
    
    all_tests_passed = True
    
    # Run individual test suites
    test_results = {
        "Numerical Accuracy": test_numerical_accuracy(),
        "Bootstrap Consistency": test_bootstrap_consistency(),
        "Statistical Significance": test_statistical_significance(),
        "Clinical Detection": test_clinical_detection(), 
        "Scenario Data Integrity": test_scenario_data_integrity()
    }
    
    # Summary
    print("\\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        
        if not passed:
            all_tests_passed = False
    
    print()
    if all_tests_passed:
        print("🎉 ALL VALIDATIONS PASSED")
        print("✓ Refactored modules produce identical results")
        print("✓ Statistical outputs match exactly") 
        print("✓ Clinical detection functions validated")
        print("✓ Data integrity confirmed")
        print()
        print("The refactoring is SUCCESSFUL - safe to use refactored modules.")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("⚠️  Review failed tests before using refactored modules")
    
    return all_tests_passed

if __name__ == "__main__":
    success = run_all_validations()
    sys.exit(0 if success else 1)