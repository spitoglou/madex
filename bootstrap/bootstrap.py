import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.utils import resample

from new_metric.cega import clarke_error_grid

# Configure warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Configure logging for narrative output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(script_dir / 'bootstrap_analysis_narrative.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a separate logger for LLM-friendly narrative
narrative_logger = logging.getLogger('narrative')
narrative_handler = logging.FileHandler(script_dir / 'bootstrap_llm_narrative.log', mode='w', encoding='utf-8')
narrative_handler.setLevel(logging.INFO)
narrative_formatter = logging.Formatter('%(message)s')
narrative_handler.setFormatter(narrative_formatter)
narrative_logger.addHandler(narrative_handler)
narrative_logger.setLevel(logging.INFO)

def log_narrative(message):
    """Helper function to log narrative messages for LLM consumption"""
    narrative_logger.info(message)
    print(f"[NARRATIVE] {message}")

# --- Data from Table 2: Reference, Model A, Model B (each array corresponds to a scenario) ---

reference_values = {
    'A': np.array([45, 55, 48, 52, 58, 61, 54, 50]),
    'B': np.array([280, 310, 295, 275, 290, 305, 285, 300]),
    'C': np.array([120, 180, 220, 195, 160, 140, 125, 115]),
    'D': np.array([85, 95, 110, 125, 140, 155, 150, 145]),
    'E': np.array([140, 120, 95, 75, 65, 80, 100, 125]),
    'F': np.array([100, 105, 98, 102, 99, 103, 101, 97]),
    'G': np.array([65, 180, 45, 250, 95, 200, 55, 160]),
    'H': np.array([25, 35, 400, 450, 30, 420, 40, 380]),
}

model_a_preds = {
    'A': np.array([65, 70, 68, 72, 75, 80, 74, 70]),
    'B': np.array([250, 280, 265, 245, 260, 275, 255, 270]),
    'C': np.array([110, 170, 210, 185, 150, 130, 115, 105]),
    'D': np.array([90, 100, 115, 130, 145, 160, 155, 150]),
    'E': np.array([145, 125, 100, 80, 70, 85, 105, 130]),
    'F': np.array([108, 113, 106, 110, 107, 111, 109, 105]),
    'G': np.array([75, 190, 55, 260, 105, 210, 65, 170]),
    'H': np.array([45, 55, 380, 430, 50, 400, 60, 360]),
}

model_b_preds = {
    'A': np.array([35, 45, 38, 42, 48, 51, 44, 40]),
    'B': np.array([320, 350, 335, 315, 330, 345, 325, 340]),
    'C': np.array([130, 190, 230, 205, 170, 150, 135, 125]),
    'D': np.array([80, 90, 105, 120, 135, 150, 145, 140]),
    'E': np.array([135, 115, 90, 70, 60, 75, 95, 120]),
    'F': np.array([92, 97, 90, 94, 91, 95, 93, 89]),
    'G': np.array([55, 170, 35, 240, 85, 190, 45, 150]),
    'H': np.array([15, 25, 420, 470, 20, 440, 30, 400]),
}

# --- MADEX parameters ---
a = 125.0
b = 55.0
c = 40.0

# --- Functions ---

def compute_madex(y_true, y_pred):
    residuals = np.abs(y_pred - y_true)
    exp_component = 2 - np.tanh((y_true - a)/b) * ((y_pred - y_true)/c)
    madex = np.mean(residuals ** exp_component)
    return madex

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))



def clinical_risk_score(zone_counts):
    """Calculate clinical risk from zone counts [A, B, C, D, E]"""
    weights = [0, 1, 2, 5, 5]  # A, B, C, D, E weights
    score = sum(count * weight for count, weight in zip(zone_counts, weights))
    return score

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, ci=95):
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        # Ensure indices are used consistently for both true and predicted values
        true_boot = y_true[indices]
        pred_boot = y_pred[indices]
        val = metric_func(true_boot, pred_boot)
        values.append(val)
    lower = np.percentile(values, (100 - ci)/2)
    upper = np.percentile(values, 100 - (100 - ci)/2)
    mean_val = np.mean(values)
    return mean_val, lower, upper


def extract_wilcoxon_results(result):
    """Safely extract statistic and p-value from wilcoxon result across scipy versions"""
    try:
        # For newer scipy versions (>=1.7)
        stat = float(result.statistic)
        p_val = float(result.pvalue)
        return stat, p_val
    except AttributeError:
        # For older scipy versions, result is a tuple (statistic, pvalue)
        stat = float(result[0])
        p_val = float(result[1])
        return stat, p_val

def cohens_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

# Clinical detection sensitivity example (hypoglycemia < 70 mg/dL)
def sensitivity_hypoglycemia(true_vals, predicted_vals, threshold=70):
    true_positives = np.sum((true_vals < threshold) & (predicted_vals < threshold))
    total_positives = np.sum(true_vals < threshold)
    return true_positives / total_positives if total_positives > 0 else np.nan

# Clinical detection sensitivity for hyperglycemia (>180 mg/dL)
def sensitivity_hyperglycemia(true_vals, predicted_vals, threshold=180):
    true_positives = np.sum((true_vals > threshold) & (predicted_vals > threshold))
    total_positives = np.sum(true_vals > threshold)
    return true_positives / total_positives if total_positives > 0 else np.nan


# --- Analysis ---

# Initialize narrative logging
log_narrative("\n" + "="*100)
log_narrative("MADEX BOOTSTRAP STATISTICAL VALIDATION ANALYSIS")
log_narrative("="*100)
log_narrative(f"Analysis initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_narrative(f"Analysis type: Bootstrap confidence interval validation for MADEX metric")
log_narrative(f"Clinical context: Glucose prediction model statistical validation")

log_narrative("\nANALYSIS OBJECTIVES:")
log_narrative("1. Generate bootstrap confidence intervals for MADEX across clinical scenarios")
log_narrative("2. Compare MADEX performance against traditional metrics (RMSE, MAE)")
log_narrative("3. Assess clinical risk using Clarke Error Grid Analysis")
log_narrative("4. Evaluate statistical significance of model differences")
log_narrative("5. Analyze hypoglycemia and hyperglycemia detection sensitivity")

# Clinical scenario descriptions
scenario_descriptions = {
    'A': {
        'name': 'Hypoglycemia Detection',
        'description': 'Critical low glucose events (45-61 mg/dL)',
        'clinical_significance': 'Life-threatening hypoglycemia requiring immediate intervention',
        'risk_level': 'CRITICAL'
    },
    'B': {
        'name': 'Hyperglycemia Management', 
        'description': 'Sustained high glucose (275-310 mg/dL)',
        'clinical_significance': 'Severe hyperglycemia requiring aggressive treatment',
        'risk_level': 'HIGH'
    },
    'C': {
        'name': 'Postprandial Response',
        'description': 'Meal-induced glucose spikes (115-220 mg/dL)',
        'clinical_significance': 'Post-meal glucose management challenges',
        'risk_level': 'MODERATE'
    },
    'D': {
        'name': 'Dawn Phenomenon',
        'description': 'Early morning glucose rise (85-155 mg/dL)',
        'clinical_significance': 'Natural circadian glucose elevation',
        'risk_level': 'LOW'
    },
    'E': {
        'name': 'Exercise Response',
        'description': 'Activity-induced glucose drop (65-140 mg/dL)',
        'clinical_significance': 'Exercise-induced hypoglycemia risk',
        'risk_level': 'MODERATE'
    },
    'F': {
        'name': 'Measurement Noise',
        'description': 'Sensor accuracy challenges (97-105 mg/dL)',
        'clinical_significance': 'Measurement precision in normal glucose range',
        'risk_level': 'LOW'
    },
    'G': {
        'name': 'Mixed Clinical',
        'description': 'Combined challenging scenarios (45-250 mg/dL)',
        'clinical_significance': 'Mixed critical events testing robustness',
        'risk_level': 'CRITICAL'
    },
    'H': {
        'name': 'Extreme Cases',
        'description': 'Life-threatening glucose extremes (25-450 mg/dL)',
        'clinical_significance': 'Extreme glucose values requiring emergency care',
        'risk_level': 'CRITICAL'
    }
}

log_narrative("\n" + "="*80)
log_narrative("CLINICAL SCENARIO ANALYSIS")
log_narrative("="*80)

for scenario, desc in scenario_descriptions.items():
    log_narrative(f"\nScenario {scenario}: {desc['name']}")
    log_narrative(f"Description: {desc['description']}")
    log_narrative(f"Risk Level: {desc['risk_level']}")
    log_narrative(f"Clinical Context: {desc['clinical_significance']}")

log_narrative("\n" + "="*80)
log_narrative("BOOTSTRAP ANALYSIS EXECUTION")
log_narrative("="*80)
log_narrative(f"Bootstrap parameters: n_bootstrap=1000, confidence_interval=95%")
log_narrative(f"MADEX parameters: a={a} (euglycemic center), b={b} (critical range), c={c} (slope modifier)")
log_narrative(f"Total scenarios to analyze: {len(reference_values)}")

scenarios = list(reference_values.keys())

results = {}

log_narrative("\nINITIATING SCENARIO-BY-SCENARIO ANALYSIS:")
log_narrative("-"*50)

print(f"{'Scenario':<8} {'MADEX_A':>10} {'MADEX_B':>10} {'RMSE_A':>10} {'RMSE_B':>10} {'MAE_A':>10} {'MAE_B':>10} {'Risk_A':>10} {'Risk_B':>10}")
for i, scenario in enumerate(scenarios, 1):
    y = reference_values[scenario]
    pred_a = model_a_preds[scenario]
    pred_b = model_b_preds[scenario]
    n = len(y)
    
    scenario_info = scenario_descriptions[scenario]
    log_narrative(f"\n[{i}/{len(scenarios)}] ANALYZING SCENARIO {scenario}: {scenario_info['name']}")
    log_narrative(f"Clinical Context: {scenario_info['clinical_significance']}")
    log_narrative(f"Risk Level: {scenario_info['risk_level']}")
    log_narrative(f"Data points: {n}, Glucose range: {y.min():.1f}-{y.max():.1f} mg/dL")
    log_narrative(f"Model A predictions: {pred_a.min():.1f}-{pred_a.max():.1f} mg/dL")
    log_narrative(f"Model B predictions: {pred_b.min():.1f}-{pred_b.max():.1f} mg/dL")

    # Compute metrics with bootstrapping
    log_narrative(f"Computing bootstrap confidence intervals (n=1000)...")
    madex_a, madex_a_lower, madex_a_upper = bootstrap_metric(y, pred_a, compute_madex)
    madex_b, madex_b_lower, madex_b_upper = bootstrap_metric(y, pred_b, compute_madex)
    rmse_a, rmse_a_lower, rmse_a_upper = bootstrap_metric(y, pred_a, compute_rmse)
    rmse_b, rmse_b_lower, rmse_b_upper = bootstrap_metric(y, pred_b, compute_rmse)
    mae_a, mae_a_lower, mae_a_upper = bootstrap_metric(y, pred_a, compute_mae)
    mae_b, mae_b_lower, mae_b_upper = bootstrap_metric(y, pred_b, compute_mae)

    log_narrative(f"Bootstrap results computed:")
    log_narrative(f"  MADEX - Model A: {madex_a:.2f} ({madex_a_lower:.2f}-{madex_a_upper:.2f})")
    log_narrative(f"  MADEX - Model B: {madex_b:.2f} ({madex_b_lower:.2f}-{madex_b_upper:.2f})")
    log_narrative(f"  RMSE - Model A: {rmse_a:.2f} ({rmse_a_lower:.2f}-{rmse_a_upper:.2f})")
    log_narrative(f"  RMSE - Model B: {rmse_b:.2f} ({rmse_b_lower:.2f}-{rmse_b_upper:.2f})")


    # Clarke zones and risk scores
    log_narrative(f"Performing Clarke Error Grid Analysis...")
    plot_a, zones_a = clarke_error_grid(y, pred_a, f"Scenario {scenario} Model A")
    plot_b, zones_b = clarke_error_grid(y, pred_b, f"Scenario {scenario} Model B")
    risk_a = clinical_risk_score(zones_a)
    risk_b = clinical_risk_score(zones_b)
    
    # Log CEGA results with clinical interpretation
    zone_names = ['A (Accurate)', 'B (Acceptable)', 'C (Overcorrecting)', 'D (Failure to Detect)', 'E (Erroneous)']
    log_narrative(f"Clarke Error Grid zones - Model A: {[f'{zone_names[i]}: {count}' for i, count in enumerate(zones_a)]}")
    log_narrative(f"Clarke Error Grid zones - Model B: {[f'{zone_names[i]}: {count}' for i, count in enumerate(zones_b)]}")
    log_narrative(f"Clinical risk scores: Model A = {risk_a}, Model B = {risk_b}")
    
    # Interpret clinical risk
    if risk_a < risk_b:
        log_narrative(f"[CLINICAL ASSESSMENT] Model A has lower clinical risk (better safety profile)")
    elif risk_b < risk_a:
        log_narrative(f"[CLINICAL ASSESSMENT] Model B has lower clinical risk (better safety profile)")
    else:
        log_narrative(f"[CLINICAL ASSESSMENT] Both models have equivalent clinical risk")
    
    # Interpret MADEX results
    if madex_a < madex_b:
        madex_winner = "Model A"
        madex_interpretation = f"Model A shows better MADEX performance ({madex_a:.2f} vs {madex_b:.2f})"
    else:
        madex_winner = "Model B" 
        madex_interpretation = f"Model B shows better MADEX performance ({madex_b:.2f} vs {madex_a:.2f})"
    
    log_narrative(f"[MADEX ASSESSMENT] {madex_interpretation}")
    
    # Check for confidence interval overlap
    ci_overlap_madex = not (madex_a_upper < madex_b_lower or madex_b_upper < madex_a_lower)
    if ci_overlap_madex:
        log_narrative(f"[WARNING] MADEX confidence intervals overlap - difference may not be statistically significant")
    else:
        log_narrative(f"[SUCCESS] MADEX confidence intervals do not overlap - significant difference detected")


    results[scenario] = {
        'MADEX_A': f"{madex_a:.2f} ({madex_a_lower:.2f}-{madex_a_upper:.2f})",
        'MADEX_B': f"{madex_b:.2f} ({madex_b_lower:.2f}-{madex_b_upper:.2f})",
        'RMSE_A': f"{rmse_a:.2f} ({rmse_a_lower:.2f}-{rmse_a_upper:.2f})",
        'RMSE_B': f"{rmse_b:.2f} ({rmse_b_lower:.2f}-{rmse_b_upper:.2f})",
        'MAE_A': f"{mae_a:.2f} ({mae_a_lower:.2f}-{mae_a_upper:.2f})",
        'MAE_B': f"{mae_b:.2f} ({mae_b_lower:.2f}-{mae_b_upper:.2f})",
        'Risk_A': risk_a, 'Risk_B': risk_b,
        'Zones_A': zones_a, 'Zones_B': zones_b,
        'Ref': y, 'Pred_A': pred_a, 'Pred_B': pred_b,
    }


# Print summary table with bootstrapped CIs
print(f"{'Scenario':<8} {'MADEX_A (95% CI)':<25} {'MADEX_B (95% CI)':<25} {'RMSE_A (95% CI)':<20} {'RMSE_B (95% CI)':<20} {'MAE_A (95% CI)':<20} {'MAE_B (95% CI)':<20} {'Risk_A':>10} {'Risk_B':>10}")
for scenario in scenarios:
    res = results[scenario]
    print(f"{scenario:<8} {res['MADEX_A']:<25} {res['MADEX_B']:<25} {res['RMSE_A']:<20} {res['RMSE_B']:<20} {res['MAE_A']:<20} {res['MAE_B']:<20} {res['Risk_A']:10d} {res['Risk_B']:10d}")


# Statistical comparison: Wilcoxon signed-rank test across scenarios for MADEX, RMSE, MAE
# Note: Wilcoxon test is performed on the mean values from bootstrapping for simplicity
log_narrative("\n" + "="*80)
log_narrative("STATISTICAL SIGNIFICANCE TESTING")
log_narrative("="*80)
log_narrative("Performing Wilcoxon signed-rank tests across all scenarios")
log_narrative("Testing hypothesis: Model A vs Model B performance differences")

madex_a_vals = np.array([float(results[s]['MADEX_A'].split(' ')[0]) for s in scenarios])
madex_b_vals = np.array([float(results[s]['MADEX_B'].split(' ')[0]) for s in scenarios])
rmse_a_vals = np.array([float(results[s]['RMSE_A'].split(' ')[0]) for s in scenarios])
rmse_b_vals = np.array([float(results[s]['RMSE_B'].split(' ')[0]) for s in scenarios])
mae_a_vals = np.array([float(results[s]['MAE_A'].split(' ')[0]) for s in scenarios])
mae_b_vals = np.array([float(results[s]['MAE_B'].split(' ')[0]) for s in scenarios])

log_narrative("\nWILCOXON SIGNED-RANK TEST RESULTS:")
log_narrative("-"*45)


print("\nWilcoxon Signed-Rank Test Results (based on bootstrapped means):")
for metric_name, a_vals, b_vals in [('MADEX', madex_a_vals, madex_b_vals),
                                   ('RMSE', rmse_a_vals, rmse_b_vals),
                                   ('MAE', mae_a_vals, mae_b_vals)]:
    result = wilcoxon(a_vals, b_vals)
    stat, p_value = extract_wilcoxon_results(result)
    effect = cohens_d(a_vals, b_vals)
    print(f"{metric_name}: statistic={stat:.2f}, p-value={p_value:.4f}, Cohen's d={effect:.2f}")
    
    # Add narrative interpretation
    log_narrative(f"\n{metric_name} Statistical Analysis:")
    log_narrative(f"  Wilcoxon statistic: {stat:.2f}")
    log_narrative(f"  P-value: {p_value:.4f}")
    log_narrative(f"  Cohen's d (effect size): {effect:.2f}")
    
    # Interpret p-value
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
        log_narrative(f"  [SUCCESS] {significance} - Very strong evidence of difference")
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
        log_narrative(f"  [SUCCESS] {significance} - Strong evidence of difference")
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
        log_narrative(f"  [SUCCESS] {significance} - Evidence of difference")
    elif p_value < 0.10:
        significance = "marginally significant (p < 0.10)"
        log_narrative(f"  [CAUTION] {significance} - Weak evidence of difference")
    else:
        significance = "not significant (p >= 0.10)"
        log_narrative(f"  [WARNING] {significance} - No evidence of difference")
    
    # Interpret effect size
    if abs(effect) < 0.2:
        effect_interpretation = "negligible effect size"
    elif abs(effect) < 0.5:
        effect_interpretation = "small effect size"
    elif abs(effect) < 0.8:
        effect_interpretation = "medium effect size"
    else:
        effect_interpretation = "large effect size"
    
    log_narrative(f"  Effect size interpretation: {effect_interpretation}")
    
    # Model preference interpretation
    if a_vals.mean() < b_vals.mean():
        preferred_model = "Model A"
        log_narrative(f"  [INTERPRETATION] {preferred_model} shows better {metric_name} performance on average")
    else:
        preferred_model = "Model B"
        log_narrative(f"  [INTERPRETATION] {preferred_model} shows better {metric_name} performance on average")


log_narrative("\n" + "="*80)
log_narrative("CLINICAL DETECTION SENSITIVITY ANALYSIS")
log_narrative("="*80)
log_narrative("Evaluating model sensitivity for critical glucose events")
log_narrative("This analysis assesses how well models detect dangerous glucose levels")

log_narrative("\nHYPOGLYCEMIA DETECTION SENSITIVITY (<70 mg/dL):")
log_narrative("-"*50)
log_narrative("Hypoglycemia is life-threatening and requires immediate detection")
log_narrative("Sensitivity = True Positives / (True Positives + False Negatives)")

print("\nHypoglycemia Detection Sensitivity (<70 mg/dL):")
hypo_sensitivities_a = []
hypo_sensitivities_b = []

for scenario in scenarios:
    y = reference_values[scenario]
    pred_a = model_a_preds[scenario]
    pred_b = model_b_preds[scenario]
    sens_a = sensitivity_hypoglycemia(y, pred_a)
    sens_b = sensitivity_hypoglycemia(y, pred_b)
    
    # Store for summary analysis
    if not np.isnan(sens_a):
        hypo_sensitivities_a.append(sens_a)
    if not np.isnan(sens_b):
        hypo_sensitivities_b.append(sens_b)
    
    print(f"Scenario {scenario}: Model A = {sens_a:.2f}, Model B = {sens_b:.2f}")
    
    # Add clinical interpretation
    scenario_info = scenario_descriptions[scenario]
    if not np.isnan(sens_a) and not np.isnan(sens_b):
        log_narrative(f"\nScenario {scenario} ({scenario_info['name']}) Hypoglycemia Detection:")
        log_narrative(f"  Model A sensitivity: {sens_a:.2f} ({sens_a*100:.1f}%)")
        log_narrative(f"  Model B sensitivity: {sens_b:.2f} ({sens_b*100:.1f}%)")
        
        if sens_a > sens_b:
            log_narrative(f"  [CLINICAL] Model A better at detecting hypoglycemia (safer)")
        elif sens_b > sens_a:
            log_narrative(f"  [CLINICAL] Model B better at detecting hypoglycemia (safer)")
        else:
            log_narrative(f"  [CLINICAL] Both models equivalent for hypoglycemia detection")
            
        # Assess clinical adequacy
        if max(sens_a, sens_b) < 0.8:
            log_narrative(f"  [WARNING] Both models show poor hypoglycemia sensitivity (<80%)")
        elif max(sens_a, sens_b) >= 0.9:
            log_narrative(f"  [SUCCESS] Excellent hypoglycemia detection capability (>=90%)")
    elif scenario_info['risk_level'] in ['CRITICAL', 'HIGH']:
        log_narrative(f"\nScenario {scenario} ({scenario_info['name']}): No hypoglycemic events to detect")

log_narrative("\nHYPERGLYCEMIA DETECTION SENSITIVITY (>180 mg/dL):")
log_narrative("-"*50)
log_narrative("Hyperglycemia detection is crucial for preventing diabetic complications")
log_narrative("High glucose levels require prompt intervention to prevent ketoacidosis")

print("\nHyperglycemia Detection Sensitivity (>180 mg/dL):")
hyper_sensitivities_a = []
hyper_sensitivities_b = []

for scenario in scenarios:
    y = reference_values[scenario]
    pred_a = model_a_preds[scenario]
    pred_b = model_b_preds[scenario]
    sens_a = sensitivity_hyperglycemia(y, pred_a)
    sens_b = sensitivity_hyperglycemia(y, pred_b)
    
    # Store for summary analysis
    if not np.isnan(sens_a):
        hyper_sensitivities_a.append(sens_a)
    if not np.isnan(sens_b):
        hyper_sensitivities_b.append(sens_b)
    
    print(f"Scenario {scenario}: Model A = {sens_a:.2f}, Model B = {sens_b:.2f}")
    
    # Add clinical interpretation
    scenario_info = scenario_descriptions[scenario]
    if not np.isnan(sens_a) and not np.isnan(sens_b):
        log_narrative(f"\nScenario {scenario} ({scenario_info['name']}) Hyperglycemia Detection:")
        log_narrative(f"  Model A sensitivity: {sens_a:.2f} ({sens_a*100:.1f}%)")
        log_narrative(f"  Model B sensitivity: {sens_b:.2f} ({sens_b*100:.1f}%)")
        
        if sens_a > sens_b:
            log_narrative(f"  [CLINICAL] Model A better at detecting hyperglycemia")
        elif sens_b > sens_a:
            log_narrative(f"  [CLINICAL] Model B better at detecting hyperglycemia")
        else:
            log_narrative(f"  [CLINICAL] Both models equivalent for hyperglycemia detection")
            
        # Assess clinical adequacy
        if max(sens_a, sens_b) < 0.8:
            log_narrative(f"  [WARNING] Both models show poor hyperglycemia sensitivity (<80%)")
        elif max(sens_a, sens_b) >= 0.9:
            log_narrative(f"  [SUCCESS] Excellent hyperglycemia detection capability (>=90%)")
    elif scenario_info['risk_level'] in ['CRITICAL', 'HIGH']:
        log_narrative(f"\nScenario {scenario} ({scenario_info['name']}): No hyperglycemic events to detect")

# Final comprehensive summary
log_narrative("\n" + "="*100)
log_narrative("BOOTSTRAP VALIDATION ANALYSIS SUMMARY")
log_narrative("="*100)

log_narrative(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_narrative(f"Total scenarios analyzed: {len(scenarios)}")
log_narrative(f"Bootstrap samples per metric: 1000")
log_narrative(f"Confidence interval level: 95%")

log_narrative("\nKEY FINDINGS:")
log_narrative("-"*15)

# MADEX vs RMSE/MAE comparison
madex_result = wilcoxon(madex_a_vals, madex_b_vals)
rmse_result = wilcoxon(rmse_a_vals, rmse_b_vals)
mae_result = wilcoxon(mae_a_vals, mae_b_vals)

# Extract p-values using helper function
_, madex_p = extract_wilcoxon_results(madex_result)
_, rmse_p = extract_wilcoxon_results(rmse_result)
_, mae_p = extract_wilcoxon_results(mae_result)

log_narrative(f"1. MADEX Statistical Significance: p={madex_p:.4f}")
if madex_p < 0.05:
    log_narrative(f"   [SUCCESS] MADEX shows statistically significant model differences")
else:
    log_narrative(f"   [WARNING] MADEX differences not statistically significant")

log_narrative(f"2. Traditional Metrics Comparison:")
log_narrative(f"   RMSE p-value: {rmse_p:.4f}")
log_narrative(f"   MAE p-value: {mae_p:.4f}")

if madex_p < rmse_p and madex_p < mae_p:
    log_narrative(f"   [SUCCESS] MADEX more sensitive than traditional metrics")
else:
    log_narrative(f"   [OBSERVATION] Traditional metrics show similar or better discrimination")

# Clinical detection summary
if hypo_sensitivities_a and hypo_sensitivities_b:
    avg_hypo_a = np.mean(hypo_sensitivities_a)
    avg_hypo_b = np.mean(hypo_sensitivities_b)
    log_narrative(f"3. Hypoglycemia Detection Performance:")
    log_narrative(f"   Model A average sensitivity: {avg_hypo_a:.2f} ({avg_hypo_a*100:.1f}%)")
    log_narrative(f"   Model B average sensitivity: {avg_hypo_b:.2f} ({avg_hypo_b*100:.1f}%)")
    
    if avg_hypo_b > avg_hypo_a:
        log_narrative(f"   [CLINICAL RECOMMENDATION] Model B preferred for hypoglycemia detection")
    else:
        log_narrative(f"   [CLINICAL RECOMMENDATION] Model A preferred for hypoglycemia detection")

if hyper_sensitivities_a and hyper_sensitivities_b:
    avg_hyper_a = np.mean(hyper_sensitivities_a)
    avg_hyper_b = np.mean(hyper_sensitivities_b)
    log_narrative(f"4. Hyperglycemia Detection Performance:")
    log_narrative(f"   Model A average sensitivity: {avg_hyper_a:.2f} ({avg_hyper_a*100:.1f}%)")
    log_narrative(f"   Model B average sensitivity: {avg_hyper_b:.2f} ({avg_hyper_b*100:.1f}%)")

log_narrative("\nCONCLUSIONS:")
log_narrative("-"*12)
log_narrative("This bootstrap validation provides statistical confidence intervals for MADEX")
log_narrative("and traditional metrics across clinically relevant glucose prediction scenarios.")
log_narrative("The analysis enables evidence-based model selection for glucose monitoring systems.")

log_narrative("\n[ANALYSIS COMPLETE] Bootstrap validation analysis finished successfully.")
log_narrative("Results logged to bootstrap_llm_narrative.log for LLM consumption.")

# This template can be expanded with further bootstrap CI, effect size per scenario, and other analyses.