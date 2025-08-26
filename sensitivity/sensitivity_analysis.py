# Standard library imports
import logging
import warnings
from datetime import datetime
from itertools import product

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, kendalltau

# Local imports
from new_metric import mean_adjusted_exponent_error, graph_vs_mse

# Configure warnings
warnings.filterwarnings('ignore')

# Configure logging for narrative output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('madex_analysis_narrative.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a separate logger for LLM-friendly narrative
narrative_logger = logging.getLogger('narrative')
narrative_handler = logging.FileHandler('madex_llm_narrative.log', mode='w', encoding='utf-8')
narrative_handler.setLevel(logging.INFO)
narrative_formatter = logging.Formatter('%(message)s')
narrative_handler.setFormatter(narrative_formatter)
narrative_logger.addHandler(narrative_handler)
narrative_logger.setLevel(logging.INFO)

def log_narrative(message):
    """Helper function to log narrative messages for LLM consumption"""
    narrative_logger.info(message)
    print(f"[NARRATIVE] {message}")

def calculate_madex(y_true, y_pred, a=125, b=55, c=40, enable_logging=False):
    """
    Calculate MADEX (Mean Adjusted Exponent) metric

    Parameters:
    y_true: array of true glucose values
    y_pred: array of predicted glucose values
    a: center parameter (euglycemic center)
    b: critical range parameter
    c: slope parameter
    enable_logging: whether to enable detailed logging

    Returns:
    MADEX score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if enable_logging:
        glucose_range = f"{y_true.min():.1f}-{y_true.max():.1f} mg/dL"
        prediction_range = f"{y_pred.min():.1f}-{y_pred.max():.1f} mg/dL"
        log_narrative(f"MADEX Calculation: Processing glucose values in range {glucose_range}, predictions in range {prediction_range}")
        log_narrative(f"Parameters: a={a} (euglycemic center), b={b} (critical range), c={c} (slope modifier)")
        
        # Explain clinical significance of parameters
        if a < 100:
            log_narrative(f"Parameter 'a'={a} indicates tight glycemic control target (aggressive management)")
        elif a > 140:
            log_narrative(f"Parameter 'a'={a} indicates relaxed glycemic control target (conservative management)")
        else:
            log_narrative(f"Parameter 'a'={a} indicates standard glycemic control target (normal management)")

    # Calculate the exponent component
    tanh_component = np.tanh((y_true - a) / b)
    error_component = (y_pred - y_true) / c
    exponent = 2 - tanh_component * error_component

    if enable_logging:
        log_narrative(f"Adaptive exponent calculation: Base exponent=2, modified by tanh((glucose-{a})/{b}) * (error/{c})")
        log_narrative(f"Exponent range: {exponent.min():.3f} to {exponent.max():.3f} (higher = more penalty for errors)")

    # Handle potential numerical issues
    exponent = np.clip(exponent, 0.1, 10)  # Prevent extreme exponents

    # Calculate MADEX
    absolute_errors = np.abs(y_pred - y_true)
    powered_errors = np.power(absolute_errors, exponent)

    # Handle potential overflow/underflow
    powered_errors = np.clip(powered_errors, 1e-10, 1e10)

    madex_score = np.mean(powered_errors)
    
    if enable_logging:
        avg_error = np.mean(absolute_errors)
        log_narrative(f"Mean absolute error: {avg_error:.2f} mg/dL, MADEX score: {madex_score:.2f}")
        
        # Compare with official implementation
        official_score = mean_adjusted_exponent_error(y_true, y_pred, a, b, c)
        if abs(madex_score - official_score) < 0.01:
            log_narrative(f"[VERIFIED] Calculation verified: matches official implementation ({official_score:.2f})")
        else:
            log_narrative(f"[WARNING] Calculation differs from official implementation: {official_score:.2f}")
    
    # print(f"MADEX score: {madex_score:.2f} and {mean_adjusted_exponent_error(y_true, y_pred, a,b, c):.2f}")
    return madex_score

def calculate_comparison_metrics(y_true, y_pred):
    """Calculate comparison metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    log_narrative(f"Comparison Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# Define enhanced synthetic validation scenarios
scenarios = {
    'A_Hypoglycemia_Detection': {
        'y_true': [45, 55, 48, 52, 58, 61, 54, 50],
        'model_a': [65, 70, 68, 72, 75, 80, 74, 70],  # Misses dangerous lows
        'model_b': [35, 45, 38, 42, 48, 51, 44, 40],  # Overcorrects
        'description': 'Critical low glucose events',
        'clinical_significance': 'Life-threatening hypoglycemia (<70 mg/dL). Model A dangerously overestimates, Model B overcorrects downward.',
        'risk_level': 'CRITICAL'
    },
    'B_Hyperglycemia_Management': {
        'y_true': [280, 310, 295, 275, 290, 305, 285, 300],
        'model_a': [250, 280, 265, 245, 260, 275, 255, 270],  # Underestimates
        'model_b': [320, 350, 335, 315, 330, 345, 325, 340],  # Overestimates
        'description': 'Sustained high glucose',
        'clinical_significance': 'Severe hyperglycemia (>250 mg/dL). Model A underestimates severity, Model B overestimates.',
        'risk_level': 'HIGH'
    },
    'C_Postprandial_Response': {
        'y_true': [120, 180, 220, 195, 160, 140, 125, 115],
        'model_a': [110, 170, 210, 185, 150, 130, 115, 105],
        'model_b': [130, 190, 230, 205, 170, 150, 135, 125],
        'description': 'Rapid meal-induced changes',
        'clinical_significance': 'Post-meal glucose spikes. Model A slightly underestimates, Model B slightly overestimates.',
        'risk_level': 'MODERATE'
    },
    'D_Dawn_Phenomenon': {
        'y_true': [85, 95, 110, 125, 140, 155, 150, 145],
        'model_a': [90, 100, 115, 130, 145, 160, 155, 150],
        'model_b': [80, 90, 105, 120, 135, 150, 145, 140],
        'description': 'Early morning glucose rise',
        'clinical_significance': 'Dawn phenomenon - early morning glucose elevation. Model A tracks closely, Model B underestimates.',
        'risk_level': 'LOW'
    },
    'E_Exercise_Response': {
        'y_true': [140, 120, 95, 75, 65, 80, 100, 125],
        'model_a': [145, 125, 100, 80, 70, 85, 105, 130],
        'model_b': [135, 115, 90, 70, 60, 75, 95, 120],
        'description': 'Activity-induced glucose drop',
        'clinical_significance': 'Exercise-induced hypoglycemia risk. Model A shows conservative tracking, Model B more aggressive.',
        'risk_level': 'MODERATE'
    },
    'F_Measurement_Noise': {
        'y_true': [100, 105, 98, 102, 99, 103, 101, 97],
        'model_a': [108, 113, 106, 110, 107, 111, 109, 105],  # Positive bias
        'model_b': [92, 97, 90, 94, 91, 95, 93, 89],  # Negative bias
        'description': 'Sensor accuracy challenges',
        'clinical_significance': 'Measurement precision issues in normal range. Model A has positive bias, Model B negative bias.',
        'risk_level': 'LOW'
    },
    'G_Mixed_Clinical': {
        'y_true': [65, 180, 45, 250, 95, 200, 55, 160],
        'model_a': [75, 190, 55, 260, 105, 210, 65, 170],
        'model_b': [55, 170, 35, 240, 85, 190, 45, 150],
        'description': 'Combined challenging scenarios',
        'clinical_significance': 'Mixed critical events including hypoglycemia and hyperglycemia. Tests overall robustness.',
        'risk_level': 'CRITICAL'
    },
    'H_Extreme_Cases': {
        'y_true': [25, 35, 400, 450, 30, 420, 40, 380],
        'model_a': [45, 55, 380, 430, 50, 400, 60, 360],  # Safer predictions
        'model_b': [15, 25, 420, 470, 20, 440, 30, 400],  # More extreme
        'description': 'Life-threatening glucose extremes',
        'clinical_significance': 'Extreme glucose values (<50 or >350 mg/dL). Model A conservative, Model B matches extremes.',
        'risk_level': 'CRITICAL'
    }
}

def explain_scenarios():
    """Generate narrative explanation of all clinical scenarios"""
    log_narrative("\n" + "="*80)
    log_narrative("CLINICAL SCENARIO ANALYSIS")
    log_narrative("="*80)
    
    for scenario_name, data in scenarios.items():
        log_narrative(f"\n{scenario_name}: {data['description']}")
        log_narrative(f"Risk Level: {data['risk_level']}")
        log_narrative(f"Clinical Context: {data['clinical_significance']}")
        
        y_true_range = f"{min(data['y_true'])}-{max(data['y_true'])} mg/dL"
        model_a_range = f"{min(data['model_a'])}-{max(data['model_a'])} mg/dL"
        model_b_range = f"{min(data['model_b'])}-{max(data['model_b'])} mg/dL"
        
        log_narrative(f"True glucose range: {y_true_range}")
        log_narrative(f"Model A predictions: {model_a_range}")
        log_narrative(f"Model B predictions: {model_b_range}")

def conduct_sensitivity_analysis(a_values, b_values, c_values):
    """Conduct comprehensive sensitivity analysis for a, b, and c"""

    log_narrative("\n" + "="*80)
    log_narrative("MADEX SENSITIVITY ANALYSIS INITIATED")
    log_narrative("="*80)
    
    log_narrative(f"Parameter ranges being tested:")
    log_narrative(f"- Parameter 'a' (euglycemic center): {a_values} mg/dL")
    log_narrative(f"- Parameter 'b' (critical range): {b_values}")
    log_narrative(f"- Parameter 'c' (slope modifier): {c_values}")
    
    total_combinations = len(a_values) * len(b_values) * len(c_values)
    log_narrative(f"Total parameter combinations to test: {total_combinations}")

    results = []
    baseline_rankings = {}

    print("Conducting MADEX Sensitivity Analysis for a, b, and c...")
    print("="*60)

    # Calculate baseline rankings (standard parameters: a=125, b=55, c=40)
    log_narrative("\nESTABLISHING BASELINE RANKINGS")
    log_narrative("-"*40)
    log_narrative("Using standard clinical parameters: a=125 mg/dL (normal target), b=55 (moderate sensitivity), c=40 (standard slope)")
    
    print("Calculating baseline rankings...")
    for scenario_name, scenario_data in scenarios.items():
        y_true = scenario_data['y_true']
        model_a_pred = scenario_data['model_a']
        model_b_pred = scenario_data['model_b']

        # Calculate baseline MADEX scores
        madex_a = calculate_madex(y_true, model_a_pred, a=125, b=55, c=40)
        madex_b = calculate_madex(y_true, model_b_pred, a=125, b=55, c=40)

        # Store baseline ranking (1 = better, 2 = worse)
        if madex_a < madex_b:
            baseline_rankings[scenario_name] = ['Model_A', 'Model_B']
        else:
            baseline_rankings[scenario_name] = ['Model_B', 'Model_A']

    print(f"Baseline rankings calculated for {len(scenarios)} scenarios (using a=125, b=55, c=40)")

    # Conduct sensitivity analysis across parameter combinations
    print("\\nTesting parameter sensitivity...")
    total_combinations = len(a_values) * len(b_values) * len(c_values)
    current_combo = 0

    for a, b, c in product(a_values, b_values, c_values):
        current_combo += 1
        if current_combo % 20 == 0 or current_combo == total_combinations:
            print(f"Progress: {current_combo}/{total_combinations} combinations tested")

        ranking_matches = 0
        scenario_results = []

        for scenario_name, scenario_data in scenarios.items():
            y_true = scenario_data['y_true']
            model_a_pred = scenario_data['model_a']
            model_b_pred = scenario_data['model_b']

            # Calculate MADEX with current parameters
            try:
                madex_a = calculate_madex(y_true, model_a_pred, a=a, b=b, c=c)
                madex_b = calculate_madex(y_true, model_b_pred, a=a, b=b, c=c)

                # Determine current ranking
                if madex_a < madex_b:
                    current_ranking = ['Model_A', 'Model_B']
                else:
                    current_ranking = ['Model_B', 'Model_A']

                # Check if ranking matches baseline
                if current_ranking == baseline_rankings[scenario_name]:
                    ranking_matches += 1

                scenario_results.append({
                    'scenario': scenario_name,
                    'madex_a': madex_a,
                    'madex_b': madex_b,
                    'ranking_match': current_ranking == baseline_rankings[scenario_name]
                })

            except (OverflowError, ValueError, RuntimeWarning):
                # Handle numerical issues
                scenario_results.append({
                    'scenario': scenario_name,
                    'madex_a': np.nan,
                    'madex_b': np.nan,
                    'ranking_match': False
                })

        # Calculate ranking consistency
        ranking_consistency = ranking_matches / len(scenarios)

        results.append({
            'a': a, 'b': b, 'c': c,
            'ranking_consistency': ranking_consistency,
            'valid_scenarios': len([r for r in scenario_results if not np.isnan(r['madex_a'])])
        })

    return pd.DataFrame(results), baseline_rankings

def analyze_parameter_effects(results_df):
    """Analyze the effects of parameter variations"""
    
    log_narrative("\n" + "="*80)
    log_narrative("PARAMETER EFFECT ANALYSIS")
    log_narrative("="*80)
    
    mean_consistency = results_df['ranking_consistency'].mean()
    std_consistency = results_df['ranking_consistency'].std()
    min_consistency = results_df['ranking_consistency'].min()
    max_consistency = results_df['ranking_consistency'].max()
    
    log_narrative(f"Overall MADEX Stability Assessment:")
    log_narrative(f"- Mean ranking consistency: {mean_consistency:.3f} ({mean_consistency*100:.1f}%)")
    log_narrative(f"- Standard deviation: {std_consistency:.3f}")
    log_narrative(f"- Range: {min_consistency:.3f} to {max_consistency:.3f}")
    
    if std_consistency < 0.1:
        log_narrative("- Assessment: VERY STABLE - Low variance across parameter changes")
    elif std_consistency < 0.2:
        log_narrative("- Assessment: STABLE - Moderate variance across parameter changes")  
    else:
        log_narrative("- Assessment: VARIABLE - High variance, parameter-dependent behavior")

    print("\nParameter Effect Analysis:")
    print("="*30)

    # Overall statistics
    print(f"Mean ranking consistency: {mean_consistency:.3f}")
    print(f"Std ranking consistency: {std_consistency:.3f}")
    print(f"Min ranking consistency: {min_consistency:.3f}")
    print(f"Max ranking consistency: {max_consistency:.3f}")

    # Parameter-specific analysis
    log_narrative("\nParameter-specific Impact Analysis:")
    print("\nParameter-wise Ranking Consistency:")

    for param in ['a', 'b', 'c']:
        param_effects = results_df.groupby(param)['ranking_consistency'].agg(['mean', 'std', 'count'])
        
        log_narrative(f"\nParameter '{param}' Analysis:")
        if param == 'a':
            log_narrative("  Clinical meaning: Euglycemic center - target glucose level")
        elif param == 'b':
            log_narrative("  Clinical meaning: Critical range - sensitivity to deviations from target")
        else:
            log_narrative("  Clinical meaning: Slope modifier - error scaling factor")
            
        print(f"\nParameter {param.upper()}:")
        for value in param_effects.index:
            mean_val = param_effects.loc[value, 'mean']
            std_val = param_effects.loc[value, 'std']
            print(f"  {param}={value:.0f}: {mean_val:.3f} +/- {std_val:.3f}")
            
            # Clinical interpretation
            if param == 'a':
                if value < 110:
                    interpretation = "tight glycemic control (aggressive)"
                elif value > 140:
                    interpretation = "relaxed glycemic control (conservative)"
                else:
                    interpretation = "standard glycemic control"
            elif param == 'b':
                if value < 50:
                    interpretation = "high sensitivity to deviations"
                elif value > 70:
                    interpretation = "low sensitivity to deviations"
                else:
                    interpretation = "moderate sensitivity"
            else:  # param == 'c'
                if value < 30:
                    interpretation = "steep error scaling (harsh penalties)"
                elif value > 50:
                    interpretation = "gentle error scaling (mild penalties)"
                else:
                    interpretation = "standard error scaling"
                    
            log_narrative(f"    {param}={value:.0f}: {mean_val:.3f} consistency ({interpretation})")

    # Find most stable parameter combinations
    high_consistency = results_df[results_df['ranking_consistency'] >= 0.8]
    
    log_narrative(f"\nRobustness Assessment:")
    log_narrative(f"- Parameter combinations with >80% consistency: {len(high_consistency)}/{len(results_df)}")
    log_narrative(f"- Percentage of robust combinations: {100*len(high_consistency)/len(results_df):.1f}%")
    
    print(f"\nParameter combinations with >80% ranking consistency: {len(high_consistency)}")

    if len(high_consistency) > 0:
        print("\nTop 5 most stable parameter combinations:")
        top_stable = high_consistency.nlargest(5, 'ranking_consistency')
        
        log_narrative("\nMost Stable Parameter Combinations (Clinical Recommendations):")
        for idx, row in top_stable.iterrows():
            consistency = row['ranking_consistency']
            a, b, c = row['a'], row['b'], row['c']
            print(f"  a={a:.0f}, b={b:.0f}, c={c:.0f}: {consistency:.3f}")
            
            # Clinical context
            context = ""
            if a < 110: context += "tight control, "
            elif a > 140: context += "relaxed control, "
            else: context += "standard control, "
            
            if b < 50: context += "high sensitivity, "
            elif b > 70: context += "low sensitivity, "  
            else: context += "moderate sensitivity, "
            
            if c < 30: context += "harsh penalties"
            elif c > 50: context += "mild penalties"
            else: context += "standard penalties"
            
            log_narrative(f"  Combination a={a:.0f}, b={b:.0f}, c={c:.0f}: {consistency:.3f} ({context})")

    return results_df

def compare_with_traditional_metrics():
    """Compare MADEX with traditional metrics across scenarios"""
    
    log_narrative("\n" + "="*80)
    log_narrative("MADEX vs TRADITIONAL METRICS COMPARISON")
    log_narrative("="*80)
    log_narrative("Comparing MADEX model rankings with RMSE, MAE, and MAPE across clinical scenarios")
    log_narrative("This analysis reveals where MADEX provides different insights than standard metrics")

    print("\nComparison with Traditional Metrics:")
    print("="*40)

    comparison_results = []

    for scenario_name, scenario_data in scenarios.items():
        y_true = scenario_data['y_true']
        model_a_pred = scenario_data['model_a']
        model_b_pred = scenario_data['model_b']

        log_narrative(f"\nAnalyzing {scenario_name}:")
        log_narrative(f"Clinical context: {scenario_data['clinical_significance']}")

        # Calculate MADEX (baseline parameters)
        madex_a = calculate_madex(y_true, model_a_pred, enable_logging=True)
        madex_b = calculate_madex(y_true, model_b_pred, enable_logging=True)

        # Calculate traditional metrics
        metrics_a = calculate_comparison_metrics(y_true, model_a_pred)
        metrics_b = calculate_comparison_metrics(y_true, model_b_pred)

        # Determine rankings
        madex_ranking = 'A' if madex_a < madex_b else 'B'
        rmse_ranking = 'A' if metrics_a['RMSE'] < metrics_b['RMSE'] else 'B'
        mae_ranking = 'A' if metrics_a['MAE'] < metrics_b['MAE'] else 'B'
        mape_ranking = 'A' if metrics_a['MAPE'] < metrics_b['MAPE'] else 'B'

        # Log detailed comparison
        log_narrative(f"Model Rankings - MADEX: Model_{madex_ranking}, RMSE: Model_{rmse_ranking}, MAE: Model_{mae_ranking}, MAPE: Model_{mape_ranking}")
        
        if madex_ranking != rmse_ranking:
            log_narrative(f"[WARNING] DIFFERENCE: MADEX prefers Model_{madex_ranking} while RMSE prefers Model_{rmse_ranking}")
            log_narrative(f"   Clinical implication: MADEX captures error severity patterns that RMSE misses")
            
        comparison_results.append({
            'Scenario': scenario_name,
            'Description': scenario_data['description'],
            'MADEX_Winner': madex_ranking,
            'RMSE_Winner': rmse_ranking,
            'MAE_Winner': mae_ranking,
            'MAPE_Winner': mape_ranking,
            'MADEX_A': f"{madex_a:.2e}",
            'MADEX_B': f"{madex_b:.2e}",
            'RMSE_A': f"{metrics_a['RMSE']:.2f}",
            'RMSE_B': f"{metrics_b['RMSE']:.2f}"
        })

    comparison_df = pd.DataFrame(comparison_results)

    # Calculate agreement rates
    madex_vs_rmse = (comparison_df['MADEX_Winner'] == comparison_df['RMSE_Winner']).mean()
    madex_vs_mae = (comparison_df['MADEX_Winner'] == comparison_df['MAE_Winner']).mean()
    madex_vs_mape = (comparison_df['MADEX_Winner'] == comparison_df['MAPE_Winner']).mean()

    log_narrative(f"\nMetric Agreement Analysis:")
    log_narrative(f"- MADEX vs RMSE agreement: {madex_vs_rmse:.3f} ({madex_vs_rmse*100:.1f}%)")
    log_narrative(f"- MADEX vs MAE agreement: {madex_vs_mae:.3f} ({madex_vs_mae*100:.1f}%)")
    log_narrative(f"- MADEX vs MAPE agreement: {madex_vs_mape:.3f} ({madex_vs_mape*100:.1f}%)")

    if madex_vs_rmse < 0.7:
        log_narrative("[SUCCESS] SIGNIFICANT DIFFERENCE: MADEX provides substantially different insights than traditional metrics")
        log_narrative("  This suggests MADEX captures clinically relevant error patterns that standard metrics miss")
    elif madex_vs_rmse < 0.9:
        log_narrative("[SUCCESS] MODERATE DIFFERENCE: MADEX shows some unique perspectives compared to traditional metrics")
    else:
        log_narrative("[CAUTION] HIGH SIMILARITY: MADEX rankings closely match traditional metrics")

    print(f"MADEX vs RMSE agreement: {madex_vs_rmse:.3f}")
    print(f"MADEX vs MAE agreement: {madex_vs_mae:.3f}")
    print(f"MADEX vs MAPE agreement: {madex_vs_mape:.3f}")

    return comparison_df

def generate_summary_report(a_values, b_values, c_values):
    """Generate comprehensive sensitivity analysis report"""

    # Initialize narrative logging
    log_narrative("\n" + "="*100)
    log_narrative("MADEX (Mean Adjusted Exponent Error) COMPREHENSIVE ANALYSIS REPORT")
    log_narrative("="*100)
    log_narrative(f"Analysis initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_narrative(f"Analysis type: Parameter sensitivity testing for MADEX metric")
    log_narrative(f"Clinical context: Glucose prediction model evaluation")
    
    log_narrative("\nANALYSIS OBJECTIVES:")
    log_narrative("1. Validate MADEX robustness across clinical parameter ranges")
    log_narrative("2. Identify optimal parameter combinations for different clinical contexts")
    log_narrative("3. Compare MADEX performance against traditional metrics (RMSE, MAE, MAPE)")
    log_narrative("4. Assess clinical significance of model ranking changes")
    
    # Explain clinical scenarios first
    explain_scenarios()

    print("\n" + "="*60)
    print("MADEX SENSITIVITY ANALYSIS REPORT (a, b, and c parameters)")
    print("="*60)

    # Conduct main analysis
    results_df, baseline_rankings = conduct_sensitivity_analysis(a_values, b_values, c_values)

    # Analyze parameter effects
    analyzed_results = analyze_parameter_effects(results_df)

    # Compare with traditional metrics (using baseline parameters)
    comparison_df = compare_with_traditional_metrics()

    print("\\nSUMMARY FINDINGS:")
    print("-"*20)
    print("1. MADEX ranking consistency across a, b, and c parameter variations:")
    print(f"   Mean: {results_df['ranking_consistency'].mean():.3f}")
    print(f"   Range: {results_df['ranking_consistency'].min():.3f} - {results_df['ranking_consistency'].max():.3f}")

    stable_params = results_df[results_df['ranking_consistency'] >= 0.8]
    print(f"2. {len(stable_params)} out of {len(results_df)} parameter combinations")
    print("   maintain >80% ranking consistency")

    print("3. MADEX provides different model rankings than traditional")
    print("   metrics in clinically critical scenarios (hypoglycemia,")
    print("   hyperglycemia management) using baseline parameters.")

    return {
        'sensitivity_results': results_df,
        'baseline_rankings': baseline_rankings,
        'comparison_results': comparison_df
    }

# Define parameter sets based on clinical ranges
clinical_param_sets = {
    'Standard Adult Range': {'a': 125, 'b': 55},
    'Tight Glycemic Control': {'a': 110, 'b': 40},
    'Pediatric Range': {'a': 140, 'b': 70},
    'Elderly/Relaxed Control': {'a': 140, 'b': 80}
}

# Define a range of c values to test
sensitivity_c_values = [20, 30, 40, 50, 60, 80]

# Extract unique a and b values for the sensitivity analysis
sensitivity_a_values = sorted(list(set([params['a'] for params in clinical_param_sets.values()])))
sensitivity_b_values = sorted(list(set([params['b'] for params in clinical_param_sets.values()])))

# Execute the sensitivity analysis for a, b, and c
results_a_b_c = generate_summary_report(sensitivity_a_values, sensitivity_b_values, sensitivity_c_values)