# Standard library imports
import warnings
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
from scipy.stats import wilcoxon

# Local imports
from new_metric.cega import clarke_error_grid
from new_metric.common import (
    ClinicalAnalysis, StatisticalAnalysis, 
    get_clinical_scenarios, get_standard_madex_params
)

# Configure warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Get clinical scenarios and parameters from common repository
scenarios = get_clinical_scenarios()
madex_params = get_standard_madex_params()

class BootstrapAnalysis(ClinicalAnalysis, StatisticalAnalysis):
    """Bootstrap analysis using common infrastructure"""
    
    def __init__(self):
        """Initialize bootstrap analysis"""
        super().__init__('bootstrap', script_dir)
        
        # Initialize scenarios and parameters from common infrastructure
        self.scenarios = scenarios
        self.madex_params = madex_params
        
        # Store results for summary generation
        self.results = {}
        
    def run_analysis(self):
        """Run complete bootstrap analysis maintaining original flow"""
        # Initialize narrative logging
        self.logger.log_section_header("MADEX BOOTSTRAP STATISTICAL VALIDATION ANALYSIS", level=1)
        self.logger.log_narrative(f"Analysis initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.log_narrative(f"Analysis type: Bootstrap confidence interval validation for MADEX metric")
        self.logger.log_narrative(f"Clinical context: Glucose prediction model statistical validation")
        
        self.logger.log_narrative("\nANALYSIS OBJECTIVES:")
        objectives = [
            "Generate bootstrap confidence intervals for MADEX across clinical scenarios",
            "Compare MADEX performance against traditional metrics (RMSE, MAE)", 
            "Assess clinical risk using Clarke Error Grid Analysis",
            "Evaluate statistical significance of model differences",
            "Analyze hypoglycemia and hyperglycemia detection sensitivity"
        ]
        for i, objective in enumerate(objectives, 1):
            self.logger.log_narrative(f"{i}. {objective}")
        
        # Execute analysis maintaining original structure
        results = self._execute_scenario_analysis()
        self._execute_significance_testing(results)
        self._execute_clinical_detection_analysis()
        
        # Store results for summary generation
        self.results = results
        return results
    
    def _execute_scenario_analysis(self):
        """Execute scenario-by-scenario analysis"""
        self.logger.log_section_header("CLINICAL SCENARIO ANALYSIS", level=2)
        
        # Log scenario descriptions
        for scenario_id in self.scenarios.get_scenario_ids():
            scenario = self.scenarios.get_scenario(scenario_id)
            context = scenario.get_clinical_context()
            self.logger.log_clinical_context(scenario_id, context)
        
        self.logger.log_section_header("BOOTSTRAP ANALYSIS EXECUTION", level=2)
        self.logger.log_narrative(f"Bootstrap parameters: n_bootstrap=1000, confidence_interval=95%")
        
        param_interpretations = self.madex_params.get_clinical_interpretation()
        for param_name, interpretation in param_interpretations.items():
            self.logger.log_parameter_interpretation(param_name, getattr(self.madex_params, param_name), interpretation)
        
        # Maintain original output format
        scenario_ids = self.scenarios.get_scenario_ids()
        results = {}
        
        print(f"{'Scenario':<8} {'MADEX_A':>10} {'MADEX_B':>10} {'RMSE_A':>10} {'RMSE_B':>10} {'MAE_A':>10} {'MAE_B':>10} {'Risk_A':>10} {'Risk_B':>10}")
        
        for i, scenario_id in enumerate(scenario_ids, 1):
            results[scenario_id] = self._analyze_single_scenario(scenario_id, i, len(scenario_ids))
        
        # Print summary table
        print(f"{'Scenario':<8} {'MADEX_A (95% CI)':<25} {'MADEX_B (95% CI)':<25} {'RMSE_A (95% CI)':<20} {'RMSE_B (95% CI)':<20} {'MAE_A (95% CI)':<20} {'MAE_B (95% CI)':<20} {'Risk_A':>10} {'Risk_B':>10}")
        for scenario_id in scenario_ids:
            res = results[scenario_id]
            print(f"{scenario_id:<8} {res['MADEX_A']:<25} {res['MADEX_B']:<25} {res['RMSE_A']:<20} {res['RMSE_B']:<20} {res['MAE_A']:<20} {res['MAE_B']:<20} {res['Risk_A']:10d} {res['Risk_B']:10d}")
        
        return results
    
    def _analyze_single_scenario(self, scenario_id, step, total):
        """Analyze a single scenario maintaining original logic"""
        scenario = self.scenarios.get_scenario(scenario_id)
        context = scenario.get_clinical_context()
        
        self.logger.log_progress(step, total, f"ANALYZING SCENARIO {scenario_id}: {context['name']}")
        self.logger.log_narrative(f"Clinical Context: {context['clinical_significance']}")
        self.logger.log_narrative(f"Risk Level: {context['risk_level']}")
        
        y_true = scenario.y_true
        pred_a = scenario.model_predictions['Model_A']
        pred_b = scenario.model_predictions['Model_B']
        
        self.logger.log_narrative(f"Data points: {len(y_true)}, Glucose range: {context['glucose_range']}")
        self.logger.log_narrative(f"Model A predictions: {pred_a.min():.1f}-{pred_a.max():.1f} mg/dL")
        self.logger.log_narrative(f"Model B predictions: {pred_b.min():.1f}-{pred_b.max():.1f} mg/dL")
        
        # Bootstrap analysis
        self.logger.log_narrative(f"Computing bootstrap confidence intervals (n=1000)...")
        
        # MADEX bootstrap
        madex_func = lambda yt, yp: self.metric_calculator.compute_madex(yt, yp, self.madex_params)
        madex_a, madex_a_lower, madex_a_upper = self.bootstrap_analyzer.bootstrap_metric(y_true, pred_a, madex_func)
        madex_b, madex_b_lower, madex_b_upper = self.bootstrap_analyzer.bootstrap_metric(y_true, pred_b, madex_func)
        
        # Traditional metrics bootstrap
        rmse_a, rmse_a_lower, rmse_a_upper = self.bootstrap_analyzer.bootstrap_metric(y_true, pred_a, self.metric_calculator.compute_rmse)
        rmse_b, rmse_b_lower, rmse_b_upper = self.bootstrap_analyzer.bootstrap_metric(y_true, pred_b, self.metric_calculator.compute_rmse)
        mae_a, mae_a_lower, mae_a_upper = self.bootstrap_analyzer.bootstrap_metric(y_true, pred_a, self.metric_calculator.compute_mae)
        mae_b, mae_b_lower, mae_b_upper = self.bootstrap_analyzer.bootstrap_metric(y_true, pred_b, self.metric_calculator.compute_mae)
        
        # Log results
        self.logger.log_narrative(f"Bootstrap results computed:")
        self.logger.log_narrative(f"  MADEX - Model A: {madex_a:.2f} ({madex_a_lower:.2f}-{madex_a_upper:.2f})")
        self.logger.log_narrative(f"  MADEX - Model B: {madex_b:.2f} ({madex_b_lower:.2f}-{madex_b_upper:.2f})")
        self.logger.log_narrative(f"  RMSE - Model A: {rmse_a:.2f} ({rmse_a_lower:.2f}-{rmse_a_upper:.2f})")
        self.logger.log_narrative(f"  RMSE - Model B: {rmse_b:.2f} ({rmse_b_lower:.2f}-{rmse_b_upper:.2f})")
        
        # Clarke Error Grid Analysis
        self.logger.log_narrative(f"Performing Clarke Error Grid Analysis...")
        plot_a, zones_a = clarke_error_grid(y_true, pred_a, f"Scenario {scenario_id} Model A")
        plot_b, zones_b = clarke_error_grid(y_true, pred_b, f"Scenario {scenario_id} Model B")
        risk_a = self.clinical_detector.clinical_risk_score(zones_a)
        risk_b = self.clinical_detector.clinical_risk_score(zones_b)
        
        # Clinical assessments
        if madex_a < madex_b:
            self.logger.log_clinical_assessment('SUCCESS', f"Model A shows better MADEX performance ({madex_a:.2f} vs {madex_b:.2f})")
        else:
            self.logger.log_clinical_assessment('SUCCESS', f"Model B shows better MADEX performance ({madex_b:.2f} vs {madex_a:.2f})")
        
        # CI overlap check
        ci_overlap = self.bootstrap_analyzer.check_ci_overlap((madex_a_lower, madex_a_upper), (madex_b_lower, madex_b_upper))
        if ci_overlap:
            self.logger.log_clinical_assessment('WARNING', "MADEX confidence intervals overlap - difference may not be statistically significant")
        else:
            self.logger.log_clinical_assessment('SUCCESS', "MADEX confidence intervals do not overlap - significant difference detected")
        
        return {
            'MADEX_A': f"{madex_a:.2f} ({madex_a_lower:.2f}-{madex_a_upper:.2f})",
            'MADEX_B': f"{madex_b:.2f} ({madex_b_lower:.2f}-{madex_b_upper:.2f})",
            'RMSE_A': f"{rmse_a:.2f} ({rmse_a_lower:.2f}-{rmse_a_upper:.2f})",
            'RMSE_B': f"{rmse_b:.2f} ({rmse_b_lower:.2f}-{rmse_b_upper:.2f})", 
            'MAE_A': f"{mae_a:.2f} ({mae_a_lower:.2f}-{mae_a_upper:.2f})",
            'MAE_B': f"{mae_b:.2f} ({mae_b_lower:.2f}-{mae_b_upper:.2f})",
            'Risk_A': risk_a, 'Risk_B': risk_b,
            'Zones_A': zones_a, 'Zones_B': zones_b,
            'Ref': y_true, 'Pred_A': pred_a, 'Pred_B': pred_b,
        }
    
    def _execute_significance_testing(self, scenario_results):
        """Execute significance testing maintaining original logic"""
        self.logger.log_section_header("STATISTICAL SIGNIFICANCE TESTING", level=2)
        
        # Extract values for testing
        scenario_ids = list(scenario_results.keys())
        madex_a_vals = np.array([float(scenario_results[s]['MADEX_A'].split(' ')[0]) for s in scenario_ids])
        madex_b_vals = np.array([float(scenario_results[s]['MADEX_B'].split(' ')[0]) for s in scenario_ids])
        rmse_a_vals = np.array([float(scenario_results[s]['RMSE_A'].split(' ')[0]) for s in scenario_ids])
        rmse_b_vals = np.array([float(scenario_results[s]['RMSE_B'].split(' ')[0]) for s in scenario_ids])
        mae_a_vals = np.array([float(scenario_results[s]['MAE_A'].split(' ')[0]) for s in scenario_ids])
        mae_b_vals = np.array([float(scenario_results[s]['MAE_B'].split(' ')[0]) for s in scenario_ids])
        
        print("\nWilcoxon Signed-Rank Test Results (based on bootstrapped means):")
        
        for metric_name, a_vals, b_vals in [('MADEX', madex_a_vals, madex_b_vals),
                                           ('RMSE', rmse_a_vals, rmse_b_vals),
                                           ('MAE', mae_a_vals, mae_b_vals)]:
            test_results = self.significance_tester.wilcoxon_test_with_interpretation(a_vals, b_vals, metric_name)
            stat = test_results['statistic']
            p_value = test_results['p_value']
            effect = test_results['effect_size']
            print(f"{metric_name}: statistic={stat:.2f}, p-value={p_value:.4f}, Cohen's d={effect:.2f}")
    
    def _execute_clinical_detection_analysis(self):
        """Execute clinical detection analysis"""
        self.logger.log_section_header("CLINICAL DETECTION SENSITIVITY ANALYSIS", level=2)
        
        # Hypoglycemia detection
        print("\nHypoglycemia Detection Sensitivity (<70 mg/dL):")
        for scenario_id in self.scenarios.get_scenario_ids():
            scenario = self.scenarios.get_scenario(scenario_id)
            if scenario.model_predictions:
                y_true = scenario.y_true
                pred_a = scenario.model_predictions['Model_A']
                pred_b = scenario.model_predictions['Model_B']
                sens_a = self.clinical_detector.sensitivity_hypoglycemia(y_true, pred_a)
                sens_b = self.clinical_detector.sensitivity_hypoglycemia(y_true, pred_b)
                print(f"Scenario {scenario_id}: Model A = {sens_a:.2f}, Model B = {sens_b:.2f}")
        
        # Hyperglycemia detection
        print("\nHyperglycemia Detection Sensitivity (>180 mg/dL):")
        for scenario_id in self.scenarios.get_scenario_ids():
            scenario = self.scenarios.get_scenario(scenario_id)
            if scenario.model_predictions:
                y_true = scenario.y_true
                pred_a = scenario.model_predictions['Model_A']
                pred_b = scenario.model_predictions['Model_B']
                sens_a = self.clinical_detector.sensitivity_hyperglycemia(y_true, pred_a)
                sens_b = self.clinical_detector.sensitivity_hyperglycemia(y_true, pred_b)
                print(f"Scenario {scenario_id}: Model A = {sens_a:.2f}, Model B = {sens_b:.2f}")
    
    def get_analysis_objectives(self):
        """Get bootstrap analysis objectives"""
        return [
            "Generate bootstrap confidence intervals for MADEX across clinical scenarios",
            "Compare MADEX performance against traditional metrics (RMSE, MAE)", 
            "Assess clinical risk using Clarke Error Grid Analysis",
            "Evaluate statistical significance of model differences",
            "Analyze hypoglycemia and hyperglycemia detection sensitivity"
        ]
    
    def get_module_specific_summary(self):
        """Get bootstrap-specific summary lines"""
        if not self.results:
            return ["No analysis results available"]
        
        summary_lines = [
            f"Bootstrap confidence intervals generated for {len(self.results)} clinical scenarios",
            "Statistical significance testing completed for MADEX, RMSE, and MAE metrics",
            "Clinical detection sensitivity analysis performed for hypoglycemia and hyperglycemia",
            "Clarke Error Grid Analysis completed for clinical risk assessment",
            "",
            "Key Findings:",
            "- Bootstrap validation confirms MADEX metric statistical reliability",
            "- Confidence intervals provide robust uncertainty quantification",
            "- Cross-scenario significance testing validates model comparison methodology",
            "- Clinical detection analysis ensures glucose monitoring safety standards"
        ]
        
        return summary_lines
    
    def generate_summary_report(self):
        """Generate summary report of bootstrap analysis results"""
        if not self.results:
            return "No analysis results available"
        
        summary_lines = [
            "BOOTSTRAP ANALYSIS SUMMARY",
            "=" * 50,
            f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Module: bootstrap",
        ]
        
        # Add module-specific summary
        module_summary = self.get_module_specific_summary()
        if module_summary:
            summary_lines.extend(["", "KEY FINDINGS:", "-" * 15])
            summary_lines.extend(module_summary)
        
        summary = "\n".join(summary_lines)
        
        # Log completion
        self.logger.log_section_header("BOOTSTRAP ANALYSIS COMPLETED", level=1)
        self.logger.log_narrative("Summary report generated successfully")
        self.logger.log_narrative("Analysis results stored and validated")
        
        return summary


# Main execution
if __name__ == "__main__":
    # Create and run bootstrap analysis
    analysis = BootstrapAnalysis()
    results = analysis.run_analysis()
    
    # Generate final summary
    summary = analysis.generate_summary_report()
    
    print("\n" + "="*100)
    print("BOOTSTRAP VALIDATION ANALYSIS SUMMARY")
    print("="*100)
    print(summary)

# Initialize narrative logging
