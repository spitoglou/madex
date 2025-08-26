"""
Analysis Framework

Base classes and interfaces for clinical analysis modules providing common
workflows, result aggregation patterns, and report generation capabilities.

Features:
- Base analysis framework with common patterns
- Clinical analysis specialization for glucose monitoring
- Statistical analysis workflows
- Result aggregation and reporting utilities
- Extensible design for different analysis types
"""

import abc
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .clinical_data import ClinicalScenario, MADEXParameters, ScenarioRepository
from .clinical_logging import ClinicalLogger
from .statistical_utils import (
    BootstrapAnalyzer, ClinicalDetection, MetricCalculator, SignificanceTester
)


class AnalysisFramework(abc.ABC):
    """
    Base analysis framework providing common structure and utilities
    
    This abstract base class defines the common interface and shared functionality
    for all analysis modules in the new_metric package.
    """
    
    def __init__(self, module_name: str, script_dir: Optional[Path] = None):
        """
        Initialize analysis framework
        
        Args:
            module_name: Name of the analysis module (e.g., 'bootstrap', 'sensitivity')
            script_dir: Directory for logs and outputs (defaults to caller's directory)
        """
        self.module_name = module_name
        self.script_dir = script_dir or Path(__file__).parent
        
        # Initialize components
        self.logger = ClinicalLogger(module_name, self.script_dir)
        self.metric_calculator = MetricCalculator(self.logger)
        self.bootstrap_analyzer = BootstrapAnalyzer(self.logger)
        self.significance_tester = SignificanceTester(self.logger)
        self.clinical_detector = ClinicalDetection(self.logger)
        
        # Analysis state
        self.results: Dict[str, Any] = {}
        self.scenarios: Optional[ScenarioRepository] = None
        self.madex_params: Optional[MADEXParameters] = None
        
        # Configure warnings
        warnings.filterwarnings('ignore')
    
    def setup_analysis(self, analysis_type: str, clinical_context: str, 
                      scenarios: Optional[ScenarioRepository] = None,
                      madex_params: Optional[MADEXParameters] = None):
        """
        Setup analysis with context and parameters
        
        Args:
            analysis_type: Type of analysis being performed
            clinical_context: Clinical context description
            scenarios: Clinical scenario repository
            madex_params: MADEX parameters to use
        """
        # Initialize scenarios and parameters
        if scenarios is None:
            from .clinical_data import get_clinical_scenarios
            self.scenarios = get_clinical_scenarios()
        else:
            self.scenarios = scenarios
            
        if madex_params is None:
            from .clinical_data import get_standard_madex_params
            self.madex_params = get_standard_madex_params()
        else:
            self.madex_params = madex_params
        
        # Log analysis start
        self.logger.log_analysis_start(analysis_type, clinical_context)
        
        # Log analysis objectives
        objectives = self.get_analysis_objectives()
        if objectives:
            self.logger.log_narrative("\nANALYSIS OBJECTIVES:")
            for i, objective in enumerate(objectives, 1):
                self.logger.log_narrative(f"{i}. {objective}")
    
    @abc.abstractmethod
    def get_analysis_objectives(self) -> List[str]:
        """
        Get list of analysis objectives for logging
        
        Returns:
            List of objective descriptions
        """
        pass
    
    @abc.abstractmethod
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the main analysis
        
        Returns:
            Dictionary with analysis results
        """
        pass
    
    def generate_summary_report(self) -> str:
        """
        Generate summary report of analysis results
        
        Returns:
            Summary report string
        """
        if not self.results:
            return "No analysis results available"
        
        summary_lines = [
            f"{self.module_name.upper()} ANALYSIS SUMMARY",
            "=" * 50,
            f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Module: {self.module_name}",
        ]
        
        # Add module-specific summary
        module_summary = self.get_module_specific_summary()
        if module_summary:
            summary_lines.extend(["", "KEY FINDINGS:", "-" * 15])
            summary_lines.extend(module_summary)
        
        summary = "\n".join(summary_lines)
        self.logger.log_analysis_complete(summary)
        
        return summary
    
    def get_module_specific_summary(self) -> List[str]:
        """
        Get module-specific summary lines
        
        Returns:
            List of summary line strings
        """
        return []
    
    def log_scenario_context(self, scenario_id: str):
        """
        Log clinical scenario context
        
        Args:
            scenario_id: ID of the scenario to log
        """
        if self.scenarios and scenario_id in self.scenarios.scenarios:
            scenario = self.scenarios.get_scenario(scenario_id)
            context = scenario.get_clinical_context()
            self.logger.log_clinical_context(scenario_id, context)
    
    def get_scenario_list(self) -> List[str]:
        """Get list of available scenario IDs"""
        if self.scenarios:
            return self.scenarios.get_scenario_ids()
        return []


class ClinicalAnalysis(AnalysisFramework):
    """
    Clinical analysis specialization for glucose monitoring scenarios
    
    Provides common functionality for analyzing clinical glucose prediction
    scenarios including MADEX evaluation, clinical detection, and risk assessment.
    """
    
    def __init__(self, module_name: str, script_dir: Optional[Path] = None):
        """Initialize clinical analysis framework"""
        super().__init__(module_name, script_dir)
        
    def analyze_scenario(self, scenario_id: str, 
                        enable_detailed_logging: bool = True) -> Dict[str, Any]:
        """
        Analyze a single clinical scenario
        
        Args:
            scenario_id: ID of scenario to analyze
            enable_detailed_logging: Whether to enable detailed logging
            
        Returns:
            Dictionary with scenario analysis results
        """
        if not self.scenarios:
            raise ValueError("Scenarios not initialized. Call setup_analysis() first.")
        
        scenario = self.scenarios.get_scenario(scenario_id)
        
        if enable_detailed_logging:
            self.log_scenario_context(scenario_id)
        
        # Analyze each model if predictions available
        results = {'scenario_id': scenario_id}
        
        if scenario.model_predictions:
            for model_name, predictions in scenario.model_predictions.items():
                model_results = self.analyze_model_predictions(
                    scenario.y_true, predictions, model_name, enable_detailed_logging
                )
                results[model_name] = model_results
        
        return results
    
    def analyze_model_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str, enable_logging: bool = True) -> Dict[str, Any]:
        """
        Analyze model predictions for a scenario
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            model_name: Name of the model
            enable_logging: Whether to enable logging
            
        Returns:
            Dictionary with model analysis results
        """
        results = {'model_name': model_name}
        
        # Calculate MADEX
        madex_score = self.metric_calculator.compute_madex(
            y_true, y_pred, self.madex_params, enable_logging
        )
        results['madex'] = madex_score
        
        # Calculate traditional metrics
        traditional_metrics = self.metric_calculator.compute_comparison_metrics(y_true, y_pred)
        results.update(traditional_metrics)
        
        # Clinical detection analysis
        hypo_sensitivity = self.clinical_detector.sensitivity_hypoglycemia(y_true, y_pred)
        hyper_sensitivity = self.clinical_detector.sensitivity_hyperglycemia(y_true, y_pred)
        
        results['hypoglycemia_sensitivity'] = hypo_sensitivity
        results['hyperglycemia_sensitivity'] = hyper_sensitivity
        
        return results
    
    def compare_models(self, scenario_id: str, model_a_name: str = 'Model_A', 
                      model_b_name: str = 'Model_B') -> Dict[str, Any]:
        """
        Compare two models for a scenario
        
        Args:
            scenario_id: ID of scenario to analyze
            model_a_name: Name of first model
            model_b_name: Name of second model
            
        Returns:
            Dictionary with comparison results
        """
        if not self.scenarios:
            raise ValueError("Scenarios not initialized")
        
        scenario = self.scenarios.get_scenario(scenario_id)
        
        if not scenario.model_predictions:
            raise ValueError(f"No model predictions available for scenario {scenario_id}")
        
        if model_a_name not in scenario.model_predictions:
            raise ValueError(f"Model {model_a_name} not found in scenario {scenario_id}")
        
        if model_b_name not in scenario.model_predictions:
            raise ValueError(f"Model {model_b_name} not found in scenario {scenario_id}")
        
        y_true = scenario.y_true
        pred_a = scenario.model_predictions[model_a_name]
        pred_b = scenario.model_predictions[model_b_name]
        
        # Calculate metrics for both models
        madex_a = self.metric_calculator.compute_madex(y_true, pred_a, self.madex_params)
        madex_b = self.metric_calculator.compute_madex(y_true, pred_b, self.madex_params)
        
        # Determine winner and clinical implications
        if madex_a < madex_b:
            madex_winner = model_a_name
            clinical_implication = f"{model_a_name} shows better MADEX performance ({madex_a:.2f} vs {madex_b:.2f})"
        else:
            madex_winner = model_b_name
            clinical_implication = f"{model_b_name} shows better MADEX performance ({madex_b:.2f} vs {madex_a:.2f})"
        
        # Log comparison result
        self.logger.log_comparison_result(
            model_a_name, model_b_name, "MADEX", madex_winner, clinical_implication
        )
        
        return {
            'scenario_id': scenario_id,
            'model_a': model_a_name,
            'model_b': model_b_name,
            'madex_a': madex_a,
            'madex_b': madex_b,
            'madex_winner': madex_winner,
            'clinical_implication': clinical_implication
        }
    
    def analyze_all_scenarios(self, enable_detailed_logging: bool = True) -> Dict[str, Any]:
        """
        Analyze all available scenarios
        
        Args:
            enable_detailed_logging: Whether to enable detailed logging for each scenario
            
        Returns:
            Dictionary with all scenario results
        """
        if not self.scenarios:
            raise ValueError("Scenarios not initialized")
        
        all_results = {}
        scenario_ids = self.get_scenario_list()
        
        self.logger.log_section_header("SCENARIO-BY-SCENARIO ANALYSIS", level=2)
        self.logger.log_narrative(f"Total scenarios to analyze: {len(scenario_ids)}")
        
        for i, scenario_id in enumerate(scenario_ids, 1):
            self.logger.log_progress(i, len(scenario_ids), f"Analyzing scenario {scenario_id}")
            
            scenario_results = self.analyze_scenario(scenario_id, enable_detailed_logging)
            all_results[scenario_id] = scenario_results
        
        return all_results


class StatisticalAnalysis(AnalysisFramework):
    """
    Statistical analysis specialization for significance testing and validation
    
    Provides bootstrap confidence intervals, statistical significance testing,
    and cross-scenario statistical validation.
    """
    
    def __init__(self, module_name: str, script_dir: Optional[Path] = None):
        """Initialize statistical analysis framework"""
        super().__init__(module_name, script_dir)
    
    def bootstrap_scenario_analysis(self, scenario_id: str, n_bootstrap: int = 1000, 
                                   ci_level: float = 95) -> Dict[str, Any]:
        """
        Perform bootstrap analysis for a single scenario
        
        Args:
            scenario_id: ID of scenario to analyze
            n_bootstrap: Number of bootstrap samples
            ci_level: Confidence interval level
            
        Returns:
            Dictionary with bootstrap results
        """
        if not self.scenarios:
            raise ValueError("Scenarios not initialized")
        
        scenario = self.scenarios.get_scenario(scenario_id)
        results = {'scenario_id': scenario_id, 'n_bootstrap': n_bootstrap, 'ci_level': ci_level}
        
        if scenario.model_predictions:
            y_true = scenario.y_true
            
            for model_name, predictions in scenario.model_predictions.items():
                # Bootstrap MADEX
                madex_func = lambda yt, yp: self.metric_calculator.compute_madex(
                    yt, yp, self.madex_params
                )
                madex_mean, madex_lower, madex_upper = self.bootstrap_analyzer.bootstrap_metric(
                    y_true, predictions, madex_func, n_bootstrap, ci_level
                )
                
                # Bootstrap traditional metrics
                rmse_mean, rmse_lower, rmse_upper = self.bootstrap_analyzer.bootstrap_metric(
                    y_true, predictions, self.metric_calculator.compute_rmse, n_bootstrap, ci_level
                )
                mae_mean, mae_lower, mae_upper = self.bootstrap_analyzer.bootstrap_metric(
                    y_true, predictions, self.metric_calculator.compute_mae, n_bootstrap, ci_level
                )
                
                results[model_name] = {
                    'madex': {
                        'mean': madex_mean,
                        'ci_lower': madex_lower,
                        'ci_upper': madex_upper,
                        'formatted': f"{madex_mean:.2f} ({madex_lower:.2f}-{madex_upper:.2f})"
                    },
                    'rmse': {
                        'mean': rmse_mean,
                        'ci_lower': rmse_lower,
                        'ci_upper': rmse_upper,
                        'formatted': f"{rmse_mean:.2f} ({rmse_lower:.2f}-{rmse_upper:.2f})"
                    },
                    'mae': {
                        'mean': mae_mean,
                        'ci_lower': mae_lower,
                        'ci_upper': mae_upper,
                        'formatted': f"{mae_mean:.2f} ({mae_lower:.2f}-{mae_upper:.2f})"
                    }
                }
        
        return results
    
    def cross_scenario_significance_testing(self, metric_name: str = 'MADEX', 
                                          model_a_name: str = 'Model_A',
                                          model_b_name: str = 'Model_B') -> Dict[str, Any]:
        """
        Perform cross-scenario significance testing
        
        Args:
            metric_name: Name of metric to test
            model_a_name: Name of first model
            model_b_name: Name of second model
            
        Returns:
            Dictionary with significance test results
        """
        if not self.scenarios:
            raise ValueError("Scenarios not initialized")
        
        self.logger.log_section_header("STATISTICAL SIGNIFICANCE TESTING", level=2)
        self.logger.log_narrative(f"Performing {metric_name} significance testing across scenarios")
        self.logger.log_narrative(f"Testing hypothesis: {model_a_name} vs {model_b_name} performance differences")
        
        # Collect metric values across all scenarios
        values_a = []
        values_b = []
        scenario_ids = []
        
        for scenario_id in self.get_scenario_list():
            scenario = self.scenarios.get_scenario(scenario_id)
            
            if (scenario.model_predictions and 
                model_a_name in scenario.model_predictions and 
                model_b_name in scenario.model_predictions):
                
                y_true = scenario.y_true
                pred_a = scenario.model_predictions[model_a_name]
                pred_b = scenario.model_predictions[model_b_name]
                
                if metric_name.upper() == 'MADEX':
                    value_a = self.metric_calculator.compute_madex(y_true, pred_a, self.madex_params)
                    value_b = self.metric_calculator.compute_madex(y_true, pred_b, self.madex_params)
                elif metric_name.upper() == 'RMSE':
                    value_a = self.metric_calculator.compute_rmse(y_true, pred_a)
                    value_b = self.metric_calculator.compute_rmse(y_true, pred_b)
                elif metric_name.upper() == 'MAE':
                    value_a = self.metric_calculator.compute_mae(y_true, pred_a)
                    value_b = self.metric_calculator.compute_mae(y_true, pred_b)
                else:
                    raise ValueError(f"Unsupported metric: {metric_name}")
                
                values_a.append(value_a)
                values_b.append(value_b)
                scenario_ids.append(scenario_id)
        
        if len(values_a) == 0:
            raise ValueError("No valid scenarios found for significance testing")
        
        # Perform Wilcoxon signed-rank test
        test_results = self.significance_tester.wilcoxon_test_with_interpretation(
            values_a, values_b, metric_name
        )
        
        test_results.update({
            'scenario_count': len(scenario_ids),
            'scenario_ids': scenario_ids,
            'values_a': values_a,
            'values_b': values_b,
            'model_a_name': model_a_name,
            'model_b_name': model_b_name
        })
        
        return test_results
    
    def confidence_interval_overlap_analysis(self, scenario_results: Dict[str, Any], 
                                           model_a_name: str = 'Model_A',
                                           model_b_name: str = 'Model_B') -> Dict[str, Any]:
        """
        Analyze confidence interval overlaps between models
        
        Args:
            scenario_results: Results from bootstrap_scenario_analysis
            model_a_name: Name of first model
            model_b_name: Name of second model
            
        Returns:
            Dictionary with overlap analysis results
        """
        overlap_results = {}
        
        for scenario_id, results in scenario_results.items():
            if (model_a_name in results and model_b_name in results and
                'madex' in results[model_a_name] and 'madex' in results[model_b_name]):
                
                ci_a = (results[model_a_name]['madex']['ci_lower'], 
                       results[model_a_name]['madex']['ci_upper'])
                ci_b = (results[model_b_name]['madex']['ci_lower'], 
                       results[model_b_name]['madex']['ci_upper'])
                
                overlap = self.bootstrap_analyzer.check_ci_overlap(ci_a, ci_b)
                
                overlap_results[scenario_id] = {
                    'overlap': overlap,
                    'ci_a': ci_a,
                    'ci_b': ci_b,
                    'interpretation': "overlapping" if overlap else "non-overlapping"
                }
                
                # Log individual scenario overlap
                if overlap:
                    self.logger.log_clinical_assessment('WARNING',
                        f"Scenario {scenario_id}: MADEX confidence intervals overlap - "
                        f"difference may not be statistically significant")
                else:
                    self.logger.log_clinical_assessment('SUCCESS',
                        f"Scenario {scenario_id}: MADEX confidence intervals do not overlap - "
                        f"significant difference detected")
        
        return overlap_results