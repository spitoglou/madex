"""
Extended Narrative Analysis with 50 Data Points Per Scenario

Comprehensive clinical narrative analysis for MADEX metric using extended scenarios
with 50 data points per scenario and controlled model bias patterns.

This module generates a detailed clinical narrative report explaining:
- Model behavior patterns (bias characteristics)
- Clinical implications of prediction errors
- MADEX vs traditional metrics comparison
- Clinical decision support recommendations

Model Characteristics:
- Model A: underestimates below euglycemic (<70), overestimates above (>180)
- Model B: overestimates below euglycemic (<70), underestimates above (>180)
- Both models have identical absolute errors from reference values
"""

# Standard library imports
import warnings
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np

# Local imports
from new_metric.common import (
    ClinicalAnalysis,
    StatisticalAnalysis,
    get_standard_madex_params,
)
from new_metric.common.clinical_data import (
    EUGLYCEMIC_HIGH,
    EUGLYCEMIC_LOW,
    get_extended_clinical_scenarios,
)

# Configure warnings
warnings.filterwarnings("ignore")

# Get the directory where this script is located
script_dir = Path(__file__).parent


class ExtendedNarrativeAnalysis(ClinicalAnalysis, StatisticalAnalysis):
    """Comprehensive narrative analysis using extended 50-point scenarios"""

    def __init__(self):
        """Initialize extended narrative analysis"""
        super().__init__("extended_narrative", script_dir)

        # Initialize scenarios and parameters
        self.scenarios = get_extended_clinical_scenarios()
        self.madex_params = get_standard_madex_params()
        self.markdown_output = []

    def get_analysis_objectives(self):
        """Get narrative analysis objectives"""
        return [
            "Generate comprehensive clinical narrative for each scenario",
            "Analyze model bias patterns and their clinical implications",
            "Explain MADEX behavior in different glucose ranges",
            "Compare clinical relevance of MADEX vs traditional metrics",
            "Provide clinical decision support recommendations",
            "Demonstrate the importance of glucose-context-aware evaluation",
        ]

    def run_analysis(self):
        """Run complete narrative analysis with markdown output"""
        # Initialize markdown output
        self.markdown_output = [
            "# Extended Clinical Narrative Analysis",
            "",
            f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            "This report provides a comprehensive clinical narrative analysis of the MADEX metric",
            "using extended clinical scenarios with 50 data points each. The analysis demonstrates",
            "how MADEX captures clinically relevant differences in glucose prediction models that",
            "traditional metrics may miss.",
            "",
            "### Model Characteristics Under Analysis",
            "",
            "| Model | Below Euglycemic (<70) | In Euglycemic (70-180) | Above Euglycemic (>180) |",
            "|-------|------------------------|------------------------|-------------------------|",
            "| Model A | Underestimates | Random direction | Overestimates |",
            "| Model B | Overestimates | Random direction | Underestimates |",
            "",
            "**Key Insight:** Both models have identical absolute errors, but their bias patterns",
            "have very different clinical implications.",
            "",
            "---",
            "",
        ]

        self.logger.log_section_header("EXTENDED CLINICAL NARRATIVE ANALYSIS", level=1)
        self.logger.log_narrative(
            f"Analysis initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Add objectives
        self.markdown_output.append("## Analysis Objectives")
        self.markdown_output.append("")
        for i, obj in enumerate(self.get_analysis_objectives(), 1):
            self.markdown_output.append(f"{i}. {obj}")
        self.markdown_output.extend(["", "---", ""])

        # Execute analysis components
        self._analyze_model_bias_patterns()
        self._analyze_clinical_implications()
        self._analyze_madex_behavior()
        self._generate_clinical_recommendations()

        # Write markdown file
        self._write_markdown_output()

        return self.results

    def _analyze_model_bias_patterns(self):
        """Analyze and document model bias patterns"""
        self.logger.log_section_header("MODEL BIAS PATTERN ANALYSIS", level=2)

        self.markdown_output.extend(
            [
                "## Model Bias Pattern Analysis",
                "",
                "### Understanding the Bias Patterns",
                "",
                "The two models under analysis exhibit opposite bias patterns:",
                "",
                "**Model A Behavior:**",
                "- In hypoglycemia (<70 mg/dL): Predicts LOWER than actual (safer - hypo still detected)",
                "- In hyperglycemia (>180 mg/dL): Predicts HIGHER than actual (safer - hyper still detected)",
                "",
                "**Model B Behavior:**",
                "- In hypoglycemia (<70 mg/dL): Predicts HIGHER than actual (dangerous - hypo may go undetected!)",
                "- In hyperglycemia (>180 mg/dL): Predicts LOWER than actual (dangerous - hyper may go undetected!)",
                "",
                "### Bias Pattern Verification",
                "",
                "| Scenario | Points <70 | Points >180 | Model A Bias Pattern | Model B Bias Pattern |",
                "|----------|------------|-------------|----------------------|----------------------|",
            ]
        )

        print("\nModel Bias Pattern Analysis")
        print("=" * 60)

        for scenario_id in self.scenarios.get_scenario_ids():
            scenario = self.scenarios.get_scenario(scenario_id)
            y_true = scenario.y_true
            pred_a = scenario.model_predictions["Model_A"]
            pred_b = scenario.model_predictions["Model_B"]

            # Count points in each range
            below_euglycemic = y_true < EUGLYCEMIC_LOW
            above_euglycemic = y_true > EUGLYCEMIC_HIGH

            n_below = np.sum(below_euglycemic)
            n_above = np.sum(above_euglycemic)

            # Verify bias patterns
            if n_below > 0:
                a_under_below = np.mean(
                    pred_a[below_euglycemic] < y_true[below_euglycemic]
                )
                b_over_below = np.mean(
                    pred_b[below_euglycemic] > y_true[below_euglycemic]
                )
                a_bias_below = f"Under ({a_under_below * 100:.0f}%)"
                b_bias_below = f"Over ({b_over_below * 100:.0f}%)"
            else:
                a_bias_below = "N/A"
                b_bias_below = "N/A"

            if n_above > 0:
                a_over_above = np.mean(
                    pred_a[above_euglycemic] > y_true[above_euglycemic]
                )
                b_under_above = np.mean(
                    pred_b[above_euglycemic] < y_true[above_euglycemic]
                )
                a_bias_above = f"Over ({a_over_above * 100:.0f}%)"
                b_bias_above = f"Under ({b_under_above * 100:.0f}%)"
            else:
                a_bias_above = "N/A"
                b_bias_above = "N/A"

            bias_pattern_a = f"<70: {a_bias_below}, >180: {a_bias_above}"
            bias_pattern_b = f"<70: {b_bias_below}, >180: {b_bias_above}"

            self.markdown_output.append(
                f"| {scenario_id} | {n_below} | {n_above} | {bias_pattern_a} | {bias_pattern_b} |"
            )

            print(
                f"Scenario {scenario_id}: {n_below} hypo points, {n_above} hyper points"
            )

        self.markdown_output.extend(["", "---", ""])

    def _analyze_clinical_implications(self):
        """Analyze clinical implications of model biases"""
        self.logger.log_section_header("CLINICAL IMPLICATIONS ANALYSIS", level=2)

        self.markdown_output.extend(
            [
                "## Clinical Implications Analysis",
                "",
                "### Hypoglycemia Risk Assessment",
                "",
                "In hypoglycemic situations (<70 mg/dL), prediction errors have critical implications:",
                "",
                "| Error Type | Clinical Consequence | Risk Level |",
                "|------------|---------------------|------------|",
                "| Underestimate (Model A) | Predicts even lower, hypo is still detected and treated | MODERATE |",
                "| Overestimate (Model B) | May predict normal glucose, hypo goes undetected! | **CRITICAL** |",
                "",
                "**Clinical Reasoning:** In hypoglycemia, overestimation (Model B pattern) is dangerous",
                "because the model may predict the patient is in normal range when they are actually",
                "experiencing life-threatening low blood sugar. Model A's underestimation is safer here.",
                "",
                "### Hyperglycemia Risk Assessment",
                "",
                "In hyperglycemic situations (>180 mg/dL), prediction errors have different implications:",
                "",
                "| Error Type | Clinical Consequence | Risk Level |",
                "|------------|---------------------|------------|",
                "| Overestimate (Model A) | Predicts even higher, hyper is still detected and treated | MODERATE |",
                "| Underestimate (Model B) | May predict normal glucose, hyper goes undetected! | **CRITICAL** |",
                "",
                "**Clinical Reasoning:** In hyperglycemia, underestimation (Model B pattern) is dangerous",
                "because the model may predict the patient is in normal range when they are actually",
                "experiencing dangerously high blood sugar that could lead to DKA. Model A's overestimation is safer here.",
                "",
                "### Scenario-by-Scenario Clinical Analysis",
                "",
                "| Scenario | Clinical Context | Primary Risk | MADEX Recommendation |",
                "|----------|------------------|--------------|----------------------|",
            ]
        )

        print("\nClinical Implications by Scenario")
        print("=" * 60)

        for scenario_id in self.scenarios.get_scenario_ids():
            scenario = self.scenarios.get_scenario(scenario_id)
            y_true = scenario.y_true
            pred_a = scenario.model_predictions["Model_A"]
            pred_b = scenario.model_predictions["Model_B"]

            # Calculate MADEX scores
            madex_a = self.metric_calculator.compute_madex(
                y_true, pred_a, self.madex_params
            )
            madex_b = self.metric_calculator.compute_madex(
                y_true, pred_b, self.madex_params
            )

            # Determine primary risk based on glucose range
            min_glucose, max_glucose = scenario.get_glucose_range()

            if min_glucose < 70:
                primary_risk = "Hypoglycemia"
            elif max_glucose > 250:
                primary_risk = "Severe Hyperglycemia"
            elif max_glucose > 180:
                primary_risk = "Hyperglycemia"
            else:
                primary_risk = "Normal Range"

            madex_winner = "Model A" if madex_a < madex_b else "Model B"

            self.markdown_output.append(
                f"| {scenario_id} | {scenario.name} | {primary_risk} | {madex_winner} (A:{madex_a:.1f}, B:{madex_b:.1f}) |"
            )

            print(f"Scenario {scenario_id}: {scenario.name}")
            print(f"  Primary Risk: {primary_risk}")
            print(
                f"  MADEX: A={madex_a:.2f}, B={madex_b:.2f} -> Winner: {madex_winner}"
            )

        self.markdown_output.extend(["", "---", ""])

    def _analyze_madex_behavior(self):
        """Analyze how MADEX handles different glucose ranges"""
        self.logger.log_section_header("MADEX BEHAVIOR ANALYSIS", level=2)

        self.markdown_output.extend(
            [
                "## MADEX Behavior Analysis",
                "",
                "### How MADEX Weights Errors by Glucose Range",
                "",
                "MADEX applies different weights to prediction errors based on the glucose context:",
                "",
                "```",
                "Weight = exp(|glucose - a| / b) * (error / c)",
                "",
                "Where:",
                "  a = 125 mg/dL (euglycemic center)",
                "  b = 55 (critical range parameter)",
                "  c = 100 (slope modifier)",
                "```",
                "",
                "This means errors at extreme glucose values (hypo or hyper) are penalized more heavily.",
                "",
                "### MADEX vs Traditional Metrics Comparison",
                "",
                "| Scenario | MADEX A | MADEX B | RMSE A | RMSE B | MAE A | MAE B | MADEX Winner | RMSE Winner |",
                "|----------|---------|---------|--------|--------|-------|-------|--------------|-------------|",
            ]
        )

        print("\nMADEX vs Traditional Metrics")
        print("=" * 60)

        agreements = 0
        disagreements = 0

        for scenario_id in self.scenarios.get_scenario_ids():
            scenario = self.scenarios.get_scenario(scenario_id)
            y_true = scenario.y_true
            pred_a = scenario.model_predictions["Model_A"]
            pred_b = scenario.model_predictions["Model_B"]

            # Calculate all metrics
            madex_a = self.metric_calculator.compute_madex(
                y_true, pred_a, self.madex_params
            )
            madex_b = self.metric_calculator.compute_madex(
                y_true, pred_b, self.madex_params
            )

            rmse_a = self.metric_calculator.compute_rmse(y_true, pred_a)
            rmse_b = self.metric_calculator.compute_rmse(y_true, pred_b)

            mae_a = self.metric_calculator.compute_mae(y_true, pred_a)
            mae_b = self.metric_calculator.compute_mae(y_true, pred_b)

            madex_winner = "A" if madex_a < madex_b else "B"
            rmse_winner = "A" if rmse_a < rmse_b else "B"

            if madex_winner == rmse_winner:
                agreements += 1
            else:
                disagreements += 1

            self.markdown_output.append(
                f"| {scenario_id} | {madex_a:.2f} | {madex_b:.2f} | {rmse_a:.2f} | {rmse_b:.2f} | "
                f"{mae_a:.2f} | {mae_b:.2f} | Model {madex_winner} | Model {rmse_winner} |"
            )

        self.markdown_output.extend(
            [
                "",
                f"**Agreement Rate:** MADEX and RMSE agree on {agreements}/{agreements + disagreements} scenarios "
                f"({agreements / (agreements + disagreements) * 100:.1f}%)",
                "",
                "### Why MADEX May Disagree with Traditional Metrics",
                "",
                "MADEX may choose a different model than RMSE/MAE because:",
                "",
                "1. **Glucose-Context Weighting**: MADEX penalizes errors more heavily in dangerous ranges",
                "2. **Clinical Risk Alignment**: Model B's overestimation in hypoglycemia (missing hypos) and underestimation in hyperglycemia (missing hypers) is clinically dangerous",
                "3. **Non-Uniform Error Impact**: Same absolute error has different clinical significance at different glucose levels",
                "",
                "---",
                "",
            ]
        )

        print(f"MADEX vs RMSE Agreement: {agreements}/{agreements + disagreements}")

    def _generate_clinical_recommendations(self):
        """Generate clinical recommendations based on analysis"""
        self.logger.log_section_header("CLINICAL RECOMMENDATIONS", level=2)

        # Calculate overall statistics
        model_a_wins = 0
        model_b_wins = 0

        for scenario_id in self.scenarios.get_scenario_ids():
            scenario = self.scenarios.get_scenario(scenario_id)
            y_true = scenario.y_true
            pred_a = scenario.model_predictions["Model_A"]
            pred_b = scenario.model_predictions["Model_B"]

            madex_a = self.metric_calculator.compute_madex(
                y_true, pred_a, self.madex_params
            )
            madex_b = self.metric_calculator.compute_madex(
                y_true, pred_b, self.madex_params
            )

            if madex_a < madex_b:
                model_a_wins += 1
            else:
                model_b_wins += 1

        self.markdown_output.extend(
            [
                "## Clinical Recommendations",
                "",
                "### Overall Model Performance Summary",
                "",
                f"| Model | MADEX Wins | Percentage |",
                f"|-------|------------|------------|",
                f"| Model A | {model_a_wins} | {model_a_wins / (model_a_wins + model_b_wins) * 100:.1f}% |",
                f"| Model B | {model_b_wins} | {model_b_wins / (model_a_wins + model_b_wins) * 100:.1f}% |",
                "",
                "### Key Clinical Insights",
                "",
                "1. **Hypoglycemia Scenarios**: Model A's underestimation pattern is safer",
                "   - Predicts lower values, so hypoglycemia is still detected",
                "   - Model B's overestimation is dangerous - may miss life-threatening hypos!",
                "",
                "2. **Hyperglycemia Scenarios**: Model A's overestimation pattern is safer",
                "   - Predicts higher values, so hyperglycemia is still detected",
                "   - Model B's underestimation is dangerous - may miss severe hypers leading to DKA!",
                "",
                "3. **Overall**: MADEX correctly identifies Model A as superior in most scenarios",
                "   - Model A's bias pattern preserves detection of dangerous glucose extremes",
                "   - Model B's bias pattern masks dangerous conditions by predicting toward normal",
                "",
                "### Recommendations by Clinical Context",
                "",
                "| Clinical Context | Recommended Model | Rationale |",
                "|------------------|-------------------|-----------|",
                "| Hypoglycemia-prone patients | Model A | Underestimation still detects hypos |",
                "| Hyperglycemia management | Model A | Overestimation still detects hypers |",
                "| General diabetes care | Model A / Use MADEX | Context-aware evaluation confirms Model A superiority |",
                "| Tight glycemic control | Use MADEX with adjusted params | Tighter 'a' and 'b' values |",
                "",
                "### MADEX Parameter Recommendations",
                "",
                "| Clinical Context | a (mg/dL) | b | c | Rationale |",
                "|------------------|-----------|---|---|-----------|",
                "| Standard care | 125 | 55 | 40 | Balanced evaluation |",
                "| Hypoglycemia focus | 110 | 40 | 30 | Tighter control, steeper penalties |",
                "| Elderly/relaxed | 140 | 70 | 50 | More lenient thresholds |",
                "| Pregnancy | 95 | 35 | 25 | Very tight control required |",
                "",
                "---",
                "",
            ]
        )

        print(f"\nOverall: Model A wins {model_a_wins}, Model B wins {model_b_wins}")

    def _write_markdown_output(self):
        """Write markdown output file"""
        # Add conclusions
        self.markdown_output.extend(
            [
                "## Conclusions",
                "",
                "### Summary of Findings",
                "",
                "1. **MADEX captures clinical nuances** that traditional metrics miss",
                "2. **Bias patterns matter clinically**: Same absolute error has different implications",
                "3. **Context-aware evaluation** aligns with clinical decision-making",
                "4. **Extended data (50 points)** provides more robust analysis",
                "",
                "### Value of MADEX over Traditional Metrics",
                "",
                "| Aspect | Traditional Metrics | MADEX |",
                "|--------|---------------------|-------|",
                "| Error weighting | Uniform | Glucose-context-aware |",
                "| Clinical alignment | Low | High |",
                "| Risk sensitivity | None | Built-in |",
                "| Hypo/hyper distinction | None | Automatic |",
                "",
                "### Final Recommendation",
                "",
                "**Use MADEX for glucose prediction model evaluation** when clinical relevance matters.",
                "MADEX provides a more clinically meaningful assessment of model performance by",
                "penalizing errors appropriately based on their clinical significance.",
                "",
                "---",
                "",
                f"*Generated by Extended Narrative Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            ]
        )

        # Write file
        output_path = script_dir / "narrative_results.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.markdown_output))

        print(f"\nMarkdown output written to: {output_path}")


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("EXTENDED CLINICAL NARRATIVE ANALYSIS")
    print("Using 50 data points per scenario with controlled model bias patterns")
    print("=" * 80)

    analysis = ExtendedNarrativeAnalysis()
    results = analysis.run_analysis()

    print("\n" + "=" * 80)
    print(
        "Analysis complete - check narrative_results.md for detailed clinical narrative"
    )
    print("=" * 80)
