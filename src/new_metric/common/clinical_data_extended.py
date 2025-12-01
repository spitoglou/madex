"""
Extended Clinical Data with 50 Data Points Per Scenario

Clinical scenario definitions with time-series-like data where:
- Model A underestimates values below euglycemic range, overestimates above it
- Model B overestimates values below euglycemic range, underestimates above it
- Both models have identical absolute errors from reference values
- Euglycemic range: 70-180 mg/dL (standard clinical definition)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .clinical_data import ClinicalScenario, MADEXParameters, ScenarioRepository

# Euglycemic range boundaries
EUGLYCEMIC_LOW = 70  # mg/dL
EUGLYCEMIC_HIGH = 180  # mg/dL


def generate_model_predictions(
    y_true: np.ndarray, error_magnitude: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Model A and Model B predictions with opposite bias patterns.

    Model A behavior:
    - Below euglycemic range (<70): underestimates (prediction < true)
    - Above euglycemic range (>180): overestimates (prediction > true)
    - In euglycemic range: random direction, same magnitude

    Model B behavior:
    - Below euglycemic range (<70): overestimates (prediction > true)
    - Above euglycemic range (>180): underestimates (prediction < true)
    - In euglycemic range: opposite direction from Model A

    Both models have identical absolute errors.

    Args:
        y_true: Reference glucose values
        error_magnitude: Absolute error magnitude for each point
        seed: Random seed for euglycemic range direction

    Returns:
        Tuple of (model_a_predictions, model_b_predictions)
    """
    np.random.seed(seed)

    n = len(y_true)
    model_a = np.zeros(n)
    model_b = np.zeros(n)

    for i in range(n):
        true_val = y_true[i]
        error = error_magnitude[i]

        if true_val < EUGLYCEMIC_LOW:
            # Below euglycemic: A underestimates, B overestimates
            model_a[i] = true_val - error
            model_b[i] = true_val + error
        elif true_val > EUGLYCEMIC_HIGH:
            # Above euglycemic: A overestimates, B underestimates
            model_a[i] = true_val + error
            model_b[i] = true_val - error
        else:
            # In euglycemic range: random direction, opposite between models
            direction = np.random.choice([-1, 1])
            model_a[i] = true_val + direction * error
            model_b[i] = true_val - direction * error

    # Ensure no negative predictions
    model_a = np.maximum(model_a, 1)
    model_b = np.maximum(model_b, 1)

    return model_a, model_b


class ExtendedScenarioRepository(ScenarioRepository):
    """
    Extended repository with 50 data points per scenario in time-series format.
    """

    def __init__(self):
        """Initialize repository with extended clinical scenarios"""
        self.scenarios: Dict[str, ClinicalScenario] = {}
        self._initialize_extended_scenarios()

    def _initialize_extended_scenarios(self):
        """Initialize repository with extended 50-point clinical scenarios"""

        # Scenario A: Hypoglycemia Detection
        # Time series: glucose dropping into hypoglycemia, staying low, then recovering
        y_true_a = np.concatenate(
            [
                np.linspace(85, 55, 10),  # Dropping into hypoglycemia
                np.linspace(55, 40, 10),  # Deepening hypoglycemia
                np.array(
                    [38, 35, 33, 32, 30, 32, 35, 38, 42, 45]
                ),  # Critical low, start recovery
                np.linspace(45, 60, 10),  # Recovery phase
                np.linspace(60, 75, 10),  # Return toward normal
            ]
        )
        error_a = np.concatenate(
            [
                np.linspace(5, 10, 10),
                np.linspace(10, 15, 10),
                np.array([15, 16, 17, 18, 20, 18, 16, 15, 13, 12]),
                np.linspace(12, 8, 10),
                np.linspace(8, 5, 10),
            ]
        )
        pred_a_a, pred_b_a = generate_model_predictions(y_true_a, error_a, seed=1)

        # Scenario B: Hyperglycemia Management
        # Time series: glucose rising into hyperglycemia, peaking, then decreasing with treatment
        y_true_b = np.concatenate(
            [
                np.linspace(150, 200, 10),  # Rising toward hyperglycemia
                np.linspace(200, 280, 10),  # Entering severe hyperglycemia
                np.array(
                    [290, 300, 310, 320, 325, 320, 310, 300, 285, 270]
                ),  # Peak and start descent
                np.linspace(270, 220, 10),  # Treatment effect
                np.linspace(220, 170, 10),  # Return toward normal
            ]
        )
        error_b = np.concatenate(
            [
                np.linspace(8, 12, 10),
                np.linspace(12, 20, 10),
                np.array([22, 25, 28, 30, 32, 30, 28, 25, 22, 20]),
                np.linspace(20, 15, 10),
                np.linspace(15, 10, 10),
            ]
        )
        pred_a_b, pred_b_b = generate_model_predictions(y_true_b, error_b, seed=2)

        # Scenario C: Postprandial Response
        # Time series: meal spike pattern (baseline -> spike -> return)
        y_true_c = np.concatenate(
            [
                np.linspace(95, 110, 8),  # Pre-meal baseline
                np.linspace(110, 180, 10),  # Rapid rise after meal
                np.linspace(180, 220, 8),  # Peak postprandial
                np.linspace(220, 160, 12),  # Descent from peak
                np.linspace(160, 105, 12),  # Return to baseline
            ]
        )
        error_c = np.concatenate(
            [
                np.linspace(5, 6, 8),
                np.linspace(6, 10, 10),
                np.linspace(10, 15, 8),
                np.linspace(15, 10, 12),
                np.linspace(10, 5, 12),
            ]
        )
        pred_a_c, pred_b_c = generate_model_predictions(y_true_c, error_c, seed=3)

        # Scenario D: Dawn Phenomenon
        # Time series: overnight stability then early morning rise
        y_true_d = np.concatenate(
            [
                np.linspace(110, 100, 10),  # Evening descent
                np.array([98, 95, 93, 92, 90, 90, 92, 95, 100, 105]),  # Overnight low
                np.linspace(105, 130, 10),  # Dawn rise begins
                np.linspace(130, 160, 10),  # Dawn phenomenon peak
                np.linspace(160, 140, 10),  # Morning stabilization
            ]
        )
        error_d = np.concatenate(
            [
                np.linspace(5, 6, 10),
                np.array([6, 6, 7, 7, 8, 8, 7, 7, 6, 6]),
                np.linspace(6, 8, 10),
                np.linspace(8, 10, 10),
                np.linspace(10, 7, 10),
            ]
        )
        pred_a_d, pred_b_d = generate_model_predictions(y_true_d, error_d, seed=4)

        # Scenario E: Exercise Response
        # Time series: pre-exercise, rapid drop during exercise, post-exercise recovery
        y_true_e = np.concatenate(
            [
                np.linspace(140, 130, 8),  # Pre-exercise
                np.linspace(130, 85, 12),  # Exercise-induced drop
                np.linspace(85, 60, 10),  # Risk of exercise-induced hypoglycemia
                np.linspace(60, 90, 10),  # Recovery snack effect
                np.linspace(90, 120, 10),  # Return to baseline
            ]
        )
        error_e = np.concatenate(
            [
                np.linspace(6, 8, 8),
                np.linspace(8, 12, 12),
                np.linspace(12, 18, 10),
                np.linspace(18, 10, 10),
                np.linspace(10, 6, 10),
            ]
        )
        pred_a_e, pred_b_e = generate_model_predictions(y_true_e, error_e, seed=5)

        # Scenario F: Measurement Noise (Stable Euglycemic)
        # Time series: stable glucose with sensor noise variations
        base_f = 110 + 15 * np.sin(
            np.linspace(0, 4 * np.pi, 50)
        )  # Gentle oscillation around 110
        noise_f = np.random.RandomState(6).normal(0, 5, 50)
        y_true_f = np.clip(base_f + noise_f, 85, 140)
        error_f = 5 + 3 * np.abs(
            np.sin(np.linspace(0, 6 * np.pi, 50))
        )  # Varying small errors
        pred_a_f, pred_b_f = generate_model_predictions(y_true_f, error_f, seed=6)

        # Scenario G: Mixed Clinical (Full Range)
        # Time series: complex pattern hitting all glucose ranges
        y_true_g = np.concatenate(
            [
                np.linspace(100, 50, 8),  # Drop to hypoglycemia
                np.linspace(50, 35, 5),  # Severe hypo
                np.linspace(35, 80, 7),  # Recovery
                np.linspace(80, 150, 8),  # Rise through normal
                np.linspace(150, 250, 8),  # Into hyperglycemia
                np.linspace(250, 300, 6),  # Severe hyper
                np.linspace(300, 180, 8),  # Treatment bringing down
            ]
        )
        error_g = np.concatenate(
            [
                np.linspace(8, 15, 8),
                np.linspace(15, 20, 5),
                np.linspace(20, 10, 7),
                np.linspace(10, 12, 8),
                np.linspace(12, 20, 8),
                np.linspace(20, 25, 6),
                np.linspace(25, 15, 8),
            ]
        )
        pred_a_g, pred_b_g = generate_model_predictions(y_true_g, error_g, seed=7)

        # Scenario H: Extreme Cases
        # Time series: life-threatening extremes on both ends
        y_true_h = np.concatenate(
            [
                np.linspace(80, 40, 8),  # Dropping to severe hypo
                np.array([35, 30, 25, 22, 20, 22, 25, 30, 40, 55]),  # Critical low
                np.linspace(55, 120, 8),  # Emergency recovery
                np.linspace(120, 300, 8),  # Rapid rise to hyper
                np.array(
                    [320, 350, 380, 400, 420, 440, 450, 440, 420, 380]
                ),  # Extreme high
                np.linspace(380, 250, 6),  # Emergency treatment
            ]
        )
        # Error magnitudes capped to ensure predictions stay positive (error < y_true - 1)
        error_h = np.concatenate(
            [
                np.linspace(10, 15, 8),
                np.array(
                    [12, 10, 8, 7, 6, 7, 8, 10, 14, 18]
                ),  # Reduced for very low values
                np.linspace(18, 10, 8),
                np.linspace(10, 25, 8),
                np.array([28, 32, 35, 38, 40, 42, 45, 42, 40, 35]),
                np.linspace(35, 25, 6),
            ]
        )
        pred_a_h, pred_b_h = generate_model_predictions(y_true_h, error_h, seed=8)

        # Create scenario definitions
        extended_scenarios = {
            "A": {
                "name": "Hypoglycemia Detection",
                "description": "Hypoglycemic episode with recovery (30-85 mg/dL)",
                "y_true": y_true_a,
                "model_a": pred_a_a,
                "model_b": pred_b_a,
                "clinical_significance": "Life-threatening hypoglycemia requiring immediate intervention",
                "risk_level": "CRITICAL",
            },
            "B": {
                "name": "Hyperglycemia Management",
                "description": "Hyperglycemic episode with treatment (150-325 mg/dL)",
                "y_true": y_true_b,
                "model_a": pred_a_b,
                "model_b": pred_b_b,
                "clinical_significance": "Severe hyperglycemia requiring aggressive treatment",
                "risk_level": "HIGH",
            },
            "C": {
                "name": "Postprandial Response",
                "description": "Post-meal glucose spike pattern (95-220 mg/dL)",
                "y_true": y_true_c,
                "model_a": pred_a_c,
                "model_b": pred_b_c,
                "clinical_significance": "Post-meal glucose management challenges",
                "risk_level": "MODERATE",
            },
            "D": {
                "name": "Dawn Phenomenon",
                "description": "Early morning glucose rise pattern (90-160 mg/dL)",
                "y_true": y_true_d,
                "model_a": pred_a_d,
                "model_b": pred_b_d,
                "clinical_significance": "Natural circadian glucose elevation",
                "risk_level": "LOW",
            },
            "E": {
                "name": "Exercise Response",
                "description": "Exercise-induced glucose changes (60-140 mg/dL)",
                "y_true": y_true_e,
                "model_a": pred_a_e,
                "model_b": pred_b_e,
                "clinical_significance": "Exercise-induced hypoglycemia risk",
                "risk_level": "MODERATE",
            },
            "F": {
                "name": "Measurement Noise",
                "description": "Stable euglycemic with sensor variations (85-140 mg/dL)",
                "y_true": y_true_f,
                "model_a": pred_a_f,
                "model_b": pred_b_f,
                "clinical_significance": "Measurement precision in normal glucose range",
                "risk_level": "LOW",
            },
            "G": {
                "name": "Mixed Clinical",
                "description": "Full glucose range coverage (35-300 mg/dL)",
                "y_true": y_true_g,
                "model_a": pred_a_g,
                "model_b": pred_b_g,
                "clinical_significance": "Mixed critical events testing robustness",
                "risk_level": "CRITICAL",
            },
            "H": {
                "name": "Extreme Cases",
                "description": "Life-threatening glucose extremes (20-450 mg/dL)",
                "y_true": y_true_h,
                "model_a": pred_a_h,
                "model_b": pred_b_h,
                "clinical_significance": "Extreme glucose values requiring emergency care",
                "risk_level": "CRITICAL",
            },
        }

        # Create ClinicalScenario objects
        for scenario_id, data in extended_scenarios.items():
            scenario = ClinicalScenario(
                name=data["name"],
                description=data["description"],
                y_true=np.array(data["y_true"]),
                risk_level=data["risk_level"],
                clinical_significance=data["clinical_significance"],
            )

            # Add model predictions
            scenario.add_model_prediction("Model_A", np.array(data["model_a"]))
            scenario.add_model_prediction("Model_B", np.array(data["model_b"]))

            self.scenarios[scenario_id] = scenario


# Global extended repository instance
extended_clinical_scenarios = ExtendedScenarioRepository()


def get_extended_clinical_scenarios() -> ExtendedScenarioRepository:
    """Get extended clinical scenario repository with 50 data points per scenario"""
    return extended_clinical_scenarios


def verify_model_properties():
    """
    Verify that the generated data has the required properties:
    1. Same absolute error for both models
    2. Model A underestimates below euglycemic, overestimates above
    3. Model B has opposite behavior
    """
    repo = get_extended_clinical_scenarios()

    print("Verifying model prediction properties...")
    print("=" * 60)

    for scenario_id in repo.get_scenario_ids():
        scenario = repo.get_scenario(scenario_id)
        y_true = scenario.y_true
        pred_a = scenario.model_predictions["Model_A"]
        pred_b = scenario.model_predictions["Model_B"]

        # Check absolute errors are equal
        abs_error_a = np.abs(pred_a - y_true)
        abs_error_b = np.abs(pred_b - y_true)
        errors_equal = np.allclose(abs_error_a, abs_error_b, atol=0.01)

        # Check bias patterns
        below_euglycemic = y_true < EUGLYCEMIC_LOW
        above_euglycemic = y_true > EUGLYCEMIC_HIGH

        # Model A: underestimates below, overestimates above
        a_under_below = (
            np.all(pred_a[below_euglycemic] <= y_true[below_euglycemic])
            if np.any(below_euglycemic)
            else True
        )
        a_over_above = (
            np.all(pred_a[above_euglycemic] >= y_true[above_euglycemic])
            if np.any(above_euglycemic)
            else True
        )

        # Model B: overestimates below, underestimates above
        b_over_below = (
            np.all(pred_b[below_euglycemic] >= y_true[below_euglycemic])
            if np.any(below_euglycemic)
            else True
        )
        b_under_above = (
            np.all(pred_b[above_euglycemic] <= y_true[above_euglycemic])
            if np.any(above_euglycemic)
            else True
        )

        print(f"\nScenario {scenario_id}: {scenario.name}")
        print(f"  Data points: {len(y_true)}")
        print(f"  Glucose range: {y_true.min():.1f} - {y_true.max():.1f} mg/dL")
        print(f"  Equal absolute errors: {'PASS' if errors_equal else 'FAIL'}")
        print(
            f"  Model A bias pattern: {'PASS' if (a_under_below and a_over_above) else 'FAIL'}"
        )
        print(
            f"  Model B bias pattern: {'PASS' if (b_over_below and b_under_above) else 'FAIL'}"
        )
        print(f"  Points below euglycemic (<70): {np.sum(below_euglycemic)}")
        print(f"  Points above euglycemic (>180): {np.sum(above_euglycemic)}")
        print(
            f"  Points in euglycemic range: {np.sum(~below_euglycemic & ~above_euglycemic)}"
        )


if __name__ == "__main__":
    verify_model_properties()
