"""
Clinical Data Management

Standardized clinical scenario definitions, MADEX parameters, and data structures
shared between bootstrap and sensitivity analysis modules.

Features:
- Clinical scenario repository with risk level assessments
- MADEX parameter validation and interpretation
- Data validation utilities
- Clinical significance annotations
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class ClinicalScenario:
    """
    Clinical scenario data structure with validation

    Represents a clinical glucose monitoring scenario with reference values,
    model predictions, and clinical context.
    """

    name: str
    description: str
    y_true: np.ndarray
    risk_level: str
    clinical_significance: str
    model_predictions: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self):
        """Validate scenario data after initialization"""
        self.y_true = np.array(self.y_true)
        self.validate_data()

    def validate_data(self):
        """Validate clinical scenario data quality"""
        if len(self.y_true) == 0:
            raise ValueError(f"Scenario {self.name}: Empty reference values")

        if np.any(self.y_true <= 0):
            raise ValueError(
                f"Scenario {self.name}: Non-positive glucose values detected"
            )

        if self.risk_level not in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
            raise ValueError(
                f"Scenario {self.name}: Invalid risk level '{self.risk_level}'"
            )

        # Validate model predictions if provided
        if self.model_predictions:
            for model_name, predictions in self.model_predictions.items():
                predictions = np.array(predictions)
                if len(predictions) != len(self.y_true):
                    raise ValueError(
                        f"Scenario {self.name}, Model {model_name}: "
                        f"Prediction length {len(predictions)} != reference length {len(self.y_true)}"
                    )

    def get_glucose_range(self) -> Tuple[float, float]:
        """Get glucose value range for this scenario"""
        return float(self.y_true.min()), float(self.y_true.max())

    def add_model_prediction(
        self, model_name: str, predictions: Union[List, np.ndarray]
    ):
        """Add model predictions for this scenario"""
        if self.model_predictions is None:
            self.model_predictions = {}

        predictions = np.array(predictions)
        if len(predictions) != len(self.y_true):
            raise ValueError(
                f"Prediction length {len(predictions)} != reference length {len(self.y_true)}"
            )

        self.model_predictions[model_name] = predictions

    def get_clinical_context(self) -> Dict[str, str]:
        """Get clinical context information"""
        min_glucose, max_glucose = self.get_glucose_range()
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "clinical_significance": self.clinical_significance,
            "glucose_range": f"{min_glucose:.1f}-{max_glucose:.1f} mg/dL",
            "data_points": str(len(self.y_true)),
        }


@dataclass
class MADEXParameters:
    """
    MADEX parameter configuration with clinical interpretation.

    This is the legacy parameter class using short names (a, b, c).
    For new code, prefer importing MADEXParameters from new_metric.madex
    which uses descriptive names (center, critical_range, slope).

    Parameters:
    - a: Euglycemic center (target glucose level) - alias for 'center'
    - b: Critical range (sensitivity to deviations from target) - alias for 'critical_range'
    - c: Slope modifier (error scaling factor) - alias for 'slope'

    Note: Default c=100 aligns with the canonical implementation in madex.py.
    """

    a: float = 125.0  # mg/dL (center)
    b: float = 55.0  # sensitivity parameter (critical_range)
    c: float = 100.0  # slope parameter - FIXED: was 40, now 100 to match madex.py

    # Aliases for descriptive names (for compatibility with core MADEXParameters)
    @property
    def center(self) -> float:
        """Alias: euglycemic center point."""
        return self.a

    @property
    def critical_range(self) -> float:
        """Alias: sensitivity parameter."""
        return self.b

    @property
    def slope(self) -> float:
        """Alias: error scaling factor."""
        return self.c

    def __post_init__(self):
        """Validate parameters after initialization"""
        self.validate_parameters()

    def validate_parameters(self):
        """Validate MADEX parameter ranges"""
        if not (50 <= self.a <= 250):
            raise ValueError(
                f"Parameter 'a' ({self.a}) must be between 50-250 mg/dL. "
                f"This represents the target glucose level."
            )

        if not (10 <= self.b <= 150):
            raise ValueError(
                f"Parameter 'b' ({self.b}) must be between 10-150. "
                f"This controls sensitivity to glucose deviations."
            )

        if not (10 <= self.c <= 200):
            raise ValueError(
                f"Parameter 'c' ({self.c}) must be between 10-200. "
                f"This controls error scaling."
            )

    def get_clinical_interpretation(self) -> Dict[str, str]:
        """Get clinical interpretation of parameter values"""
        interpretations = {}

        # Interpret parameter 'a' (euglycemic center)
        if self.a < 100:
            interpretations["a"] = (
                f"Tight glycemic control target ({self.a} mg/dL) - aggressive management"
            )
        elif self.a > 140:
            interpretations["a"] = (
                f"Relaxed glycemic control target ({self.a} mg/dL) - conservative management"
            )
        else:
            interpretations["a"] = (
                f"Standard glycemic control target ({self.a} mg/dL) - normal management"
            )

        # Interpret parameter 'b' (critical range)
        if self.b < 40:
            interpretations["b"] = (
                f"High sensitivity to deviations (b={self.b}) - strict error penalties"
            )
        elif self.b > 70:
            interpretations["b"] = (
                f"Low sensitivity to deviations (b={self.b}) - lenient error penalties"
            )
        else:
            interpretations["b"] = (
                f"Moderate sensitivity to deviations (b={self.b}) - balanced error penalties"
            )

        # Interpret parameter 'c' (slope modifier)
        if self.c < 50:
            interpretations["c"] = (
                f"Steep error scaling (c={self.c}) - harsh penalties for large errors"
            )
        elif self.c > 120:
            interpretations["c"] = (
                f"Gentle error scaling (c={self.c}) - mild penalties for large errors"
            )
        else:
            interpretations["c"] = (
                f"Standard error scaling (c={self.c}) - balanced error penalties"
            )

        return interpretations

    def get_parameter_dict(self) -> Dict[str, float]:
        """Get parameters as dictionary for function calls"""
        return {"a": self.a, "b": self.b, "c": self.c}

    def to_core_params(self):
        """
        Convert to core MADEXParameters from madex.py.

        Returns:
            MADEXParameters instance from new_metric.madex
        """
        from ..madex import MADEXParameters as CoreMADEXParameters

        return CoreMADEXParameters(center=self.a, critical_range=self.b, slope=self.c)


class ScenarioRepository:
    """
    Repository for managing clinical scenarios

    Provides centralized access to clinical scenarios with validation
    and filtering capabilities.
    """

    def __init__(self):
        """Initialize repository with standard clinical scenarios"""
        self.scenarios: Dict[str, ClinicalScenario] = {}
        self._initialize_standard_scenarios()

    def _initialize_standard_scenarios(self):
        """Initialize repository with standard clinical scenarios from literature"""

        # Standard scenarios from bootstrap and sensitivity analysis modules
        standard_scenarios = {
            "A": {
                "name": "Hypoglycemia Detection",
                "description": "Critical low glucose events (45-61 mg/dL)",
                "y_true": [45, 55, 48, 52, 58, 61, 54, 50],
                "model_a": [65, 70, 68, 72, 75, 80, 74, 70],
                "model_b": [35, 45, 38, 42, 48, 51, 44, 40],
                "clinical_significance": "Life-threatening hypoglycemia requiring immediate intervention",
                "risk_level": "CRITICAL",
            },
            "B": {
                "name": "Hyperglycemia Management",
                "description": "Sustained high glucose (275-310 mg/dL)",
                "y_true": [280, 310, 295, 275, 290, 305, 285, 300],
                "model_a": [250, 280, 265, 245, 260, 275, 255, 270],
                "model_b": [320, 350, 335, 315, 330, 345, 325, 340],
                "clinical_significance": "Severe hyperglycemia requiring aggressive treatment",
                "risk_level": "HIGH",
            },
            "C": {
                "name": "Postprandial Response",
                "description": "Meal-induced glucose spikes (115-220 mg/dL)",
                "y_true": [120, 180, 220, 195, 160, 140, 125, 115],
                "model_a": [110, 170, 210, 185, 150, 130, 115, 105],
                "model_b": [130, 190, 230, 205, 170, 150, 135, 125],
                "clinical_significance": "Post-meal glucose management challenges",
                "risk_level": "MODERATE",
            },
            "D": {
                "name": "Dawn Phenomenon",
                "description": "Early morning glucose rise (85-155 mg/dL)",
                "y_true": [85, 95, 110, 125, 140, 155, 150, 145],
                "model_a": [90, 100, 115, 130, 145, 160, 155, 150],
                "model_b": [80, 90, 105, 120, 135, 150, 145, 140],
                "clinical_significance": "Natural circadian glucose elevation",
                "risk_level": "LOW",
            },
            "E": {
                "name": "Exercise Response",
                "description": "Activity-induced glucose drop (65-140 mg/dL)",
                "y_true": [140, 120, 95, 75, 65, 80, 100, 125],
                "model_a": [145, 125, 100, 80, 70, 85, 105, 130],
                "model_b": [135, 115, 90, 70, 60, 75, 95, 120],
                "clinical_significance": "Exercise-induced hypoglycemia risk",
                "risk_level": "MODERATE",
            },
            "F": {
                "name": "Measurement Noise",
                "description": "Sensor accuracy challenges (97-105 mg/dL)",
                "y_true": [100, 105, 98, 102, 99, 103, 101, 97],
                "model_a": [108, 113, 106, 110, 107, 111, 109, 105],
                "model_b": [92, 97, 90, 94, 91, 95, 93, 89],
                "clinical_significance": "Measurement precision in normal glucose range",
                "risk_level": "LOW",
            },
            "G": {
                "name": "Mixed Clinical",
                "description": "Combined challenging scenarios (45-250 mg/dL)",
                "y_true": [65, 180, 45, 250, 95, 200, 55, 160],
                "model_a": [75, 190, 55, 260, 105, 210, 65, 170],
                "model_b": [55, 170, 35, 240, 85, 190, 45, 150],
                "clinical_significance": "Mixed critical events testing robustness",
                "risk_level": "CRITICAL",
            },
            "H": {
                "name": "Extreme Cases",
                "description": "Life-threatening glucose extremes (25-450 mg/dL)",
                "y_true": [25, 35, 400, 450, 30, 420, 40, 380],
                "model_a": [45, 55, 380, 430, 50, 400, 60, 360],
                "model_b": [15, 25, 420, 470, 20, 440, 30, 400],
                "clinical_significance": "Extreme glucose values requiring emergency care",
                "risk_level": "CRITICAL",
            },
        }

        # Create ClinicalScenario objects
        for scenario_id, data in standard_scenarios.items():
            scenario = ClinicalScenario(
                name=data["name"],
                description=data["description"],
                y_true=np.array(data["y_true"]),
                risk_level=data["risk_level"],
                clinical_significance=data["clinical_significance"],
            )

            # Add model predictions
            scenario.add_model_prediction("Model_A", data["model_a"])
            scenario.add_model_prediction("Model_B", data["model_b"])

            self.scenarios[scenario_id] = scenario

    def get_scenario(self, scenario_id: str) -> ClinicalScenario:
        """Get scenario by ID"""
        if scenario_id not in self.scenarios:
            raise KeyError(f"Scenario '{scenario_id}' not found")
        return self.scenarios[scenario_id]

    def get_all_scenarios(self) -> Dict[str, ClinicalScenario]:
        """Get all scenarios"""
        return self.scenarios.copy()

    def get_scenarios_by_risk_level(
        self, risk_level: str
    ) -> Dict[str, ClinicalScenario]:
        """Get scenarios filtered by risk level"""
        return {
            scenario_id: scenario
            for scenario_id, scenario in self.scenarios.items()
            if scenario.risk_level == risk_level
        }

    def get_scenario_ids(self) -> List[str]:
        """Get list of scenario IDs"""
        return list(self.scenarios.keys())

    def add_scenario(self, scenario_id: str, scenario: ClinicalScenario):
        """Add new scenario to repository"""
        scenario.validate_data()
        self.scenarios[scenario_id] = scenario

    def validate_all_scenarios(self) -> bool:
        """Validate all scenarios in repository"""
        try:
            for scenario in self.scenarios.values():
                scenario.validate_data()
            return True
        except ValueError as e:
            print(f"Validation error: {e}")
            return False

    def get_reference_values_dict(self) -> Dict[str, np.ndarray]:
        """Get reference values as dictionary (backward compatibility)"""
        return {
            scenario_id: scenario.y_true
            for scenario_id, scenario in self.scenarios.items()
        }

    def get_model_predictions_dict(self, model_name: str) -> Dict[str, np.ndarray]:
        """Get model predictions as dictionary (backward compatibility)"""
        result = {}
        for scenario_id, scenario in self.scenarios.items():
            if scenario.model_predictions and model_name in scenario.model_predictions:
                result[scenario_id] = scenario.model_predictions[model_name]
        return result

    def get_scenario_descriptions_dict(self) -> Dict[str, Dict[str, str]]:
        """Get scenario descriptions as dictionary (backward compatibility)"""
        return {
            scenario_id: scenario.get_clinical_context()
            for scenario_id, scenario in self.scenarios.items()
        }


# Global repository instance for shared access
clinical_scenarios = ScenarioRepository()

# Standard MADEX parameters (aligned with madex.py defaults: slope=100)
standard_madex_params = MADEXParameters(a=125.0, b=55.0, c=100.0)

# Clinical parameter sets for different contexts
# Note: c (slope) values adjusted to align with canonical defaults
clinical_parameter_sets = {
    "Standard Adult Range": MADEXParameters(a=125, b=55, c=100),
    "Tight Glycemic Control": MADEXParameters(a=110, b=40, c=80),
    "Pediatric Range": MADEXParameters(a=140, b=70, c=100),
    "Elderly/Relaxed Control": MADEXParameters(a=140, b=80, c=120),
}


def get_clinical_scenarios() -> ScenarioRepository:
    """Get global clinical scenario repository"""
    return clinical_scenarios


def get_standard_madex_params() -> MADEXParameters:
    """Get standard MADEX parameters"""
    return standard_madex_params


def get_clinical_parameter_sets() -> Dict[str, MADEXParameters]:
    """Get predefined clinical parameter sets"""
    return clinical_parameter_sets.copy()


# =============================================================================
# Extended Clinical Data with 50 Data Points Per Scenario
# =============================================================================

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
        y_true_a = np.concatenate(
            [
                np.linspace(85, 55, 10),
                np.linspace(55, 40, 10),
                np.array([38, 35, 33, 32, 30, 32, 35, 38, 42, 45]),
                np.linspace(45, 60, 10),
                np.linspace(60, 75, 10),
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
        y_true_b = np.concatenate(
            [
                np.linspace(150, 200, 10),
                np.linspace(200, 280, 10),
                np.array([290, 300, 310, 320, 325, 320, 310, 300, 285, 270]),
                np.linspace(270, 220, 10),
                np.linspace(220, 170, 10),
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
        y_true_c = np.concatenate(
            [
                np.linspace(95, 110, 8),
                np.linspace(110, 180, 10),
                np.linspace(180, 220, 8),
                np.linspace(220, 160, 12),
                np.linspace(160, 105, 12),
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
        y_true_d = np.concatenate(
            [
                np.linspace(110, 100, 10),
                np.array([98, 95, 93, 92, 90, 90, 92, 95, 100, 105]),
                np.linspace(105, 130, 10),
                np.linspace(130, 160, 10),
                np.linspace(160, 140, 10),
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
        y_true_e = np.concatenate(
            [
                np.linspace(140, 130, 8),
                np.linspace(130, 85, 12),
                np.linspace(85, 60, 10),
                np.linspace(60, 90, 10),
                np.linspace(90, 120, 10),
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
        base_f = 110 + 15 * np.sin(np.linspace(0, 4 * np.pi, 50))
        noise_f = np.random.RandomState(6).normal(0, 5, 50)
        y_true_f = np.clip(base_f + noise_f, 85, 140)
        error_f = 5 + 3 * np.abs(np.sin(np.linspace(0, 6 * np.pi, 50)))
        pred_a_f, pred_b_f = generate_model_predictions(y_true_f, error_f, seed=6)

        # Scenario G: Mixed Clinical (Full Range)
        y_true_g = np.concatenate(
            [
                np.linspace(100, 50, 8),
                np.linspace(50, 35, 5),
                np.linspace(35, 80, 7),
                np.linspace(80, 150, 8),
                np.linspace(150, 250, 8),
                np.linspace(250, 300, 6),
                np.linspace(300, 180, 8),
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
        y_true_h = np.concatenate(
            [
                np.linspace(80, 40, 8),
                np.array([35, 30, 25, 22, 20, 22, 25, 30, 40, 55]),
                np.linspace(55, 120, 8),
                np.linspace(120, 300, 8),
                np.array([320, 350, 380, 400, 420, 440, 450, 440, 420, 380]),
                np.linspace(380, 250, 6),
            ]
        )
        error_h = np.concatenate(
            [
                np.linspace(10, 15, 8),
                np.array([12, 10, 8, 7, 6, 7, 8, 10, 14, 18]),
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
