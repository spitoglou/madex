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

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any


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
            raise ValueError(f"Scenario {self.name}: Non-positive glucose values detected")
        
        if self.risk_level not in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW']:
            raise ValueError(f"Scenario {self.name}: Invalid risk level '{self.risk_level}'")
        
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
    
    def add_model_prediction(self, model_name: str, predictions: Union[List, np.ndarray]):
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
            'name': self.name,
            'description': self.description,
            'risk_level': self.risk_level,
            'clinical_significance': self.clinical_significance,
            'glucose_range': f"{min_glucose:.1f}-{max_glucose:.1f} mg/dL",
            'data_points': str(len(self.y_true))
        }


@dataclass
class MADEXParameters:
    """
    MADEX parameter configuration with clinical interpretation
    
    Parameters:
    - a: Euglycemic center (target glucose level)
    - b: Critical range (sensitivity to deviations from target)  
    - c: Slope modifier (error scaling factor)
    """
    
    a: float = 125.0  # mg/dL
    b: float = 55.0   # sensitivity parameter
    c: float = 40.0   # slope parameter
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        self.validate_parameters()
    
    def validate_parameters(self):
        """Validate MADEX parameter ranges"""
        if not (80 <= self.a <= 180):
            raise ValueError(f"Parameter 'a' ({self.a}) must be between 80-180 mg/dL")
        
        if not (20 <= self.b <= 100):
            raise ValueError(f"Parameter 'b' ({self.b}) must be between 20-100")
        
        if not (10 <= self.c <= 100):
            raise ValueError(f"Parameter 'c' ({self.c}) must be between 10-100")
    
    def get_clinical_interpretation(self) -> Dict[str, str]:
        """Get clinical interpretation of parameter values"""
        interpretations = {}
        
        # Interpret parameter 'a' (euglycemic center)
        if self.a < 100:
            interpretations['a'] = f"Tight glycemic control target ({self.a} mg/dL) - aggressive management"
        elif self.a > 140:
            interpretations['a'] = f"Relaxed glycemic control target ({self.a} mg/dL) - conservative management"
        else:
            interpretations['a'] = f"Standard glycemic control target ({self.a} mg/dL) - normal management"
        
        # Interpret parameter 'b' (critical range)
        if self.b < 50:
            interpretations['b'] = f"High sensitivity to deviations (b={self.b}) - strict error penalties"
        elif self.b > 70:
            interpretations['b'] = f"Low sensitivity to deviations (b={self.b}) - lenient error penalties"
        else:
            interpretations['b'] = f"Moderate sensitivity to deviations (b={self.b}) - balanced error penalties"
        
        # Interpret parameter 'c' (slope modifier)
        if self.c < 30:
            interpretations['c'] = f"Steep error scaling (c={self.c}) - harsh penalties for large errors"
        elif self.c > 50:
            interpretations['c'] = f"Gentle error scaling (c={self.c}) - mild penalties for large errors"
        else:
            interpretations['c'] = f"Standard error scaling (c={self.c}) - balanced error penalties"
        
        return interpretations
    
    def get_parameter_dict(self) -> Dict[str, float]:
        """Get parameters as dictionary for function calls"""
        return {'a': self.a, 'b': self.b, 'c': self.c}


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
            'A': {
                'name': 'Hypoglycemia Detection',
                'description': 'Critical low glucose events (45-61 mg/dL)',
                'y_true': [45, 55, 48, 52, 58, 61, 54, 50],
                'model_a': [65, 70, 68, 72, 75, 80, 74, 70],
                'model_b': [35, 45, 38, 42, 48, 51, 44, 40],
                'clinical_significance': 'Life-threatening hypoglycemia requiring immediate intervention',
                'risk_level': 'CRITICAL'
            },
            'B': {
                'name': 'Hyperglycemia Management',
                'description': 'Sustained high glucose (275-310 mg/dL)',
                'y_true': [280, 310, 295, 275, 290, 305, 285, 300],
                'model_a': [250, 280, 265, 245, 260, 275, 255, 270],
                'model_b': [320, 350, 335, 315, 330, 345, 325, 340],
                'clinical_significance': 'Severe hyperglycemia requiring aggressive treatment',
                'risk_level': 'HIGH'
            },
            'C': {
                'name': 'Postprandial Response',
                'description': 'Meal-induced glucose spikes (115-220 mg/dL)',
                'y_true': [120, 180, 220, 195, 160, 140, 125, 115],
                'model_a': [110, 170, 210, 185, 150, 130, 115, 105],
                'model_b': [130, 190, 230, 205, 170, 150, 135, 125],
                'clinical_significance': 'Post-meal glucose management challenges',
                'risk_level': 'MODERATE'
            },
            'D': {
                'name': 'Dawn Phenomenon',
                'description': 'Early morning glucose rise (85-155 mg/dL)',
                'y_true': [85, 95, 110, 125, 140, 155, 150, 145],
                'model_a': [90, 100, 115, 130, 145, 160, 155, 150],
                'model_b': [80, 90, 105, 120, 135, 150, 145, 140],
                'clinical_significance': 'Natural circadian glucose elevation',
                'risk_level': 'LOW'
            },
            'E': {
                'name': 'Exercise Response',
                'description': 'Activity-induced glucose drop (65-140 mg/dL)',
                'y_true': [140, 120, 95, 75, 65, 80, 100, 125],
                'model_a': [145, 125, 100, 80, 70, 85, 105, 130],
                'model_b': [135, 115, 90, 70, 60, 75, 95, 120],
                'clinical_significance': 'Exercise-induced hypoglycemia risk',
                'risk_level': 'MODERATE'
            },
            'F': {
                'name': 'Measurement Noise',
                'description': 'Sensor accuracy challenges (97-105 mg/dL)',
                'y_true': [100, 105, 98, 102, 99, 103, 101, 97],
                'model_a': [108, 113, 106, 110, 107, 111, 109, 105],
                'model_b': [92, 97, 90, 94, 91, 95, 93, 89],
                'clinical_significance': 'Measurement precision in normal glucose range',
                'risk_level': 'LOW'
            },
            'G': {
                'name': 'Mixed Clinical',
                'description': 'Combined challenging scenarios (45-250 mg/dL)',
                'y_true': [65, 180, 45, 250, 95, 200, 55, 160],
                'model_a': [75, 190, 55, 260, 105, 210, 65, 170],
                'model_b': [55, 170, 35, 240, 85, 190, 45, 150],
                'clinical_significance': 'Mixed critical events testing robustness',
                'risk_level': 'CRITICAL'
            },
            'H': {
                'name': 'Extreme Cases',
                'description': 'Life-threatening glucose extremes (25-450 mg/dL)',
                'y_true': [25, 35, 400, 450, 30, 420, 40, 380],
                'model_a': [45, 55, 380, 430, 50, 400, 60, 360],
                'model_b': [15, 25, 420, 470, 20, 440, 30, 400],
                'clinical_significance': 'Extreme glucose values requiring emergency care',
                'risk_level': 'CRITICAL'
            }
        }
        
        # Create ClinicalScenario objects
        for scenario_id, data in standard_scenarios.items():
            scenario = ClinicalScenario(
                name=data['name'],
                description=data['description'],
                y_true=np.array(data['y_true']),
                risk_level=data['risk_level'],
                clinical_significance=data['clinical_significance']
            )
            
            # Add model predictions
            scenario.add_model_prediction('Model_A', data['model_a'])
            scenario.add_model_prediction('Model_B', data['model_b'])
            
            self.scenarios[scenario_id] = scenario
    
    def get_scenario(self, scenario_id: str) -> ClinicalScenario:
        """Get scenario by ID"""
        if scenario_id not in self.scenarios:
            raise KeyError(f"Scenario '{scenario_id}' not found")
        return self.scenarios[scenario_id]
    
    def get_all_scenarios(self) -> Dict[str, ClinicalScenario]:
        """Get all scenarios"""
        return self.scenarios.copy()
    
    def get_scenarios_by_risk_level(self, risk_level: str) -> Dict[str, ClinicalScenario]:
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

# Standard MADEX parameters  
standard_madex_params = MADEXParameters(a=125.0, b=55.0, c=40.0)

# Clinical parameter sets for different contexts
clinical_parameter_sets = {
    'Standard Adult Range': MADEXParameters(a=125, b=55, c=40),
    'Tight Glycemic Control': MADEXParameters(a=110, b=40, c=30),
    'Pediatric Range': MADEXParameters(a=140, b=70, c=50),
    'Elderly/Relaxed Control': MADEXParameters(a=140, b=80, c=60)
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