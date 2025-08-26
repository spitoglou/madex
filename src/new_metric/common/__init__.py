"""
Common components for new_metric analysis modules

This package contains shared infrastructure used by both bootstrap and sensitivity
analysis modules, including logging systems, clinical data structures, statistical
utilities, and analysis frameworks.

Components:
- clinical_logging: LLM-friendly narrative logging system
- clinical_data: Clinical scenario definitions and MADEX parameters  
- statistical_utils: Common statistical calculation functions
- analysis_framework: Base classes for analysis workflows
"""

__version__ = "0.1.0"

# Import common components for easy access
try:
    from .clinical_logging import ClinicalLogger
    from .clinical_data import ClinicalScenario, ScenarioRepository, MADEXParameters
    from .statistical_utils import MetricCalculator, BootstrapAnalyzer, SignificanceTester, ClinicalDetection
    from .analysis_framework import AnalysisFramework, ClinicalAnalysis, StatisticalAnalysis
except ImportError:
    # Handle import issues during development
    pass

__all__ = [
    "ClinicalLogger",
    "ClinicalScenario", 
    "ScenarioRepository",
    "MADEXParameters",
    "MetricCalculator",
    "BootstrapAnalyzer", 
    "SignificanceTester",
    "ClinicalDetection",
    "AnalysisFramework",
    "ClinicalAnalysis", 
    "StatisticalAnalysis",
    "get_clinical_scenarios",
    "get_standard_madex_params",
    "get_clinical_parameter_sets"
]

# Import convenience functions
try:
    from .clinical_data import get_clinical_scenarios, get_standard_madex_params, get_clinical_parameter_sets
except ImportError:
    pass