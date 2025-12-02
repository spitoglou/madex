"""
Clinical Logging System

Unified LLM-friendly narrative logging infrastructure for clinical analysis modules.
Provides dual logging system with technical details and narrative explanations
optimized for LLM consumption.

Features:
- ASCII-safe encoding for cross-platform compatibility  
- Structured narrative sections with clear headers
- Clinical significance annotations
- Progress tracking with contextual explanations
- Parameter interpretations with medical context
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class ClinicalLogger:
    """
    Unified clinical logging system with LLM-friendly narrative output
    
    Provides dual logging system:
    1. Technical log: Detailed analysis with timestamps
    2. Narrative log: Clean, structured output optimized for LLM consumption
    """
    
    def __init__(self, module_name: str, script_dir: Optional[Path] = None):
        """
        Initialize clinical logging system
        
        Args:
            module_name: Name of the analysis module (e.g., 'bootstrap', 'sensitivity')
            script_dir: Directory where logs should be created (defaults to caller's directory)
        """
        self.module_name = module_name
        self.script_dir = script_dir or Path.cwd()
        
        # Configure warnings
        warnings.filterwarnings('ignore')
        
        # Initialize loggers
        self.logger = self._setup_technical_logger()
        self.narrative_logger = self._setup_narrative_logger()
        
    def _setup_technical_logger(self) -> logging.Logger:
        """Setup technical logging with timestamps and detailed format"""
        log_file = self.script_dir / f'{self.module_name}_analysis_narrative.log'
        
        # Configure basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(f'{self.module_name}_technical')
    
    def _setup_narrative_logger(self) -> logging.Logger:
        """Setup LLM-friendly narrative logging"""
        log_file = self.script_dir / f'{self.module_name}_llm_narrative.log'
        
        # Create narrative logger
        narrative_logger = logging.getLogger(f'{self.module_name}_narrative')
        narrative_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        narrative_handler.setLevel(logging.INFO)
        narrative_formatter = logging.Formatter('%(message)s')
        narrative_handler.setFormatter(narrative_formatter)
        narrative_logger.addHandler(narrative_handler)
        narrative_logger.setLevel(logging.INFO)
        
        return narrative_logger
    
    def log_narrative(self, message: str):
        """
        Log narrative message for LLM consumption
        
        Args:
            message: Message to log with narrative context
        """
        # Ensure ASCII-safe encoding for cross-platform compatibility
        safe_message = self._ensure_ascii_safe(message)
        self.narrative_logger.info(safe_message)
        print(f"[NARRATIVE] {safe_message}")
    
    def log_section_header(self, title: str, level: int = 1):
        """
        Log structured section header
        
        Args:
            title: Section title
            level: Header level (1=main, 2=sub, 3=detail)
        """
        if level == 1:
            separator = "=" * 100
            self.log_narrative(f"\n{separator}")
            self.log_narrative(title.upper())
            self.log_narrative(separator)
        elif level == 2:
            separator = "=" * 80
            self.log_narrative(f"\n{separator}")
            self.log_narrative(title.upper())
            self.log_narrative(separator)
        else:
            separator = "-" * 50
            self.log_narrative(f"\n{separator}")
            self.log_narrative(title)
            self.log_narrative(separator)
    
    def log_analysis_start(self, analysis_type: str, clinical_context: str):
        """Log analysis initiation with context"""
        self.log_section_header(f"{analysis_type} ANALYSIS", level=1)
        self.log_narrative(f"Analysis initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_narrative(f"Analysis type: {analysis_type}")
        self.log_narrative(f"Clinical context: {clinical_context}")
    
    def log_clinical_context(self, scenario_name: str, context: dict):
        """
        Log clinical scenario context with significance
        
        Args:
            scenario_name: Name of clinical scenario
            context: Dictionary with clinical context information
        """
        self.log_narrative(f"\n{scenario_name}: {context.get('name', 'Unknown')}")
        self.log_narrative(f"Description: {context.get('description', 'No description')}")
        self.log_narrative(f"Risk Level: {context.get('risk_level', 'UNKNOWN')}")
        self.log_narrative(f"Clinical Context: {context.get('clinical_significance', 'No significance noted')}")
    
    def log_progress(self, step: int, total: int, description: str):
        """
        Log progress with contextual explanation
        
        Args:
            step: Current step number
            total: Total number of steps
            description: Description of current step
        """
        self.log_narrative(f"\n[{step}/{total}] {description}")
    
    def log_clinical_assessment(self, level: str, message: str):
        """
        Log clinical assessment with appropriate formatting
        
        Args:
            level: Assessment level (SUCCESS, WARNING, CAUTION, etc.)
            message: Assessment message
        """
        # Use ASCII-safe symbols
        symbol_map = {
            'SUCCESS': '[SUCCESS]',
            'WARNING': '[WARNING]', 
            'CAUTION': '[CAUTION]',
            'VERIFIED': '[VERIFIED]',
            'MISSING': '[MISSING]',
            'OK': '[OK]'
        }
        
        symbol = symbol_map.get(level.upper(), f'[{level.upper()}]')
        self.log_narrative(f"  {symbol} {message}")
    
    def log_statistical_result(self, metric: str, value: float, interpretation: str, 
                              confidence_interval: Optional[tuple] = None):
        """
        Log statistical result with clinical interpretation
        
        Args:
            metric: Name of statistical metric
            value: Calculated value
            interpretation: Clinical interpretation
            confidence_interval: Optional confidence interval tuple (lower, upper)
        """
        if confidence_interval:
            lower, upper = confidence_interval
            self.log_narrative(f"{metric}: {value:.3f} ({lower:.3f}-{upper:.3f})")
        else:
            self.log_narrative(f"{metric}: {value:.3f}")
        
        self.log_narrative(f"  Interpretation: {interpretation}")
    
    def log_parameter_interpretation(self, param: str, value: float, context: str):
        """
        Log parameter interpretation with clinical context
        
        Args:
            param: Parameter name
            value: Parameter value
            context: Clinical context/meaning
        """
        self.log_narrative(f"Parameter '{param}'={value} indicates {context}")
    
    def log_comparison_result(self, model_a: str, model_b: str, metric: str, 
                             winner: str, clinical_implication: str):
        """
        Log model comparison result with clinical implications
        
        Args:
            model_a: Name of first model
            model_b: Name of second model
            metric: Metric used for comparison
            winner: Winning model
            clinical_implication: Clinical significance of the result
        """
        self.log_narrative(f"\n{metric} Comparison: {model_a} vs {model_b}")
        self.log_narrative(f"Winner: {winner}")
        self.log_narrative(f"Clinical Implication: {clinical_implication}")
    
    def log_analysis_complete(self, summary: str):
        """Log analysis completion with summary"""
        self.log_section_header("ANALYSIS COMPLETE", level=1)
        self.log_narrative(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_narrative(f"Summary: {summary}")
        self.log_narrative(f"Results logged to {self.module_name}_llm_narrative.log for LLM consumption.")
    
    def _ensure_ascii_safe(self, text: str) -> str:
        """
        Ensure text uses ASCII-safe symbols for cross-platform compatibility
        
        Args:
            text: Input text that may contain Unicode symbols
            
        Returns:
            ASCII-safe version of the text
        """
        # Replace common Unicode symbols with ASCII equivalents
        replacements = {
            '✓': '[VERIFIED]',
            '⚠️': '[WARNING]', 
            '±': '+/-',
            '✗': '[MISSING]',
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v'
        }
        
        result = text
        for unicode_char, ascii_replacement in replacements.items():
            result = result.replace(unicode_char, ascii_replacement)
        
        return result


# Convenience function for backward compatibility
def setup_clinical_logging(module_name: str, script_dir: Optional[Path] = None) -> ClinicalLogger:
    """
    Setup clinical logging system
    
    Args:
        module_name: Name of the analysis module
        script_dir: Directory for log files
        
    Returns:
        Configured ClinicalLogger instance
    """
    return ClinicalLogger(module_name, script_dir)


# Legacy function for existing code compatibility
def log_narrative(message: str):
    """
    Legacy function for backward compatibility.
    This should be replaced with ClinicalLogger.log_narrative() in refactored code.
    """
    print(f"[NARRATIVE] {message}")