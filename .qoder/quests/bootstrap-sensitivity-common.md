# Bootstrap-Sensitivity Common Components Refactoring Design

## Overview

This design document outlines the refactoring of common functionality between the bootstrap and sensitivity analysis modules into shared components. The analysis reveals significant code duplication in logging infrastructure, clinical scenario management, statistical calculations, and data validation patterns.

## Architecture

### Current State Analysis

Both modules implement similar patterns:

```mermaid
graph TD
    subgraph "Current Bootstrap Module"
        BA[bootstrap.py] --> BL[Logging Setup]
        BA --> BD[Data Definitions]
        BA --> BS[Statistical Functions]
        BA --> BC[Clinical Analysis]
    end
    
    subgraph "Current Sensitivity Module"
        SA[sensitivity_analysis.py] --> SL[Logging Setup]
        SA --> SD[Data Definitions]
        SA --> SS[Statistical Functions]
        SA --> SC[Clinical Analysis]
    end
    
    subgraph "Duplication Issues"
        BL -.-> SL
        BD -.-> SD
        BS -.-> SS
        BC -.-> SC
    end
```

### Target Architecture

```mermaid
graph TD
    subgraph "Shared Analysis Framework"
        CF[common/analysis_framework.py]
        CL[common/clinical_logging.py]
        CD[common/clinical_data.py]
        CS[common/statistical_utils.py]
        CV[common/visualization_utils.py]
    end
    
    subgraph "Specialized Modules"
        BA[bootstrap/bootstrap_analysis.py] --> CF
        BA --> CL
        BA --> CD
        BA --> CS
        
        SA[sensitivity/sensitivity_analysis.py] --> CF
        SA --> CL
        SA --> CD
        SA --> CS
        
        BA --> CV
        SA --> CV
    end
```

## Component Architecture

### 1. Common Analysis Framework (`common/analysis_framework.py`)

Core base classes and interfaces for statistical analysis:

```mermaid
classDiagram
    class AnalysisFramework {
        +script_dir: Path
        +logger: Logger
        +narrative_logger: Logger
        +results: Dict
        +setup_logging()
        +run_analysis()
        +generate_report()
    }
    
    class ClinicalAnalysis {
        +scenarios: Dict
        +madex_params: Dict
        +analyze_scenario()
        +interpret_results()
    }
    
    class StatisticalAnalysis {
        +bootstrap_metric()
        +calculate_significance()
        +extract_wilcoxon_results()
        +cohens_d()
    }
    
    AnalysisFramework <|-- ClinicalAnalysis
    AnalysisFramework <|-- StatisticalAnalysis
```

### 2. Clinical Logging System (`common/clinical_logging.py`)

Unified LLM-friendly narrative logging infrastructure:

```mermaid
classDiagram
    class ClinicalLogger {
        +narrative_logger: Logger
        +technical_logger: Logger
        +script_dir: Path
        +setup_loggers(module_name)
        +log_narrative(message)
        +log_section_header(title)
        +log_clinical_context(scenario, context)
        +log_progress(step, total, description)
    }
    
    class LogFormatter {
        +format_clinical_significance(level, message)
        +format_statistical_result(metric, value, interpretation)
        +format_parameter_interpretation(param, value, context)
        +ensure_ascii_safe(text)
    }
    
    ClinicalLogger --> LogFormatter
```

### 3. Clinical Data Management (`common/clinical_data.py`)

Standardized clinical scenario definitions and data structures:

```mermaid
classDiagram
    class ClinicalScenario {
        +name: str
        +description: str
        +y_true: Array
        +risk_level: str
        +clinical_significance: str
        +get_glucose_range()
        +validate_data()
    }
    
    class ScenarioRepository {
        +scenarios: Dict[str, ClinicalScenario]
        +get_scenario(name)
        +get_all_scenarios()
        +get_scenarios_by_risk_level(level)
        +validate_all_scenarios()
    }
    
    class MADEXParameters {
        +a: float
        +b: float
        +c: float
        +validate_parameters()
        +get_clinical_interpretation()
    }
    
    ScenarioRepository --> ClinicalScenario
```

### 4. Statistical Utilities (`common/statistical_utils.py`)

Common statistical calculation functions:

```mermaid
classDiagram
    class MetricCalculator {
        +compute_madex(y_true, y_pred, params)
        +compute_rmse(y_true, y_pred)
        +compute_mae(y_true, y_pred)
        +compute_mape(y_true, y_pred)
    }
    
    class BootstrapAnalyzer {
        +bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap, ci)
        +bootstrap_confidence_interval(values, ci_level)
        +check_ci_overlap(ci1, ci2)
    }
    
    class SignificanceTester {
        +extract_wilcoxon_results(result)
        +cohens_d(x, y)
        +interpret_p_value(p_value)
        +interpret_effect_size(effect_size)
    }
    
    class ClinicalDetection {
        +sensitivity_hypoglycemia(y_true, y_pred, threshold)
        +sensitivity_hyperglycemia(y_true, y_pred, threshold)
        +clinical_risk_score(zone_counts)
    }
```

### 5. Visualization Utilities (`common/visualization_utils.py`)

Shared plotting and data presentation functions:

```mermaid
classDiagram
    class PlotGenerator {
        +create_comparison_plot(results, title)
        +create_sensitivity_heatmap(data, params)
        +create_confidence_interval_plot(data)
    }
    
    class TableFormatter {
        +format_results_table(results, columns)
        +format_statistical_summary(stats)
        +format_clinical_summary(scenarios)
    }
```

## Data Models & Validation

### Clinical Scenario Structure

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| name | str | Scenario identifier | Required, unique |
| description | str | Brief clinical description | Required |
| y_true | Array | Reference glucose values | Required, positive values |
| risk_level | str | Clinical risk assessment | One of: CRITICAL, HIGH, MODERATE, LOW |
| clinical_significance | str | Detailed clinical context | Required |
| model_predictions | Dict | Model prediction arrays | Optional, same length as y_true |

### MADEX Parameters Structure

| Parameter | Type | Range | Clinical Meaning |
|-----------|------|-------|------------------|
| a | float | 80-180 mg/dL | Euglycemic center (target glucose) |
| b | float | 20-100 | Critical range (sensitivity to deviations) |
| c | float | 10-100 | Slope modifier (error scaling factor) |

## Business Logic Layer

### 1. Scenario Analysis Engine

```mermaid
flowchart TD
    A[Load Clinical Scenarios] --> B[Validate Data Quality]
    B --> C[Initialize MADEX Parameters]
    C --> D[For Each Scenario]
    D --> E[Calculate Metrics]
    E --> F[Perform Statistical Tests]
    F --> G[Generate Clinical Interpretation]
    G --> H[Log Narrative Results]
    H --> I{More Scenarios?}
    I -->|Yes| D
    I -->|No| J[Aggregate Results]
    J --> K[Generate Summary Report]
```

### 2. Statistical Validation Pipeline

```mermaid
flowchart TD
    A[Input Data] --> B[Bootstrap Sampling]
    B --> C[Calculate Confidence Intervals]
    C --> D[Perform Significance Tests]
    D --> E[Calculate Effect Sizes]
    E --> F[Interpret Statistical Results]
    F --> G[Generate Clinical Recommendations]
```

### 3. Parameter Sensitivity Analysis

```mermaid
flowchart TD
    A[Define Parameter Ranges] --> B[Generate Parameter Combinations]
    B --> C[For Each Combination]
    C --> D[Calculate MADEX Across Scenarios]
    D --> E[Compare Rankings with Baseline]
    E --> F[Calculate Ranking Consistency]
    F --> G{More Combinations?}
    G -->|Yes| C
    G -->|No| H[Analyze Parameter Effects]
    H --> I[Identify Stable Configurations]
    I --> J[Generate Recommendations]
```

## Refactoring Implementation Plan

### Phase 1: Core Infrastructure

1. **Create common directory structure**
   - `src/new_metric/common/`
   - Move shared utilities to common package

2. **Extract logging infrastructure**
   - Consolidate dual logging system (technical + narrative)
   - Standardize LLM-friendly formatting
   - Implement ASCII-safe encoding

3. **Standardize clinical data definitions**
   - Create unified scenario repository
   - Implement data validation
   - Define standard data structures

### Phase 2: Statistical Functions

1. **Extract common statistical functions**
   - MADEX calculation with logging
   - Traditional metrics (RMSE, MAE, MAPE)
   - Bootstrap analysis utilities
   - Significance testing helpers

2. **Create clinical interpretation layer**
   - Parameter meaning explanations
   - Risk level assessments
   - Clinical detection functions

### Phase 3: Analysis Framework

1. **Develop base analysis classes**
   - Common analysis workflow
   - Results aggregation patterns
   - Report generation framework

2. **Implement specialized analyses**
   - Bootstrap confidence interval analysis
   - Parameter sensitivity testing
   - Comparative metric evaluation

### Phase 4: Module Refactoring

1. **Refactor bootstrap module**
   - Use common infrastructure
   - Reduce code duplication
   - Maintain existing functionality

2. **Refactor sensitivity module**
   - Use shared components
   - Preserve analysis logic
   - Ensure result consistency

## Testing Strategy

### Validation Approach

1. **Functional Equivalence Testing**
   - Compare outputs before/after refactoring
   - Verify numerical accuracy within tolerance
   - Validate statistical test results

2. **Clinical Scenario Validation**
   - Test with original data sets
   - Verify clinical interpretations
   - Check ranking consistency

3. **Logging System Testing**
   - Compare log outputs
   - Verify LLM-friendly formatting
   - Test cross-platform compatibility

### Test Cases

| Test Category | Description | Acceptance Criteria |
|---------------|-------------|-------------------|
| Numerical Accuracy | MADEX calculations match exactly | Difference < 1e-10 |
| Statistical Tests | Wilcoxon results identical | p-values match to 6 decimals |
| Clinical Rankings | Model rankings unchanged | 100% ranking consistency |
| Log Content | Narrative logs equivalent | Content semantically identical |
| Performance | Execution time comparable | <10% performance degradation |

## Migration Strategy

### Backward Compatibility

1. **Preserve existing interfaces**
   - Maintain function signatures
   - Keep module entry points
   - Ensure script executability

2. **Gradual migration approach**
   - Phase-by-phase implementation
   - Parallel execution during transition
   - Rollback capability

### Risk Mitigation

1. **Version control strategy**
   - Feature branches for each phase
   - Comprehensive testing before merge
   - Rollback procedures documented

2. **Data validation**
   - Automated comparison testing
   - Statistical validation checks
   - Clinical outcome verification

## Performance Considerations

### Optimization Opportunities

1. **Shared computation caching**
   - Cache MADEX calculations
   - Reuse bootstrap samples
   - Optimize parameter combinations

2. **Memory efficiency**
   - Streaming data processing
   - Efficient numpy operations
   - Reduced object creation

### Monitoring

1. **Performance metrics**
   - Execution time tracking
   - Memory usage monitoring
   - Statistical accuracy validation

2. **Clinical validation**
   - Result consistency checking
   - Ranking stability monitoring
   - Clinical interpretation accuracy

## File Structure Changes

### New Common Package Structure

```
src/new_metric/common/
├── __init__.py
├── analysis_framework.py      # Base analysis classes
├── clinical_logging.py        # LLM-friendly logging system
├── clinical_data.py          # Scenario and parameter definitions
├── statistical_utils.py      # Common statistical functions
└── visualization_utils.py    # Shared plotting utilities
```

### Updated Module Structure

```
bootstrap/
├── __init__.py
├── bootstrap_analysis.py     # Refactored using common components
└── README.md

sensitivity/
├── __init__.py
├── sensitivity_analysis.py   # Refactored using common components
├── narrative_demo.py
└── README.md
```

### Import Path Changes

Before refactoring:
- Direct implementation in each module
- Duplicated logging setup code
- Copied statistical functions

After refactoring:
```python
from new_metric.common.analysis_framework import ClinicalAnalysis
from new_metric.common.clinical_logging import ClinicalLogger
from new_metric.common.statistical_utils import MetricCalculator
```

## Validation Checklist

- [ ] All original test cases pass
- [ ] MADEX calculations numerically identical
- [ ] Statistical test results unchanged
- [ ] Clinical rankings preserved
- [ ] Log outputs semantically equivalent
- [ ] Performance within acceptable range
- [ ] Documentation updated
- [ ] Common package properly structured
- [ ] Import paths working correctly
- [ ] Cross-platform compatibility verified