# Design: Core Metric API Refactoring

## Context

The MADEX (Mean Adjusted Exponent Error) library has evolved with two parallel implementations that have diverged in parameter naming and default values. This creates a confusing API and potential for subtle bugs when users mix different parts of the library.

**Stakeholders**: Library users, clinical researchers using MADEX for glucose prediction evaluation

**Constraints**:
- Must maintain backward compatibility for existing `mean_adjusted_exponent_error()` function
- Clinical validity of default parameters must be verified
- Performance should not regress

## Goals / Non-Goals

### Goals
- Single source of truth for MADEX calculation
- Consistent parameter naming across all modules
- Clear, validated defaults with clinical justification
- Type-safe API with good IDE support

### Non-Goals
- Changing the mathematical formula of MADEX
- Adding new metrics (scope creep)
- Refactoring the analysis framework abstractions (separate change)

## Decisions

### Decision 1: Parameter Naming Convention

**Choice**: Use descriptive names (`center`, `critical_range`, `slope`) as primary, with `a`, `b`, `c` as aliases

**Rationale**: 
- Descriptive names are self-documenting for clinical users
- Short names useful for mathematical notation in papers
- Aligns with original `madex.py` implementation

**Alternatives considered**:
- Use only `a`, `b`, `c`: Rejected - loses clinical meaning
- Use only descriptive names: Rejected - inconvenient for mathematical work

### Decision 2: Default Value Resolution

**Choice**: Investigate clinical literature to determine correct `slope`/`c` default, then align both implementations

**Rationale**:
- Current mismatch (100 vs 40) produces significantly different scores
- Cannot arbitrarily pick one without understanding clinical implications
- May need to consult original MADEX paper/research

**Risk**: If original implementation has wrong default, fixing it is a breaking change

### Decision 3: Implementation Architecture

**Choice**: Single `MADEXCalculator` class in `madex.py`, with `MetricCalculator` delegating to it

```
madex.py
├── MADEXCalculator (canonical implementation)
│   ├── __init__(params: MADEXParameters | None)
│   ├── calculate(y_true, y_pred) -> float
│   └── calculate_detailed(y_true, y_pred) -> MADEXResult
└── mean_adjusted_exponent_error() -> float  (convenience function)

statistical_utils.py
└── MetricCalculator
    └── compute_madex() -> delegates to MADEXCalculator
```

**Rationale**:
- Single implementation eliminates divergence risk
- Class allows configuration reuse across multiple calculations
- Convenience function maintains backward compatibility

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Breaking change if defaults change | Document clearly, provide migration guide |
| Users relying on inconsistent behavior | Add deprecation warnings before changes |
| Performance regression from abstraction | Benchmark before/after, optimize if needed |

## Migration Plan

1. **Phase 1** (this change): Unify implementations, add deprecation warnings
2. **Phase 2** (future): Remove deprecated parameter names after one release cycle
3. **Rollback**: Keep old function signatures internally, revert exports if issues

## Open Questions

1. What is the clinically validated default for `slope`/`c` parameter?
   - Need to review original MADEX research/paper
   - May need input from clinical domain expert

2. Should `MADEXParameters` validation ranges be configurable?
   - Current: hard-coded ranges (a: 80-180, b: 20-100, c: 10-100)
   - Some research contexts may need different ranges

3. Is the exponent clipping (0.1-10) in `statistical_utils.py` clinically appropriate?
   - Original `madex.py` has no clipping
   - Need to understand if this affects edge case handling
