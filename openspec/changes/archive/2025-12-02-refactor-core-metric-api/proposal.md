# Change: Refactor Core Metric API for Consistency and Usability

## Why

The codebase has grown organically with two parallel implementations of MADEX calculation:
1. The original `madex.py` using positional parameters (`center`, `critical_range`, `slope`)
2. The `common/statistical_utils.py` using `MADEXParameters` dataclass with different defaults (`a`, `b`, `c`)

This creates confusion, inconsistent defaults, and maintenance burden. The parameter naming is inconsistent (`center`/`a`, `critical_range`/`b`, `slope`/`c`) and the default values differ between implementations (`slope=100` vs `c=40`).

## What Changes

### 1. Unify MADEX Parameter Interface
- Standardize on single parameter naming convention across all modules
- Reconcile default values between implementations
- **BREAKING**: Update `mean_adjusted_exponent_error()` signature to use consistent parameter names

### 2. Extract Core MADEX Calculator Class
- Move core MADEX calculation to a dedicated class in `madex.py`
- Make `statistical_utils.MetricCalculator` delegate to core implementation
- Eliminate duplicate calculation logic

### 3. Improve API Ergonomics
- Add `MADEXMetric` class with fluent interface for common use cases
- Support both positional parameters and `MADEXParameters` dataclass
- Add validation with clear error messages

### 4. Consolidate Exports
- Update `__init__.py` to expose unified API
- Deprecate direct access to internal implementations
- Add type hints to public API

## Impact

- Affected specs: `madex-metric` (new)
- Affected code:
  - `src/new_metric/madex.py` - Core refactoring
  - `src/new_metric/common/statistical_utils.py` - Delegate to core
  - `src/new_metric/common/clinical_data.py` - Parameter alignment
  - `src/new_metric/__init__.py` - Export updates
  - `tests/test_madex.py` - Test updates for new API

## Architecture Assessment Summary

### Current Issues Identified

1. **Duplicate MADEX Implementations**
   - `madex.py:mean_adjusted_exponent_error()` - original implementation
   - `statistical_utils.py:MetricCalculator.compute_madex()` - parallel implementation with different formula handling

2. **Inconsistent Parameter Naming**
   - `madex.py`: `center=125`, `critical_range=55`, `slope=100`
   - `clinical_data.py`: `a=125.0`, `b=55.0`, `c=40.0` (note: `c` differs!)
   
3. **Default Value Mismatch**
   - Original: `slope=100`
   - MADEXParameters: `c=40`
   - This causes different MADEX scores for the same input!

4. **Over-Abstraction in Common Module**
   - `analysis_framework.py` defines abstract classes not fully utilized
   - Multiple layers of abstraction for simple operations
   - Backward compatibility functions create maintenance overhead

5. **Test Coverage Gaps**
   - Minimal tests for core functions
   - No tests for common module classes
   - No integration tests between modules

### Recommended Priority

1. **High Priority**: Fix parameter default mismatch (breaking change)
2. **Medium Priority**: Unify parameter naming convention
3. **Medium Priority**: Consolidate MADEX implementations
4. **Lower Priority**: Simplify analysis framework abstractions
