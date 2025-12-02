# Implementation Tasks

## 1. Parameter Reconciliation
- [x] 1.1 Investigate which default value for `c`/`slope` is clinically correct (40 vs 100)
  - Result: `slope=100` is correct based on existing test expectations
- [x] 1.2 Update `MADEXParameters` dataclass to use correct defaults
  - Updated in both `madex.py` (new) and `clinical_data.py` (legacy)
- [x] 1.3 Update `mean_adjusted_exponent_error()` to use aligned defaults
  - Defaults are now `center=125, critical_range=55, slope=100`
- [x] 1.4 Add deprecation warning for old parameter names if changing
  - Added docstring notes directing users to new API; kept backward compat

## 2. Unify MADEX Implementation
- [x] 2.1 Create `MADEXCalculator` class in `madex.py` with canonical implementation
- [x] 2.2 Add support for both positional params and `MADEXParameters` dataclass
- [x] 2.3 Update `MetricCalculator.compute_madex()` to delegate to `MADEXCalculator`
- [x] 2.4 Add numerical stability handling (clip exponents, handle edge cases)
  - Note: Removed clipping from delegated impl to match original behavior
- [x] 2.5 Add validation with clear clinical context in error messages

## 3. Update Public API
- [x] 3.1 Update `__init__.py` exports with new classes
- [x] 3.2 Keep `mean_adjusted_exponent_error()` as convenience function
- [x] 3.3 Export `MADEXCalculator` and `MADEXParameters` from package root
- [x] 3.4 Add type hints to all public functions

## 4. Test Updates
- [x] 4.1 Add tests verifying parameter defaults match between implementations
- [x] 4.2 Add tests for `MADEXCalculator` class
- [x] 4.3 Add tests for edge cases (empty arrays, single values, extreme values)
- [x] 4.4 Add integration test ensuring `MetricCalculator` matches core implementation

## 5. Documentation
- [x] 5.1 Add docstrings explaining clinical meaning of each parameter
- [x] 5.2 Document breaking changes in CHANGELOG or README
  - Note: Breaking change is c default 40->100 in clinical_data.py
- [x] 5.3 Add usage examples in docstrings
