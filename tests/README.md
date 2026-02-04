# Testing Guide

This document describes the test suite structure, how to run tests, and guidelines for adding new tests.

## Overview

The test suite focuses primarily on **metric correctness**, ensuring that evaluation metrics produce expected values for known inputs. Tests are written using `pytest`.

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_exact_line_metrics.py -v
```

### Run Specific Test

```bash
pytest tests/test_exact_line_metrics.py::test_perfect_match -v
```

### Run with Coverage

```bash
pytest tests/ --cov=metrics --cov=utils --cov-report=html
```

## Test Structure

```
tests/
├── __init__.py
├── test_exact_line_metrics.py    # Exact line matching tests (21 tests)
└── __pycache__/
```

## Exact Line Metrics Tests

**File:** `tests/test_exact_line_metrics.py`

**Purpose:** Validate the Counter-based exact line matching implementation

**Test coverage:**

### 1. Acceptance Tests (4 tests)

Tests from requirements specification:

**`test_example_from_discussion`**
- Original example with partial matches
- Expected: Precision=0.5, Recall=0.667

**`test_duplicate_lines`**
- Lines appearing multiple times
- Counter-based matching handles multiplicities

**`test_empty_predictions`**
- Prediction is empty string
- Expected: Precision=0.0 or 1.0, Recall=0.0

**`test_empty_ground_truth`**
- Ground truth is empty
- Expected: Precision=0.0, Recall=0.0 or 1.0

### 2. Functional Tests (6 tests)

Core functionality tests:

**`test_perfect_match`**
- All lines identical in order
- Expected: P=R=F1=1.0

**`test_no_match`**
- No lines match
- Expected: P=R=F1=0.0

**`test_reordered_lines`**
- Lines correct but in different order
- Expected: P=R=F1=1.0 (order-independent)

**`test_multiple_duplicates`**
- Complex duplicate scenarios
- Matches min(count_gt, count_pred) per line

**`test_more_predictions_than_gt`**
- Prediction has extra lines
- Precision < 1.0, Recall = 1.0

**`test_more_gt_than_predictions`**
- Prediction missing lines
- Precision = 1.0, Recall < 1.0

### 3. Text Conversion Tests (7 tests)

Testing `exact_line_prf_from_text()` wrapper:

**`test_from_text_basic`**
- Simple text-to-lines conversion
- Splits on newlines correctly

**`test_from_text_trailing_newline`**
- Handles trailing newlines
- Doesn't create empty lines

**`test_from_text_normalization`**
- Whitespace normalization within lines
- Collapses multiple spaces

**`test_from_text_with_empty_lines`**
- Empty lines in input
- Preserved correctly

**`test_from_text_empty_strings`**
- Both strings empty
- Edge case handling

**`test_from_text_whitespace_only`**
- Only whitespace in strings
- Handles gracefully

**`test_from_text_leading_whitespace`**
- Leading/trailing spaces on lines
- Normalized correctly

### 4. Edge Cases (4 tests)

Boundary conditions:

**`test_single_line_match`**
- One line, perfect match
- P=R=F1=1.0

**`test_single_line_no_match`**
- One line, no match
- P=R=F1=0.0

**`test_case_sensitivity`**
- Lines differ only in case
- Should NOT match (case-sensitive)

**`test_punctuation_sensitivity`**
- Lines differ in punctuation
- Should NOT match (exact matching)

## Test Results

All 21 tests pass successfully.

**Example output:**
```
tests/test_exact_line_metrics.py::test_example_from_discussion PASSED
tests/test_exact_line_metrics.py::test_duplicate_lines PASSED
tests/test_exact_line_metrics.py::test_empty_predictions PASSED
tests/test_exact_line_metrics.py::test_empty_ground_truth PASSED
tests/test_exact_line_metrics.py::test_perfect_match PASSED
tests/test_exact_line_metrics.py::test_no_match PASSED
tests/test_exact_line_metrics.py::test_reordered_lines PASSED
...
====================== 21 passed in 0.15s ======================
```

## Adding New Tests

### Test Template

```python
def test_my_new_feature():
    """Test description."""
    # Arrange
    gt_lines = ["line 1", "line 2"]
    pred_lines = ["line 1", "line 3"]
    
    # Act
    result = exact_line_prf(gt_lines, pred_lines)
    
    # Assert
    assert result["exact_line_precision"] == pytest.approx(0.5)
    assert result["exact_line_recall"] == pytest.approx(0.5)
    assert result["exact_line_f1"] == pytest.approx(0.5)
```

### Guidelines

1. **Use descriptive test names:** `test_<what>_<condition>`
2. **Include docstrings:** Brief explanation of what's tested
3. **Use pytest.approx():** For floating-point comparisons
4. **Test edge cases:** Empty inputs, single items, extreme values
5. **Test both positive and negative cases**

### Example: Testing a New Metric

```python
def test_new_metric_basic():
    """Test new metric with simple input."""
    gt = "Line 1\nLine 2\nLine 3"
    pred = "Line 1\nLine 2\nLine 4"
    
    result = evaluate_prediction(gt, pred, "test_id")
    
    assert "new_metric" in result
    assert result["new_metric"] == pytest.approx(0.667)
```

## Testing Best Practices

### 1. Test One Thing Per Test

**Bad:**
```python
def test_all_metrics():
    # Tests precision, recall, F1, normalization, edge cases...
    pass
```

**Good:**
```python
def test_precision_with_duplicates():
    # Tests only precision calculation with duplicate lines
    pass

def test_recall_with_missing_lines():
    # Tests only recall when lines are missing
    pass
```

### 2. Use Clear Assertions

**Bad:**
```python
assert result["f1"] > 0.5  # Vague
```

**Good:**
```python
assert result["exact_line_f1"] == pytest.approx(0.667, abs=1e-3)
```

### 3. Test Edge Cases Explicitly

**Important edge cases:**
- Empty inputs
- Single-item inputs
- All matches / no matches
- Maximum/minimum values
- Special characters
- Unicode

### 4. Use Fixtures for Common Setup

```python
@pytest.fixture
def sample_data():
    """Common test data."""
    return {
        "gt": "Line 1\nLine 2\nLine 3",
        "pred": "Line 1\nLine 2\nLine 4"
    }

def test_with_fixture(sample_data):
    result = evaluate_prediction(
        sample_data["gt"], 
        sample_data["pred"], 
        "test"
    )
    assert result["exact_line_f1"] == pytest.approx(0.667)
```

## Test Coverage Goals

**Current coverage:**
- `metrics.py`: ~95% (exact line functions fully covered)
- `utils/evaluation.py`: ~70% (core metric computation covered)

**Areas needing more tests:**
- Character-level metrics (WER/CER)
- Line accuracy metrics
- Normalization edge cases
- Multi-page handling in evaluation

## Continuous Integration

**If using CI/CD (GitHub Actions, etc.):**

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

## Debugging Failed Tests

### 1. Verbose Output

```bash
pytest tests/test_exact_line_metrics.py::test_duplicate_lines -v -s
```

### 2. Print Intermediate Values

```python
def test_debug_example():
    gt_lines = ["A", "B"]
    pred_lines = ["A", "C"]
    
    result = exact_line_prf(gt_lines, pred_lines)
    
    # Debug print
    print(f"Result: {result}")
    
    assert result["exact_line_precision"] == pytest.approx(0.5)
```

### 3. Use pytest --pdb

```bash
pytest tests/ --pdb  # Drop into debugger on failure
```

### 4. Check Floating-Point Precision

```python
# If test fails with "0.666666... != 0.667"
assert result["f1"] == pytest.approx(0.667, abs=1e-3)  # Allow small error
```

## Manual Testing

**For integration testing:**

```bash
# Quick smoke test on one sample
python run_eval_m1.py --ids 0001 --shots 0

# Check output
cat predictions_m1/0001.txt
head -n 2 evaluation_qwen_m1.csv
```

## Performance Testing

**For large-scale evaluation:**

```python
import time

def test_metric_performance():
    """Test metric computation speed."""
    gt = "Line\n" * 1000  # 1000 lines
    pred = "Line\n" * 1000
    
    start = time.time()
    result = evaluate_prediction(gt, pred, "perf_test")
    elapsed = time.time() - start
    
    assert elapsed < 0.1  # Should be fast (<100ms)
```

## Test Data

**Location:** Embedded in test files

**For larger test datasets:**
- Consider `tests/fixtures/` directory
- Store sample GT and prediction pairs
- Load via pytest fixtures

**Example:**
```
tests/
├── fixtures/
│   ├── sample_001_gt.txt
│   └── sample_001_pred.txt
└── test_integration.py
```

## Related Documentation

- **[METRICS.md](METRICS.md)** - Metric definitions being tested
- **[docs/architecture.md](docs/architecture.md)** - System components
- **pytest documentation:** https://docs.pytest.org/
