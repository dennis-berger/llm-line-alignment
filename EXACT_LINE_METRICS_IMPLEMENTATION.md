# Exact Line Matching Metrics - Implementation Summary

## Overview
This implementation adds a new metric for measuring exact line matches between ground truth and predicted texts, reporting precision, recall, and F1 scores. The metric is robust to line reordering and count differences, using Counter-based matching to avoid double-counting.

## Changes Made

### 1. Core Metrics Module (`metrics.py`)
Added two new functions:

#### `exact_line_prf(gt_lines: List[str], pred_lines: List[str]) -> Dict[str, float]`
- Computes exact line matching metrics: precision, recall, and F1
- Uses Counter-based matching (equivalent to maximum bipartite matching)
- Each unique line text can be matched at most once
- Returns: `{"exact_line_precision": ..., "exact_line_recall": ..., "exact_line_f1": ...}`

#### `exact_line_prf_from_text(ref: str, hyp: str, normalize: bool = False) -> Dict[str, float]`
- Convenience wrapper that splits full text into lines
- Optionally normalizes whitespace within each line
- Returns same dictionary as `exact_line_prf`

**Key Properties:**
- **Matching Strategy**: For each unique line text, matches `min(count_gt[line], count_pred[line])` occurrences
- **No Double-Counting**: Each line in GT can match at most one line in predictions
- **Edge Cases Handled**: Empty predictions, empty GT, zero division, etc.
- **Normalization**: Uses existing `normalize_whitespace` function (collapses whitespace to single spaces)

### 2. Evaluation Module (`utils/evaluation.py`)
Updated `evaluate_prediction()` to compute and return 6 new metrics:
- `exact_line_precision` (raw)
- `exact_line_recall` (raw)
- `exact_line_f1` (raw)
- `exact_line_precision_norm` (whitespace-normalized)
- `exact_line_recall_norm` (whitespace-normalized)
- `exact_line_f1_norm` (whitespace-normalized)

### 3. Evaluation Scripts
Updated all three method evaluation scripts:
- `run_eval_m1.py`
- `run_eval_m2.py`
- `run_eval_m3.py`

**Changes in each script:**
1. Added aggregation variables for new metrics
2. Updated per-sample CSV rows to include 6 new columns
3. Updated macro-average computation to include new metrics
4. Updated CSV headers with new column names
5. Updated logging to display exact line precision/recall/F1

**Aggregation Strategy:**
- Uses **macro-averaging**: averages per-sample metrics across all samples
- This matches the existing aggregation strategy for line_acc and other metrics
- Alternative would be micro-averaging (aggregate counts first, then compute metrics)

### 4. Unit Tests (`tests/test_exact_line_metrics.py`)
Created comprehensive test suite with 21 test cases covering:

**Test Categories:**
- **Acceptance Tests** (from requirements):
  - Example from discussion (2/4 precision, 2/3 recall)
  - Duplicate lines handling
  - Empty predictions
  - Empty ground truth

- **Functional Tests**:
  - Perfect match (all lines identical)
  - Reordered lines (tests order-independence)
  - No matches (zero metrics)
  - Multiple duplicates
  - More predictions than GT
  - More GT than predictions

- **Text Conversion Tests**:
  - Basic text-to-lines conversion
  - Trailing/leading whitespace handling
  - Whitespace normalization
  - Empty lines in input
  - Empty strings
  - Only whitespace strings

- **Edge Cases**:
  - Single line match/no-match
  - Case sensitivity
  - Punctuation sensitivity

**All 21 tests pass successfully.**

## Usage Example

```python
from utils.evaluation import evaluate_prediction

gt = "Dear Anna,\nthank you for your\ndetailed feedback."
pred = "Dear Anna,\nthank you\nfor your\ndetailed feedback."

result = evaluate_prediction(gt, pred, "sample_id")

# New metrics available:
print(f"Precision: {result['exact_line_precision']:.3f}")  # 0.500
print(f"Recall: {result['exact_line_recall']:.3f}")        # 0.667
print(f"F1: {result['exact_line_f1']:.3f}")                # 0.571
```

## CSV Output Format

New columns added to evaluation CSV files:
```
id, len_gt, len_pred, wer, cer, wer_norm, cer_norm,
line_acc, line_acc_norm, rev_line_acc, rev_line_acc_norm,
exact_line_precision, exact_line_recall, exact_line_f1,
exact_line_precision_norm, exact_line_recall_norm, exact_line_f1_norm
```

## Mathematical Definition

Given:
- GT lines: `[g1, g2, ..., gn]`
- Predicted lines: `[p1, p2, ..., pm]`

Compute line counts:
- `count_gt[line]` = number of occurrences of `line` in GT
- `count_pred[line]` = number of occurrences of `line` in predictions

Total matches:
```
matches = Σ min(count_gt[line], count_pred[line]) for all unique lines
```

Metrics:
- **Precision** = matches / m (total predicted lines)
- **Recall** = matches / n (total GT lines)
- **F1** = 2 * precision * recall / (precision + recall)

## Design Decisions

1. **Counter-based matching**: More efficient than Hungarian algorithm for exact string matching
2. **Micro vs Macro averaging**: Used macro-averaging to match existing metrics' aggregation strategy
3. **Normalization consistency**: Reuses existing `normalize_whitespace` and `_split_lines` functions
4. **Separate raw/normalized metrics**: Provides both for flexibility in analysis
5. **Zero handling**: Carefully handles division by zero and both-zero cases

## Backward Compatibility

- ✅ Existing metrics (`line_acc`, `reverse_line_acc`) remain unchanged
- ✅ All existing functionality preserved
- ✅ New metrics are additive only
- ✅ No breaking changes to existing code

## Testing

Run tests:
```bash
pytest tests/test_exact_line_metrics.py -v
```

All 21 tests pass with expected values matching the acceptance criteria.
