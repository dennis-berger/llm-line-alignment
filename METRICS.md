# Evaluation Metrics

All metrics are computed both with raw text and with normalized text (whitespace collapsed).

## Character-Level Metrics

### Word Error Rate (WER)

**Formula:**

$$\text{WER} = \frac{\text{Levenshtein}(\text{ref}_\text{words}, \text{hyp}_\text{words})}{\max(1, |\text{ref}_\text{words}|)}$$

**Calculation:**
1. Split reference and hypothesis texts into word tokens
2. Compute Levenshtein (minimum edit) distance at word level
3. Normalize by the number of reference words

**Interpretation:**
- `WER = 0.0` → Perfect match
- `WER = 0.1` → 10% of words need to be inserted, deleted, or substituted
- Lower is better

**Implementation:** `metrics.py::wer(ref, hyp)`

---

### Character Error Rate (CER)

**Formula:**

$$\text{CER} = \frac{\text{Levenshtein}(\text{ref}_\text{chars}, \text{hyp}_\text{chars})}{\max(1, |\text{ref}_\text{chars}|)}$$

**Calculation:**
1. Convert reference and hypothesis texts into character sequences
2. Compute Levenshtein distance at character level
3. Normalize by the number of reference characters

**Interpretation:**
- `CER = 0.0` → Perfect match
- `CER = 0.01` → 1% of characters are incorrect
- More fine-grained than WER; lower is better

**Implementation:** `metrics.py::cer(ref, hyp)`

---

## Line-Level Metrics

### Line Accuracy (Forward)

**Formula:**

$$\text{Line Accuracy} = \frac{\sum_{i=1}^{n} \mathbb{1}[\text{ref}_i = \text{hyp}_i]}{n}$$

where 
$$
n = \max(|\text{ref}_\text{lines}|, |\text{hyp}_\text{lines}|)
$$

**Calculation:**
1. Split reference and hypothesis into lines
2. For each position $i$ from start to end:
   - Compare line $i$ of reference with line $i$ of hypothesis
   - If one side has fewer lines, treat missing lines as empty strings
   - Count as correct only if lines match exactly
3. Divide correct matches by total positions

**Interpretation:**
- Measures positional line accuracy from top to bottom
- Sensitive to line order and line count differences
- `1.0` = all lines match at corresponding positions
- Useful for detecting early misalignment

**Implementation:** `metrics.py::line_accuracy(ref, hyp)`

---

### Line Accuracy (Reverse)

**Formula:**

$$\text{Reverse Line Accuracy} = \frac{\sum_{i=1}^{n} \mathbb{1}[\text{ref}_{-i} = \text{hyp}_{-i}]}{n}$$

where 
$$
n = \max(|\text{ref}_\text{lines}|, |\text{hyp}_\text{lines}|)
$$
and negative indexing aligns from the last line.

**Calculation:**
1. Split reference and hypothesis into lines
2. For each position $i$ from end to start (reverse order):
   - Compare line from the end of reference with line from the end of hypothesis
   - If one side has fewer lines, treat missing lines as empty strings
   - Count as correct only if lines match exactly
3. Divide correct matches by total positions

**Interpretation:**
- Measures positional line accuracy from bottom to top
- Complements forward line accuracy
- Useful for detecting late misalignment
- Together with forward accuracy, helps diagnose where alignment breaks down

**Implementation:** `metrics.py::reverse_line_accuracy(ref, hyp)`

---

### Exact Line Precision/Recall/F1

**Core Matching Strategy:**
Uses Counter-based matching (equivalent to maximum bipartite matching):
- For each unique line text, match the minimum count in GT and predictions
- Each line can be matched at most once (no double-counting)
- Order-independent: lines can appear in any position

**Formulas:**

$$
\text{matches} = \sum_{\ell \in \text{all unique lines}} \min(\text{count}_\text{GT}(\ell), \text{count}_\text{pred}(\ell))
$$

$$
\text{Exact Line Precision (ELP)} = \frac{\text{matches}}{|\text{pred}_\text{lines}|}
$$

$$
\text{Exact Line Recall (ELR)} = \frac{\text{matches}}{|\text{GT}_\text{lines}|}
$$

$$
\text{Exact Line F1 (ELF1)} = \frac{2 \times \text{ELP} \times \text{ELR}}{\text{ELP} + \text{ELR}}
$$

**Calculation Example:**
```
GT lines:        ["Hello", "World", "Hello"]
Predicted lines: ["Hello", "World", "Test"]

Counts in GT:   {Hello: 2, World: 1}
Counts in pred: {Hello: 1, World: 1, Test: 1}

Matches: min(2,1) + min(1,1) + min(0,1) = 1 + 1 + 0 = 2

Precision = 2/3 = 0.667  (2 out of 3 predicted lines are correct)
Recall    = 2/3 = 0.667  (2 out of 3 GT lines were predicted)
F1        = 0.667
```

**Interpretation:**
- **Precision:** What fraction of predicted lines are actually in the ground truth?
- **Recall:** What fraction of ground truth lines were successfully predicted?
- **F1:** Harmonic mean balancing precision and recall
- Order-independent: handles line reordering gracefully
- Duplicate-aware: properly handles repeated lines without double-counting

**Edge Cases:**
- Empty predictions → Precision = 0.0
- Empty ground truth → Recall = 0.0
- Both zero → F1 = 0.0

**Implementation:** `metrics.py::exact_line_prf(gt_lines, pred_lines)` and `metrics.py::exact_line_prf_from_text(ref, hyp, normalize)`

## Raw vs. Normalized

- **Raw:** Exact whitespace preserved
- **Normalized:** Whitespace collapsed to single spaces
- Use normalized for fair method comparison

## CSV Output

```csv
id,len_gt,len_pred,wer,cer,wer_norm,cer_norm,
line_acc,line_acc_norm,rev_line_acc,rev_line_acc_norm,
exact_line_precision,exact_line_recall,exact_line_f1,
exact_line_precision_norm,exact_line_recall_norm,exact_line_f1_norm
```

Last row: Macro-averages (id = "MACRO_AVG")

## Usage

```python
from utils.evaluation import evaluate_prediction
result = evaluate_prediction(gt, pred, "sample_001")
```

## Testing

```bash
pytest tests/test_exact_line_metrics.py -v
```
