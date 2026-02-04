# Evaluation Metrics

All metrics are computed both with raw text and with normalized text (whitespace collapsed).

## Character-Level Metrics

### Word Error Rate (WER)
- Minimum edit distance at word level / total words
- `WER = 0.0` = perfect, `WER = 0.1` = 10% changed

### Character Error Rate (CER)
- Minimum edit distance at character level / total characters  
- `CER = 0.0` = perfect, `CER = 0.01` = 1% changed

## Line-Level Metrics

### Line Accuracy (Forward/Reverse)
- Fraction of lines matching at corresponding positions
- Forward: start to end, Reverse: end to start
- Sensitive to line order and count

### Exact Line Precision/Recall/F1
- **Precision:** Fraction of predicted lines in GT
- **Recall:** Fraction of GT lines in prediction
- **F1:** Harmonic mean of precision and recall
- Counter-based matching (order-independent, handles duplicates)

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
