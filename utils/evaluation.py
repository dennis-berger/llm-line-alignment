"""Evaluation functions for line alignment metrics."""
import logging
from metrics import (
    wer, cer, normalize_whitespace,
    line_accuracy, reverse_line_accuracy,
    line_accuracy_norm, reverse_line_accuracy_norm,
    exact_line_prf_from_text
)

logger = logging.getLogger(__name__)


def evaluate_prediction(gt: str, pred: str, sample_id: str) -> dict:
    """Evaluate prediction against ground truth with all metrics.
    
    Args:
        gt: Ground truth text with line breaks
        pred: Predicted text with line breaks
        sample_id: Sample identifier for logging
        
    Returns:
        Dictionary with all evaluation metrics including:
        - WER/CER (raw and whitespace-normalized)
        - line_accuracy and reverse variants
        - exact_line_precision/recall/f1 (raw and normalized)
    """
    # Compute exact line matching metrics (raw and normalized)
    exact_prf_raw = exact_line_prf_from_text(gt, pred, normalize=False)
    exact_prf_norm = exact_line_prf_from_text(gt, pred, normalize=True)
    
    return {
        'id': sample_id,
        'len_gt': len(gt),
        'len_pred': len(pred),
        'wer': wer(gt, pred),
        'cer': cer(gt, pred),
        'wer_whitespace_normalized': wer(normalize_whitespace(gt), normalize_whitespace(pred)),
        'cer_whitespace_normalized': cer(normalize_whitespace(gt), normalize_whitespace(pred)),
        'line_accuracy': line_accuracy(gt, pred),
        'line_accuracy_reverse': reverse_line_accuracy(gt, pred),
        'line_accuracy_whitespace_normalized': line_accuracy_norm(gt, pred),
        'line_accuracy_whitespace_normalized_reverse': reverse_line_accuracy_norm(gt, pred),
        # Exact line matching metrics (raw)
        'exact_line_precision': exact_prf_raw['exact_line_precision'],
        'exact_line_recall': exact_prf_raw['exact_line_recall'],
        'exact_line_f1': exact_prf_raw['exact_line_f1'],
        # Exact line matching metrics (whitespace-normalized)
        'exact_line_precision_norm': exact_prf_norm['exact_line_precision'],
        'exact_line_recall_norm': exact_prf_norm['exact_line_recall'],
        'exact_line_f1_norm': exact_prf_norm['exact_line_f1'],
    }
