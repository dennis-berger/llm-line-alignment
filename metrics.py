# metrics.py
from typing import List, Dict
from collections import Counter

def _lev(a: List[str], b: List[str]) -> int:
    """Levenshtein distance on sequences a, b."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = tmp
    return dp[m]

def wer(ref: str, hyp: str) -> float:
    """Word Error Rate."""
    rt, ht = ref.strip().split(), hyp.strip().split()
    if not rt:
        return 0.0 if not ht else 1.0
    return _lev(rt, ht) / max(1, len(rt))

def cer(ref: str, hyp: str) -> float:
    """Character Error Rate."""
    rc, hc = list(ref.strip()), list(hyp.strip())
    if not rc:
        return 0.0 if not hc else 1.0
    return _lev(rc, hc) / max(1, len(rc))

def normalize_whitespace(s: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(s.split())

# -------- Line-level accuracy --------

def _split_lines(s: str) -> List[str]:
    # keep line order, strip trailing/leading whitespace per line
    if not s.strip():
        return []
    return [line.strip() for line in s.strip().splitlines()]

def line_accuracy(ref: str, hyp: str) -> float:
    """
    Line-level accuracy:
    fraction of lines that match exactly (per index).

    - Compare line i of ref with line i of hyp.
    - If ref or hyp has fewer lines, missing lines are treated as empty.
    - Exact string match required for a line to be counted as correct.
    """
    ref_lines = _split_lines(ref)
    hyp_lines = _split_lines(hyp)

    if not ref_lines and not hyp_lines:
        return 1.0
    if not ref_lines and hyp_lines:
        return 0.0

    n = max(len(ref_lines), len(hyp_lines))
    correct = 0
    for i in range(n):
        r = ref_lines[i] if i < len(ref_lines) else ""
        h = hyp_lines[i] if i < len(hyp_lines) else ""
        if r == h:
            correct += 1
    return correct / n

def reverse_line_accuracy(ref: str, hyp: str) -> float:
    """
    Line-level accuracy, but align from the last line upward.

    - Compare the last line of ref with the last line of hyp, then move upward.
    - If one side has fewer lines, missing lines are treated as empty.
    - Exact string match required for a line to be counted as correct.
    """
    ref_lines = _split_lines(ref)
    hyp_lines = _split_lines(hyp)

    if not ref_lines and not hyp_lines:
        return 1.0
    if not ref_lines and hyp_lines:
        return 0.0

    n = max(len(ref_lines), len(hyp_lines))
    correct = 0
    for i in range(n):
        r_idx = len(ref_lines) - 1 - i
        h_idx = len(hyp_lines) - 1 - i
        r = ref_lines[r_idx] if r_idx >= 0 else ""
        h = hyp_lines[h_idx] if h_idx >= 0 else ""
        if r == h:
            correct += 1
    return correct / n

def reverse_line_accuracy_norm(ref: str, hyp: str) -> float:
    """
    Normalized reverse line-level accuracy:
    same as reverse_line_accuracy, but normalize whitespace inside each line.
    """
    def norm_line(line: str) -> str:
        return normalize_whitespace(line)

    ref_lines = [norm_line(l) for l in _split_lines(ref)]
    hyp_lines = [norm_line(l) for l in _split_lines(hyp)]

    if not ref_lines and not hyp_lines:
        return 1.0
    if not ref_lines and hyp_lines:
        return 0.0

    n = max(len(ref_lines), len(hyp_lines))
    correct = 0
    for i in range(n):
        r_idx = len(ref_lines) - 1 - i
        h_idx = len(hyp_lines) - 1 - i
        r = ref_lines[r_idx] if r_idx >= 0 else ""
        h = hyp_lines[h_idx] if h_idx >= 0 else ""
        if r == h:
            correct += 1
    return correct / n

def line_accuracy_norm(ref: str, hyp: str) -> float:
    """
    Normalized line-level accuracy:
    same as line_accuracy, but normalize whitespace inside each line first.
    """
    def norm_line(line: str) -> str:
        return normalize_whitespace(line)

    ref_lines = [norm_line(l) for l in _split_lines(ref)]
    hyp_lines = [norm_line(l) for l in _split_lines(hyp)]

    if not ref_lines and not hyp_lines:
        return 1.0
    if not ref_lines and hyp_lines:
        return 0.0

    n = max(len(ref_lines), len(hyp_lines))
    correct = 0
    for i in range(n):
        r = ref_lines[i] if i < len(ref_lines) else ""
        h = hyp_lines[i] if i < len(hyp_lines) else ""
        if r == h:
            correct += 1
    return correct / n


# -------- Exact line matching with precision/recall/F1 --------

def exact_line_prf(gt_lines: List[str], pred_lines: List[str]) -> Dict[str, float]:
    """
    Compute exact line matching metrics: precision, recall, and F1.
    
    This metric counts how many lines are exactly correct, regardless of line ordering
    or line count differences, without double-counting matches.
    
    Matching strategy:
    - Lines are matched using exact string equality (after normalization)
    - Uses Counter-based matching: for each unique line text, matches the minimum
      of its count in GT and predictions (equivalent to maximum bipartite matching)
    - Each line can be matched at most once
    
    Metrics computed (micro-averaged when aggregated across samples):
    - exact_line_precision (ELP) = (# matched lines) / (# predicted lines)
    - exact_line_recall (ELR) = (# matched lines) / (# ground-truth lines)
    - exact_line_f1 (ELF1) = harmonic mean of precision and recall
    
    Args:
        gt_lines: List of ground-truth line strings (already normalized)
        pred_lines: List of predicted line strings (already normalized)
        
    Returns:
        Dictionary with keys: exact_line_precision, exact_line_recall, exact_line_f1
        
    Edge cases:
        - Empty predictions: precision = 0.0
        - Empty ground truth: recall = 0.0
        - F1 = 0.0 if both precision and recall are 0
    """
    # Count occurrences of each unique line
    gt_counter = Counter(gt_lines)
    pred_counter = Counter(pred_lines)
    
    # For each unique line, match the minimum count between GT and predictions
    # This ensures no double-counting and is equivalent to maximum bipartite matching
    all_lines = set(gt_counter.keys()) | set(pred_counter.keys())
    matches = sum(min(gt_counter[line], pred_counter[line]) for line in all_lines)
    
    total_pred = len(pred_lines)
    total_gt = len(gt_lines)
    
    # Compute precision: avoid division by zero
    if total_pred == 0:
        precision = 0.0
    else:
        precision = matches / total_pred
    
    # Compute recall: avoid division by zero
    if total_gt == 0:
        recall = 0.0
    else:
        recall = matches / total_gt
    
    # Compute F1: harmonic mean, handle zero case
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "exact_line_precision": precision,
        "exact_line_recall": recall,
        "exact_line_f1": f1,
    }


def exact_line_prf_from_text(ref: str, hyp: str, normalize: bool = False) -> Dict[str, float]:
    """
    Compute exact line matching metrics from full text strings.
    
    Args:
        ref: Ground truth text with line breaks
        hyp: Predicted text with line breaks
        normalize: If True, normalize whitespace within each line before matching
        
    Returns:
        Dictionary with exact_line_precision, exact_line_recall, exact_line_f1
    """
    ref_lines = _split_lines(ref)
    hyp_lines = _split_lines(hyp)
    
    if normalize:
        ref_lines = [normalize_whitespace(line) for line in ref_lines]
        hyp_lines = [normalize_whitespace(line) for line in hyp_lines]
    
    return exact_line_prf(ref_lines, hyp_lines)
