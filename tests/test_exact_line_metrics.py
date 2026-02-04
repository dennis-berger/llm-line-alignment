"""Unit tests for exact line matching metrics (precision/recall/F1)."""
import pytest
from metrics import exact_line_prf, exact_line_prf_from_text, normalize_whitespace


class TestExactLinePRF:
    """Test cases for exact_line_prf function."""
    
    def test_line_break_variation(self):
        """Test case A: Lines split differently between GT and prediction."""
        gt = ["Hello there,", "thank you for your", "detailed response."]
        pred = ["Hello there,", "thank you", "for your", "detailed response."]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 2 ("Hello there," and "detailed response.")
        # precision = 2/4 = 0.5
        # recall = 2/3 = 0.666...
        # f1 = 2 * 0.5 * 0.666... / (0.5 + 0.666...) = 0.571...
        
        assert result['exact_line_precision'] == pytest.approx(0.5, abs=0.001)
        assert result['exact_line_recall'] == pytest.approx(2/3, abs=0.001)
        assert result['exact_line_f1'] == pytest.approx(0.571, abs=0.01)
    
    def test_duplicate_lines(self):
        """Test case B: Duplicate lines handled correctly (no double-counting)."""
        gt = ["a", "a", "b"]
        pred = ["a", "b", "b"]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 2 (one "a" + one "b")
        # precision = 2/3
        # recall = 2/3
        # f1 = 2/3
        
        assert result['exact_line_precision'] == pytest.approx(2/3, abs=0.001)
        assert result['exact_line_recall'] == pytest.approx(2/3, abs=0.001)
        assert result['exact_line_f1'] == pytest.approx(2/3, abs=0.001)
    
    def test_empty_pred(self):
        """Test case C: Empty predictions."""
        gt = ["x"]
        pred = []
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 0, precision = 0, recall = 0, f1 = 0
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0
    
    def test_empty_gt(self):
        """Test case D: Empty ground truth."""
        gt = []
        pred = ["x"]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 0, precision = 0, recall = 0, f1 = 0
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0
    
    def test_both_empty(self):
        """Test case: Both GT and predictions are empty."""
        gt = []
        pred = []
        
        result = exact_line_prf(gt, pred)
        
        # Expected: all zeros (no lines to match)
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0
    
    def test_perfect_match(self):
        """Test case: Perfect match (all lines identical and in same order)."""
        gt = ["line 1", "line 2", "line 3"]
        pred = ["line 1", "line 2", "line 3"]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 3, precision = 1.0, recall = 1.0, f1 = 1.0
        assert result['exact_line_precision'] == 1.0
        assert result['exact_line_recall'] == 1.0
        assert result['exact_line_f1'] == 1.0
    
    def test_reordered_lines(self):
        """Test case: Lines are correct but reordered."""
        gt = ["first", "second", "third"]
        pred = ["third", "first", "second"]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 3 (all lines match, order doesn't matter)
        # precision = 1.0, recall = 1.0, f1 = 1.0
        assert result['exact_line_precision'] == 1.0
        assert result['exact_line_recall'] == 1.0
        assert result['exact_line_f1'] == 1.0
    
    def test_no_matches(self):
        """Test case: No lines match at all."""
        gt = ["a", "b", "c"]
        pred = ["x", "y", "z"]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 0, precision = 0, recall = 0, f1 = 0
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0
    
    def test_multiple_duplicates(self):
        """Test case: Multiple occurrences of the same line."""
        gt = ["a", "a", "a", "b"]
        pred = ["a", "a", "b", "b"]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 3 (min(3,2) for "a" + min(1,2) for "b")
        # precision = 3/4 = 0.75
        # recall = 3/4 = 0.75
        # f1 = 0.75
        
        assert result['exact_line_precision'] == 0.75
        assert result['exact_line_recall'] == 0.75
        assert result['exact_line_f1'] == 0.75
    
    def test_more_pred_than_gt(self):
        """Test case: More predicted lines than GT."""
        gt = ["line1", "line2"]
        pred = ["line1", "line2", "line3", "line4"]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 2
        # precision = 2/4 = 0.5
        # recall = 2/2 = 1.0
        # f1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 0.666...
        
        assert result['exact_line_precision'] == 0.5
        assert result['exact_line_recall'] == 1.0
        assert result['exact_line_f1'] == pytest.approx(2/3, abs=0.001)
    
    def test_more_gt_than_pred(self):
        """Test case: More GT lines than predicted."""
        gt = ["line1", "line2", "line3", "line4"]
        pred = ["line1", "line2"]
        
        result = exact_line_prf(gt, pred)
        
        # Expected: matches = 2
        # precision = 2/2 = 1.0
        # recall = 2/4 = 0.5
        # f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.666...
        
        assert result['exact_line_precision'] == 1.0
        assert result['exact_line_recall'] == 0.5
        assert result['exact_line_f1'] == pytest.approx(2/3, abs=0.001)


class TestExactLinePRFFromText:
    """Test cases for exact_line_prf_from_text function."""
    
    def test_from_text_basic(self):
        """Test basic text-to-lines conversion."""
        gt = "Dear John,\nthank you for your\ndetailed feedback."
        pred = "Dear John,\nthank you\nfor your\ndetailed feedback."
        
        result = exact_line_prf_from_text(gt, pred, normalize=False)
        
        # Same as test_example_from_discussion
        assert result['exact_line_precision'] == pytest.approx(0.5, abs=0.001)
        assert result['exact_line_recall'] == pytest.approx(2/3, abs=0.001)
        assert result['exact_line_f1'] == pytest.approx(0.571, abs=0.01)
    
    def test_from_text_with_trailing_whitespace(self):
        """Test that trailing/leading whitespace is stripped per line."""
        gt = "  line1  \n  line2  \n  line3  "
        pred = "line1\nline2\nline3"
        
        result = exact_line_prf_from_text(gt, pred, normalize=False)
        
        # After stripping, all lines should match
        assert result['exact_line_precision'] == 1.0
        assert result['exact_line_recall'] == 1.0
        assert result['exact_line_f1'] == 1.0
    
    def test_from_text_normalize_whitespace(self):
        """Test whitespace normalization within lines."""
        gt = "hello   world\nfoo  bar"
        pred = "hello world\nfoo bar"
        
        # Without normalization, lines don't match (different whitespace)
        result_no_norm = exact_line_prf_from_text(gt, pred, normalize=False)
        assert result_no_norm['exact_line_precision'] == 0.0
        assert result_no_norm['exact_line_recall'] == 0.0
        
        # With normalization, lines should match
        result_norm = exact_line_prf_from_text(gt, pred, normalize=True)
        assert result_norm['exact_line_precision'] == 1.0
        assert result_norm['exact_line_recall'] == 1.0
        assert result_norm['exact_line_f1'] == 1.0
    
    def test_from_text_empty_lines(self):
        """Test handling of empty lines in input."""
        gt = "line1\n\nline2\n"
        pred = "line1\nline2"
        
        result = exact_line_prf_from_text(gt, pred, normalize=False)
        
        # GT has an empty line, pred doesn't
        # GT lines: ["line1", "", "line2"]
        # Pred lines: ["line1", "line2"]
        # Matches: min(1,1) for "line1" + min(0,0) for "" + min(1,1) for "line2" = 2
        # precision = 2/2 = 1.0
        # recall = 2/3 = 0.666...
        
        assert result['exact_line_precision'] == 1.0
        assert result['exact_line_recall'] == pytest.approx(2/3, abs=0.001)
    
    def test_from_text_empty_strings(self):
        """Test handling of completely empty strings."""
        result = exact_line_prf_from_text("", "", normalize=False)
        
        # Both empty -> no lines
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0
    
    def test_from_text_only_whitespace(self):
        """Test handling of strings with only whitespace."""
        result = exact_line_prf_from_text("   \n  \n  ", "", normalize=False)
        
        # After stripping, both should be empty
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0


class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_single_line_match(self):
        """Test with single line that matches."""
        gt = ["single line"]
        pred = ["single line"]
        
        result = exact_line_prf(gt, pred)
        
        assert result['exact_line_precision'] == 1.0
        assert result['exact_line_recall'] == 1.0
        assert result['exact_line_f1'] == 1.0
    
    def test_single_line_no_match(self):
        """Test with single line that doesn't match."""
        gt = ["line a"]
        pred = ["line b"]
        
        result = exact_line_prf(gt, pred)
        
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0
    
    def test_case_sensitive(self):
        """Test that matching is case-sensitive."""
        gt = ["Hello World"]
        pred = ["hello world"]
        
        result = exact_line_prf(gt, pred)
        
        # Different case -> no match
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0
    
    def test_punctuation_sensitive(self):
        """Test that matching is punctuation-sensitive."""
        gt = ["Hello, World!"]
        pred = ["Hello World"]
        
        result = exact_line_prf(gt, pred)
        
        # Different punctuation -> no match
        assert result['exact_line_precision'] == 0.0
        assert result['exact_line_recall'] == 0.0
        assert result['exact_line_f1'] == 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
