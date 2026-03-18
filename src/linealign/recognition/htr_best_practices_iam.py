"""Compatibility alias for the vendored IAM PyLaia recognizer."""
from __future__ import annotations

from .pylaia import PyLaiaRecognizer


class IAMBestPracticesRecognizer(PyLaiaRecognizer):
    """Backward-compatible alias for the public IAM PyLaia checkpoint."""

    name = "htr_best_practices_iam"
