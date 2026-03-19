#!/usr/bin/env python3
"""Build a Washington NNTP workspace with raw line-image crops for review."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linealign.data.washington_handwritten_nntp import main


if __name__ == "__main__":
    main()
