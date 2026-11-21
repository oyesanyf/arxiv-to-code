#!/usr/bin/env python3
"""Root CLI launcher for ArXiv-to-Code (`python cli.py ...`)."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from arxiv_to_code.cli import main

if __name__ == "__main__":
    sys.exit(main())
