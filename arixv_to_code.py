#!/usr/bin/env python3
"""
arixv_to_code.py
======================================================================================
Unified All-in-One Root Entrypoint for ArXiv-to-Code.

Every capability in the repository can be run directly via flags on `arixv_to_code.py`:
  python arixv_to_code.py --help
  python arixv_to_code.py --test
  python arixv_to_code.py --pytest
  python arixv_to_code.py --search "LLM container sandbox escape" --evolves --evolve-mode hybrid --evolve-count 2 --verify-llm
  python arixv_to_code.py --arxiv-id 2608.15888 --evolves --evolve-mode hybrid --evolve-count 3 --verify-llm
  python arixv_to_code.py --extract-doc ./paper.pdf --langextract-llm --html-report ./audit.html --evolves --verify-llm
  python arixv_to_code.py --dynarot
  python arixv_to_code.py --install-models
  python arixv_to_code.py --audit --arxiv-id 2603.02277
  python arixv_to_code.py --langgraph --arxiv-id 2603.02277
  python arixv_to_code.py --list-tools
  python arixv_to_code.py --mcp
"""

import sys
from pathlib import Path

_QUIET_FLAGS = {"--mcp", "--status", "--list-tools", "--mcp-tools", "--help", "-h", "-V", "--version", "--no-progress"}
if __name__ == "__main__" and not any(a in _QUIET_FLAGS for a in sys.argv[1:]):
    sys.stderr.write("[ARIXV-TO-CODE] Initializing ArXiv-to-Code (Google LangExtract + DynaRot + ShinkaEvolve/OpenEvolve)...\n")
    sys.stderr.flush()

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from arxiv_to_code.cli import main

if __name__ == "__main__":
    sys.exit(main())
