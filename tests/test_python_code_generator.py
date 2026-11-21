"""Tests for Paper-to-Python code generation and EVOLVE-BLOCK compliance."""

from pathlib import Path
from arxiv_to_code.evolve_engine import (
    EVOLVE_END_MARKER,
    EVOLVE_START_MARKER,
    evaluate_python_candidate,
)
from arxiv_to_code.langextract_engine import extract_grounded_breakouts, fetch_paper_by_id
from arxiv_to_code.python_code_generator import synthesize_production_python_code


def test_synthesize_production_python_code_and_execute(tmp_path: Path) -> None:
    for arxiv_id in ("2608.15888", "2603.02277", "2609.00005"):
        paper = fetch_paper_by_id(arxiv_id)
        assert paper is not None
        exts, _ = extract_grounded_breakouts(paper)
        bundle = synthesize_production_python_code(paper, exts, output_dir=str(tmp_path))

        assert bundle["language"] == "python"
        py_code = bundle["production_python_code"]
        assert EVOLVE_START_MARKER in py_code
        assert EVOLVE_END_MARKER in py_code
        assert Path(bundle["saved_module_path"]).exists()
        assert Path(bundle["saved_scorer_path"]).exists()

        metrics = evaluate_python_candidate(py_code, candidate_id=f"test-{arxiv_id}")
        assert metrics.error is None
        assert metrics.tests_passed == metrics.tests_total == 4
        assert metrics.total_fitness >= 70.0
