"""Tests for Combined ShinkaEvolve + OpenEvolve Sequential & Parallel Cross-Pollinating Pipelines."""

from pathlib import Path
from arxiv_to_code.evolve_engine import (
    compute_ast_structural_fingerprint,
    extract_evolve_block,
    replace_evolve_block,
    run_combined_shinka_openevolve,
)
from arxiv_to_code.langextract_engine import extract_grounded_breakouts, fetch_paper_by_id
from arxiv_to_code.python_code_generator import synthesize_production_python_code


def test_evolve_block_extract_and_replace() -> None:
    paper = fetch_paper_by_id("2603.02277")
    assert paper is not None
    exts, _ = extract_grounded_breakouts(paper)
    bundle = synthesize_production_python_code(paper, exts)
    src = bundle["production_python_code"]

    block = extract_evolve_block(src)
    assert "def scan_and_score_payload" in block
    fp1 = compute_ast_structural_fingerprint(block)
    # Formatting change should have identical AST fingerprint
    fp2 = compute_ast_structural_fingerprint(block + "\n\n")
    assert fp1 == fp2

    replaced = replace_evolve_block(src, block)
    assert "# EVOLVE-BLOCK-START" in replaced
    assert "# EVOLVE-BLOCK-END" in replaced


def test_sequential_two_stage_shinka_to_openevolve(tmp_path: Path) -> None:
    paper = fetch_paper_by_id("2608.15888")
    assert paper is not None
    exts, _ = extract_grounded_breakouts(paper)
    bundle = synthesize_production_python_code(paper, exts)

    report = run_combined_shinka_openevolve(
        seed_python_code=bundle["production_python_code"],
        arxiv_id="2608.15888",
        mode="sequential",
        generations=4,
        num_islands=3,
        elites_dir=str(tmp_path / "seq_elites"),
    )
    assert report["evolve_mode"] == "sequential"
    assert report["champion_metrics"]["tests_passed"] == 4
    assert report["champion_metrics"]["total_fitness"] >= report["baseline_seed_metrics"]["total_fitness"]
    assert Path(report["champion_file_path"]).exists()
    assert len(report["shinkaevolve_summary"]["meta_scratchpad_stepping_stones"]) >= 1
    assert report["openevolve_map_elites_summary"]["occupied_niches_count"] >= 1


def test_parallel_cross_pollinating_ensemble(tmp_path: Path) -> None:
    paper = fetch_paper_by_id("2603.02277")
    assert paper is not None
    exts, _ = extract_grounded_breakouts(paper)
    bundle = synthesize_production_python_code(paper, exts)

    report = run_combined_shinka_openevolve(
        seed_python_code=bundle["production_python_code"],
        arxiv_id="2603.02277",
        mode="parallel",
        generations=4,
        num_islands=3,
        elites_dir=str(tmp_path / "par_elites"),
    )
    assert report["evolve_mode"] == "parallel"
    assert report["champion_metrics"]["tests_passed"] == 4
    assert len(report["cross_pollination_events"]) >= 2
    directions = [e["direction"] for e in report["cross_pollination_events"]]
    assert any("OpenEvolve -> ShinkaEvolve" in d for d in directions)
    assert any("ShinkaEvolve -> OpenEvolve" in d for d in directions)
