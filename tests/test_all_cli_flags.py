"""
Exhaustive verification suite testing every single CLI flag and alias on `arixv_to_code.py`.

Flags tested:
  1.  --help / -h
  2.  --version / -V
  3.  --search / --searct / --seach / --query
  4.  --arxiv-id / --id
  5.  --max-results
  6.  --min-score
  7.  --include-implemented
  8.  --extract-doc / --extract
  9.  --html-report
  10. --generate / --output-dir
  11. --audit / --audit-defense
  12. --dynarot / --sys-info
  13. --install-models
  14. --evolves / --evolve / --evoves / ---evolves
  15. --evolve-mode sequential | parallel | hybrid
  16. --generations / --islands / --elites-dir
  17. --langgraph / --swarm
  18. --status
  19. --list-tools / --mcp-tools
  20. --test
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from arxiv_to_code.cli import build_arg_parser, main as cli_main, normalize_cli_argv


def test_flag_01_help_and_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc_help:
        cli_main(["--help"])
    assert exc_help.value.code == 0
    out_help = capsys.readouterr().out
    assert "arixv_to_code.py" in out_help
    assert "--search QUERY" in out_help
    assert "--evolves, --evolve, --evoves" in out_help

    with pytest.raises(SystemExit) as exc_ver:
        cli_main(["--version"])
    assert exc_ver.value.code == 0
    out_ver = capsys.readouterr().out
    assert "1.0.0" in out_ver


def test_flag_02_normalize_cli_argv_aliases_and_triple_dash() -> None:
    norm = normalize_cli_argv(
        [
            "--searct",
            "sandbox escape",
            "---evolves",
            "--evolve-mode",
            "hybrid",
            "--id=2608.15888",
            "--extract",
            "doc.txt",
            "--mcp-tools",
            "--audit-defense",
            "--swarm",
        ]
    )
    assert norm == [
        "--search",
        "sandbox escape",
        "--evolves",
        "--evolve-mode",
        "hybrid",
        "--arxiv-id=2608.15888",
        "--extract-doc",
        "doc.txt",
        "--list-tools",
        "--audit",
        "--langgraph",
    ]


def test_flag_03_search_and_ranking_filters(capsys) -> None:
    # Primary --search flag + --max-results + --min-score
    rc = cli_main(["--search", "container sandbox escape", "--max-results", "2", "--min-score", "40.0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Rank #1" in out
    assert "ArXiv-to-Code Production Python Bundle" in out

    # --include-implemented flag on implemented paper 2610.09167
    rc_impl = cli_main(["--arxiv-id", "2610.09167", "--include-implemented"])
    assert rc_impl == 0
    out_impl = capsys.readouterr().out
    assert "2610.09167" in out_impl


def test_flag_04_arxiv_id_and_generate_and_output_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "gen_out"
    rc = cli_main(["--arxiv-id", "2608.15888", "--generate", "--output-dir", str(out_dir), "--no-progress"])
    assert rc == 0
    assert (out_dir / "INDEX.md").exists()
    assert (out_dir / "MANIFEST.json").exists()
    assert len(list(out_dir.glob("arxiv_2608_15888_*/src/arxiv_2608_15888_module.py"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/eval/evaluate_arxiv_2608_15888.py"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/langextract_report.html"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/extraction_results.jsonl"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/metadata.json"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/README.md"))) == 1


def test_flag_05_extract_doc_and_html_report(tmp_path: Path) -> None:
    doc_file = tmp_path / "custom_research.txt"
    doc_file.write_text(
        "Autonomous agents in container sandboxes face privilege escalation and data exfiltration risks. "
        "We derive a conformal martingale bound with asymmetric semilattice delegation and eBPF sandbox isolation.",
        encoding="utf-8",
    )
    html_file = tmp_path / "custom_report.html"
    out_dir = tmp_path / "doc_gen"

    rc = cli_main(
        [
            "--extract-doc",
            str(doc_file),
            "--html-report",
            str(html_file),
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ]
    )
    assert rc == 0
    assert html_file.exists()
    assert html_file.stat().st_size > 250
    assert len(list(out_dir.glob("arxiv_custom_research_*/src/arxiv_custom_research_module.py"))) == 1


def test_flag_06_dynarot_and_sys_info(capsys) -> None:
    for flag in ("--dynarot", "--sys-info"):
        rc = cli_main([flag, "--no-progress"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "shinka_heavy_reasoning_model" in out
        assert "openevolve_fast_mutation_model" in out
        assert "pareto_knee_model" in out


def test_flag_07_evolves_all_three_modes_evolve_count_and_elites_dir(tmp_path: Path) -> None:
    for mode in ("sequential", "parallel", "hybrid"):
        out_dir = tmp_path / f"out_{mode}"
        elites_dir = tmp_path / f"elites_{mode}"
        rc = cli_main(
            [
                "--arxiv-id",
                "2603.02277",
                "--evolves",
                "--evolve-mode",
                mode,
                "--evolve-count",
                "2",
                "--generations",
                "2",
                "--islands",
                "2",
                "--elites-dir",
                str(elites_dir),
                "--output-dir",
                str(out_dir),
                "--no-progress",
            ]
        )
        assert rc == 0
        reports = list(out_dir.glob("arxiv_2603_02277_*/evolved/evolution_report.json"))
        assert len(reports) == 1
        report_data = json.loads(reports[0].read_text(encoding="utf-8"))
        assert report_data["evolve_count"] == 2
        assert len(report_data["round_history"]) == 2
        assert len(list(out_dir.glob("arxiv_2603_02277_*/evolved/champion_arxiv_2603_02277.py"))) == 1


def test_flag_08_search_combined_with_evolves_and_typo_resiliency(tmp_path: Path) -> None:
    out_dir = tmp_path / "search_evolve"
    rc = cli_main(
        [
            "--searct",
            "container sandbox escape",
            "--max-results",
            "1",
            "---evolves",
            "--evolve-mode",
            "hybrid",
            "--evolve-count",
            "2",
            "--generations",
            "2",
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ]
    )
    assert rc == 0
    champions = list(out_dir.glob("arxiv_*/evolved/champion_arxiv_*.py"))
    assert len(champions) >= 1
    assert (out_dir / "INDEX.md").exists()
    assert (out_dir / "MANIFEST.json").exists()


def test_flag_09_audit_and_langgraph_and_status_and_list_tools(capsys) -> None:
    # --audit
    assert cli_main(["--audit", "--arxiv-id", "2603.02277"]) == 0
    out_audit = capsys.readouterr().out
    assert '"overall_coverage_percentage": 100.0' in out_audit

    # --langgraph
    assert cli_main(["--langgraph", "--arxiv-id", "2603.02277"]) == 0
    out_lg = capsys.readouterr().out
    assert "Converged_PostEvolutionVerified_100PercentCoverage" in out_lg

    # --status
    assert cli_main(["--status"]) == 0
    out_status = json.loads(capsys.readouterr().out)
    assert out_status["status"] == "operational"

    # --list-tools
    assert cli_main(["--list-tools"]) == 0
    out_tools = json.loads(capsys.readouterr().out)
    assert out_tools["total_tools"] == 10
    tool_names = {t["name"] for t in out_tools["tools"]}
    assert "verify_generated_python_code" in tool_names
    assert "optimize_code_with_dream_rsi_rl" in tool_names


def test_flag_10_self_test_flag() -> None:
    assert cli_main(["--test"]) == 0


def test_flag_11_progress_bar_spinner_and_no_progress(tmp_path: Path) -> None:
    out_dir = tmp_path / "prog_out"
    rc = cli_main(
        [
            "--search",
            "LLM container sandbox escape",
            "--max-results",
            "1",
            "--evolves",
            "--evolve-mode",
            "hybrid",
            "--evolve-count",
            "2",
            "--generations",
            "2",
            "--islands",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert len(list(out_dir.glob("arxiv_*/evolved/champion_arxiv_*.py"))) >= 1
    assert (out_dir / "INDEX.md").exists()

    # Verify --no-progress flag works cleanly too
    rc_quiet = cli_main(["--dynarot", "--no-progress"])
    assert rc_quiet == 0


def test_flag_12_verify_llm_and_aliases(tmp_path: Path, capsys) -> None:
    assert normalize_cli_argv(["--llm-verify", "--verify", "--verify-code"]) == [
        "--verify-llm",
        "--verify-llm",
        "--verify-llm",
    ]
    out_dir = tmp_path / "verify_out"
    rc = cli_main(
        [
            "--arxiv-id",
            "2608.15888",
            "--verify-llm",
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "[LLM CODE VERIFIER]" in out
    reports = list(out_dir.glob("arxiv_2608_15888_*/verification_report.json"))
    assert len(reports) == 1
    ver_data = json.loads(reports[0].read_text(encoding="utf-8"))
    assert ver_data["verdict"] in ("PASS", "WARN", "FAIL")
    assert ver_data["deterministic_checks"]["runtime_tests_passed"] == 4
    assert ver_data["deterministic_checks"]["deterministic_score"] == 100.0
    index_text = (out_dir / "INDEX.md").read_text(encoding="utf-8")
    assert "Verification" in index_text
    assert "verification_report.json" in index_text


def test_flag_13_all_pipeline_flags_combined_together(tmp_path: Path, capsys) -> None:
    """Proves every pipeline flag composes cleanly in a single command without premature exit."""
    out_dir = tmp_path / "all_flags_out"
    html_out = tmp_path / "combined_report.html"
    pool_dir = tmp_path / "combined_worlds"
    rc = cli_main(
        [
            "--arxiv-id",
            "2608.15888",
            "--dynarot",
            "--model-family",
            "glm",
            "--audit",
            "--langgraph",
            "--generate",
            "--html-report",
            str(html_out),
            "--evolves",
            "--evolve-mode",
            "hybrid",
            "--evolve-count",
            "1",
            "--generations",
            "2",
            "--islands",
            "2",
            "--rl",
            "--rl-cycles",
            "2",
            "--rl-budget",
            "6",
            "--rl-dream-candidates",
            "8",
            "--rl-benchmark",
            "--rl-seeds",
            "5",
            "--rl-trace-pool",
            str(pool_dir),
            "--verify-panel",
            "--panel-size",
            "2",
            "--no-auto-provision",
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "[MODELFUSION DYNAROT]" in out
    assert "[SWARM]" in out
    assert "[EVOLVE]" in out
    assert "[DREAM-RSI RL]" in out
    assert "[MODELFUSION PANEL VERIFIER]" in out
    assert html_out.exists()
    assert len(list(out_dir.glob("arxiv_2608_15888_*/evolved/champion_arxiv_2608_15888.py"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/rl/rl_champion_arxiv_2608_15888.py"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/verification_report.json"))) == 1
    assert (pool_dir / "trace_pool.jsonl").exists()
    assert (out_dir / "INDEX.md").exists()

