"""Tests for CLI flags (--evolve / --evolves / --evoves, --dynarot, --arxiv-id) and MCP Server tools."""

import asyncio
import json
from pathlib import Path

from arxiv_to_code.cli import build_arg_parser, main as cli_main
from arxiv_to_code.mcp_server import (
    arxiv_to_code_status,
    audit_defense_coverage,
    build_mcp_arg_parser,
    dynarot_hardware_route_and_install,
    evolve_python_code,
    extract_document_langextract,
    generate_python_code_from_paper,
    main as mcp_main,
    run_langgraph_arxiv_to_code_swarm,
    search_and_rank_arxiv_papers,
    verify_generated_python_code,
)


def test_cli_and_mcp_help_documentation(capsys) -> None:
    cli_help = build_arg_parser().format_help()
    assert "1. arXiv Discovery & Multi-Factor Ranking" in cli_help
    assert "2. Google LangExtract 3-Class Source Grounding" in cli_help
    assert "4. ModelFusion DynaRot Hardware Sizing" in cli_help
    assert "5. Combined ShinkaEvolve + OpenEvolve Evolutionary Search" in cli_help
    assert "--evolves, --evolve, --evoves" in cli_help
    assert "--search QUERY, --searct QUERY" in cli_help
    assert "--install-models" in cli_help

    mcp_help = build_mcp_arg_parser().format_help()
    assert "arxiv_to_code_status" in mcp_help
    assert "dynarot_hardware_route_and_install" in mcp_help
    assert "evolve_python_code" in mcp_help

    mcp_main(["--list-tools"])
    captured = capsys.readouterr().out
    catalog = json.loads(captured)
    assert catalog["total_tools"] == 9


def test_cli_flags_evolve_and_dynarot(tmp_path: Path) -> None:
    # 1. Test --dynarot flag
    rc_dyn = cli_main(["--dynarot"])
    assert rc_dyn == 0

    # 2. Test --arxiv-id with --evolves, --evolve-count 2, and organized --output-dir
    out_dir = tmp_path / "cli_out"
    rc_evo = cli_main(
        [
            "--arxiv-id",
            "2608.15888",
            "--evolves",
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
    assert rc_evo == 0
    assert (out_dir / "INDEX.md").exists()
    assert (out_dir / "MANIFEST.json").exists()
    assert len(list(out_dir.glob("arxiv_2608_15888_*/src/arxiv_2608_15888_module.py"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/evolved/champion_arxiv_2608_15888.py"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/langextract_report.html"))) == 1
    assert len(list(out_dir.glob("arxiv_2608_15888_*/extraction_results.jsonl"))) == 1

    # 3. Test --searct, ---evolves (triple-dash), --evolve-mode hybrid, --status, --list-tools, --audit
    rc_typo_flags = cli_main(
        [
            "--searct",
            "container sandbox escape",
            "--max-results",
            "1",
            "---evolves",
            "--evolve-mode",
            "hybrid",
            "--generations",
            "2",
            "--output-dir",
            str(tmp_path / "typo_out"),
            "--no-progress",
        ]
    )
    assert rc_typo_flags == 0

    assert cli_main(["--status"]) == 0
    assert cli_main(["--list-tools"]) == 0
    assert cli_main(["--audit", "--arxiv-id", "2603.02277"]) == 0


def test_all_8_mcp_tools(tmp_path: Path) -> None:
    # Tool 1: arxiv_to_code_status
    s_raw = asyncio.run(arxiv_to_code_status())
    s_data = json.loads(s_raw)
    assert s_data["status"] == "operational"
    assert s_data["target_code_generation_language"] == "python"

    # Tool 2: dynarot_hardware_route_and_install
    d_raw = asyncio.run(
        dynarot_hardware_route_and_install(
            override_free_ram_gb=25.0,
            override_free_vram_mb=8000.0,
            auto_install_models=False,
        )
    )
    d_data = json.loads(d_raw)
    assert d_data["primary_model"] == "qwen2.5:14b"
    assert d_data["shinka_heavy_reasoning_model"] == "qwen2.5:14b"
    assert d_data["openevolve_fast_mutation_model"] == "qwen2.5:3b"

    # Tool 3: search_and_rank_arxiv_papers
    sr_raw = asyncio.run(
        search_and_rank_arxiv_papers(
            query="container sandbox escape",
            max_results=2,
            evolve=True,
            evolve_mode="sequential",
        )
    )
    sr_data = json.loads(sr_raw)
    assert sr_data["total_ranked"] >= 1
    assert "generated_python_code" in sr_data["ranked_articles"][0]
    assert "evolution_report" in sr_data["ranked_articles"][0]

    # Tool 4: extract_document_langextract
    html_out = tmp_path / "mcp_langextract.html"
    ex_raw = asyncio.run(
        extract_document_langextract(
            arxiv_id_or_path="2603.02277",
            output_html_path=str(html_out),
        )
    )
    ex_data = json.loads(ex_raw)
    assert ex_data["total_grounded_spans"] >= 3
    assert html_out.exists()

    # Tool 5: generate_python_code_from_paper
    gen_raw = asyncio.run(
        generate_python_code_from_paper(
            arxiv_id="2608.15888",
            output_dir=str(tmp_path / "mcp_gen"),
            evolve=True,
            evolve_mode="hybrid",
            generations=2,
        )
    )
    gen_data = json.loads(gen_raw)
    assert gen_data["generated_python_code"]["language"] == "python"
    assert gen_data["evolution_report"]["champion_metrics"]["tests_passed"] == 4

    # Tool 6: evolve_python_code
    evo_raw = asyncio.run(
        evolve_python_code(
            arxiv_id="2603.02277",
            evolve_mode="parallel",
            generations=2,
            num_islands=2,
            elites_dir=str(tmp_path / "mcp_elites"),
        )
    )
    evo_data = json.loads(evo_raw)
    assert evo_data["evolve_mode"] == "parallel"
    assert evo_data["champion_metrics"]["tests_passed"] == 4

    # Tool 7: audit_defense_coverage
    aud_raw = asyncio.run(audit_defense_coverage(arxiv_id="2603.02277"))
    aud_data = json.loads(aud_raw)
    assert aud_data["success"] is True
    assert aud_data["overall_coverage_percentage"] == 100.0

    # Tool 8: run_langgraph_arxiv_to_code_swarm
    sw_raw = asyncio.run(run_langgraph_arxiv_to_code_swarm(arxiv_id="2603.02277"))
    sw_data = json.loads(sw_raw)
    assert sw_data["stopping_reason"] == "Converged_PostEvolutionVerified_100PercentCoverage"
    assert sw_data["verification_completed"] is True

    # Tool 9: verify_generated_python_code
    ver_raw = asyncio.run(verify_generated_python_code(arxiv_id="2608.15888", enable_llm=False))
    ver_data = json.loads(ver_raw)
    assert ver_data["verdict"] == "PASS"
    assert ver_data["deterministic_checks"]["runtime_tests_passed"] == 4
