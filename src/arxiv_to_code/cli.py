#!/usr/bin/env python3
"""
src/arxiv_to_code/cli.py
======================================================================================
Unified All-in-One CLI Engine for `arixv_to_code.py`:
Every operation in the repository (arXiv search, Google LangExtract document extraction,
HTML visualization, Paper-to-Python code generation, ModelFusion DynaRot hardware sizing,
automatic Ollama model installation, combined ShinkaEvolve + OpenEvolve evolutionary search,
4-Arm LangGraph swarm, multi-target defense coverage audit, MCP server startup, MCP tool
catalog listing, 8-step self-test, and pytest execution) is controlled directly via flags
on `python arixv_to_code.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .dynarot import DynaRotModelRouter, size_models_by_available_memory
from .evolve_engine import evaluate_python_candidate, run_combined_shinka_openevolve
from .langextract_engine import (
    audit_defense_coverage_multi_target,
    compute_article_rank,
    extract_grounded_breakouts,
    extract_text_from_document_path,
    fetch_paper_by_id,
    generate_interactive_html,
    normalize_arxiv_id,
    search_and_rank_articles,
)
from .llm_verifier import verify_generated_python_code
from .mcp_server import arxiv_to_code_status, main as run_mcp_server
from .ollama_provisioner import (
    dynamically_provision_ollama_and_models,
    ensure_models_for_pipeline,
)
from .panel_verifier import run_model_panel_verification
from .progress import create_progress_bar, spinner
from .python_code_generator import synthesize_production_python_code
from .swarm import run_langgraph_research_swarm
from .workspace_layout import prepare_paper_workspace, write_organized_paper_artifacts

CLI_EPILOG = """
================================================================================
MODELFUSION DYNAROT DYNAMIC AVAILABLE MEMORY LAW (ANTI-OOM SIZING)
================================================================================
  Tier 1 (Free RAM >= 48 GB or Free VRAM >= 24 GB) : Primary qwen2.5:32b | Shinka Heavy qwen2.5:32b | OpenEvolve Fast qwen2.5:7b
  Tier 2 (Free RAM >= 24 GB or Free VRAM >= 14 GB) : Primary qwen2.5:14b | Shinka Heavy qwen2.5:14b | OpenEvolve Fast qwen2.5:3b
  Tier 3 (Free RAM >= 12 GB or Free VRAM >= 4.5 GB): Primary qwen2.5:7b  | Shinka Heavy qwen2.5:7b  | OpenEvolve Fast qwen2.5:1.5b
  Tier 4 (Free RAM >= 6 GB  or Free VRAM >= 2 GB)  : Primary qwen2.5:3b  | Shinka Heavy qwen2.5:3b  | OpenEvolve Fast qwen2.5:1.5b
  Tier 5 (Free RAM <  6 GB  and Free VRAM < 2 GB)  : Primary qwen2.5:1.5b| Shinka Heavy qwen2.5:1.5b| OpenEvolve Fast qwen2.5:0.5b

================================================================================
COMBINED SHINKAEVOLVE + OPENEVOLVE PAIRING MODES (--evolve / --evolves)
================================================================================
  * sequential : Stage 1 ShinkaEvolve (UCB1 bandit routing, Meta-Scratchpad, AST Novelty Rejection)
                 discovers structural algorithmic breakthroughs and seeds its top elites into
                 Stage 2 OpenEvolve (Multi-Island 2D MAP-Elites grid trading off memory vs speed).
  * parallel   : Cross-pollinating ensemble running ShinkaEvolve (heavy reasoning tier) and
                 OpenEvolve (fast lightweight mutation tier) concurrently over a shared filesystem
                 elites directory (--elites-dir).
  * hybrid     : Executes the two-stage sequential discovery-to-MAP-Elites pipeline followed by
                 a parallel cross-pollinating ensemble pass (Default).

================================================================================
EXECUTABLE CLI EXAMPLES (ALL FLAGS VIA python arixv_to_code.py)
================================================================================
  # 1. Run 8-step automated end-to-end self-test suite
  python arixv_to_code.py --test

  # 2. Run full Pytest suite directly via flag
  python arixv_to_code.py --pytest

  # 3. Probe live RAM/VRAM, Pareto knee-point model, and Shinka/OpenEvolve compute split
  python arixv_to_code.py --dynarot

  # 4. Auto-install Ollama (if missing) & dynamically pull resource-fitted models (e.g., glm4:9b + deepseek-r1:7b)
  python arixv_to_code.py --install-models
  python arixv_to_code.py --install-models --panel-models "glm4:9b,deepseek-r1:7b"

  # 5. Search arXiv, extract 3-class Google LangExtract spans, and generate Python code
  python arixv_to_code.py --search "LLM container sandbox escape" --max-results 3 --output-dir ./generated

  # 6. Search arXiv + evolve synthesized Python code in one command
  python arixv_to_code.py --search "multi-agent delegation security" --evolves --evolve-mode hybrid

  # 7. Synthesize Python code for arXiv:2608.15888 and evolve with ShinkaEvolve + OpenEvolve
  python arixv_to_code.py --arxiv-id 2608.15888 --evolves --evolve-mode hybrid --output-dir ./generated

  # 8. Extract a local PDF/LaTeX/Markdown document and write interactive LangExtract HTML
  python arixv_to_code.py --extract-doc ./paper.pdf --html-report ./audit.html --evolves

  # 9. Audit multi-target defense coverage for an arXiv paper
  python arixv_to_code.py --arxiv-id 2603.02277 --audit

  # 10. Run 4-Arm HMAC-SHA256 signed LangGraph Research & Evolutionary Swarm
  python arixv_to_code.py --arxiv-id 2603.02277 --langgraph

  # 11. Generate, evolve (x2 rounds), and verify generated Python code using a local open-weight LLM
  python arixv_to_code.py --search "LLM container sandbox escape" --max-results 2 --evolves --evolve-count 2 --verify-llm

  # 12. List all 9 registered MCP tools or start the stdio JSON-RPC MCP server
  python arixv_to_code.py --list-tools
  python arixv_to_code.py --mcp
"""


def normalize_cli_argv(argv: Optional[List[str]]) -> List[str]:
    """
    Normalizes CLI arguments for typo & multi-dash resiliency:
    - Converts triple-or-more leading dashes (`---evolves` -> `--evolves`)
    - Maps natural typo variants (`--searct`, `--seach` -> `--search`, `--evoves` -> `--evolves`)
    """
    raw_list = list(sys.argv[1:] if argv is None else argv)
    normalized: List[str] = []
    alias_map = {
        "--searct": "--search",
        "--seach": "--search",
        "--serach": "--search",
        "--evoves": "--evolves",
        "--evlove": "--evolves",
        "--evolve": "--evolves",
        "--extract": "--extract-doc",
        "--doc": "--extract-doc",
        "--id": "--arxiv-id",
        "--paper-id": "--arxiv-id",
        "--mcp-tools": "--list-tools",
        "--audit-defense": "--audit",
        "--swarm": "--langgraph",
        "--llm-verify": "--verify-llm",
        "--verify": "--verify-llm",
        "--verify-code": "--verify-llm",
        "--panel": "--verify-panel",
        "--fusion-verify": "--verify-panel",
        "--panel-verify": "--verify-panel",
        "--provision": "--install-models",
        "--install-ollama": "--install-models",
        "--pull-models": "--install-models",
    }
    for token in raw_list:
        if token.startswith("---"):
             stripped = "--" + token.lstrip("-")
        else:
            stripped = token
        if "=" in stripped and stripped.startswith("--"):
            flag_part, val_part = stripped.split("=", 1)
            mapped_flag = alias_map.get(flag_part.lower(), flag_part)
            normalized.append(f"{mapped_flag}={val_part}")
        else:
            mapped = alias_map.get(stripped.lower(), stripped)
            normalized.append(mapped)
    return normalized


def run_self_test() -> bool:
    """
    Comprehensive 9-step end-to-end functional self-test verifying:
    1. arXiv discovery & strict novelty filtering
    2. Google LangExtract 3-class character-interval source grounding
    3. Multi-factor 0-100 ranking & code repo bonus
    4. Interactive HTML report generation (`langextract.visualize`)
    5. ModelFusion DynaRot Dynamic Available Memory Law & Compute Budget Split
    6. Paper-to-Python production code generation & deterministic execution
    7. Combined ShinkaEvolve + OpenEvolve (`--evolves`) sequential & parallel cross-pollination
    8. 4-Arm HMAC-SHA256 signed LangGraph Research & Evolutionary Swarm
    9. Deterministic + LLM-as-a-Judge Generated Code Verification Gate
    """
    print("================================================================================")
    print("[ARIXV-TO-CODE] 9-STEP END-TO-END FUNCTIONAL SELF-TEST SUITE")
    print("================================================================================")

    # 1. Test arXiv discovery & novelty-filtered ranking
    print("[1/9] Testing arXiv discovery & strict novelty-filtered article ranking...")
    ranked = search_and_rank_articles(
        query="LLM container sandbox escape jailbreak",
        max_results=3,
        only_novel_enhancements=True,
    )
    assert len(ranked) >= 1, "Expected at least 1 ranked paper"
    scores = [a["rank_score"] for a in ranked]
    assert scores == sorted(scores, reverse=True), f"Articles not sorted descending: {scores}"
    for art in ranked:
        assert art["is_novel_to_codebase"] is True
        assert art["enhances_codebase"] is True
        print(
            f"  [RANK #{art['rank_position']}] Score: {art['rank_score']}/100 ({art['priority_tier']}) | "
            f"{art['title'][:52]}... (arXiv:{art['arxiv_id']})"
        )

    # 2. Test 3-class Google LangExtract source grounding on arXiv:2603.02277
    print("[2/9] Testing Google LangExtract 3-class character-span grounding on arXiv:2603.02277...")
    paper = fetch_paper_by_id("2603.02277")
    assert paper is not None
    extractions, annotated_doc = extract_grounded_breakouts(paper)
    rank_data = compute_article_rank(paper, extractions)
    assert rank_data["total_score"] >= 80.0
    for ext in extractions:
        start = ext["source_char_start"]
        end = ext["source_char_end"]
        quote = ext["source_quote"]
        assert paper["summary"][start:end] == quote, "Character-interval grounding mismatch!"
    print(f"  [PASS] Verified {len(extractions)} exact character-interval spans (Score: {rank_data['total_score']}/100)")

    # 3. Verify strict novelty rejection of already-implemented papers (2610.09167)
    print("[3/9] Verifying strict novelty rejection of already-implemented paper (arXiv:2610.09167)...")
    impl_paper = fetch_paper_by_id("2610.09167")
    assert impl_paper is not None
    impl_exts, _ = extract_grounded_breakouts(impl_paper)
    impl_rank = compute_article_rank(impl_paper, impl_exts)
    assert impl_rank["is_novel_to_codebase"] is False
    assert impl_rank["novelty_delta_score"] == 0.0
    print("  [PASS] Correctly rejected already-implemented paper 2610.09167 (Novelty Delta: 0.0 pts)")

    # 4. Test Interactive HTML Visualization
    print("[4/9] Testing Google LangExtract interactive HTML report generation...")
    test_html = Path("test_arxiv_to_code_audit.html")
    generate_interactive_html(annotated_doc, str(test_html))
    assert test_html.exists() and test_html.stat().st_size > 250
    test_html.unlink()
    print("  [PASS] Generated and verified interactive LangExtract HTML visualization")

    # 5. Test ModelFusion DynaRot Dynamic Available Memory Law across all 5 tiers
    print("[5/9] Testing ModelFusion DynaRot hardware sizing & compute budget split...")
    _, p1, s1, o1, _, _ = size_models_by_available_memory(free_ram_gb=64.0, free_vram_mb=24000.0)
    assert p1 == "qwen2.5:32b" and s1 == "qwen2.5:32b" and o1 == "qwen2.5:7b"
    _, p2, s2, o2, _, _ = size_models_by_available_memory(free_ram_gb=28.0, free_vram_mb=15000.0)
    assert p2 == "qwen2.5:14b" and s2 == "qwen2.5:14b" and o2 == "qwen2.5:3b"
    _, p3, s3, o3, _, _ = size_models_by_available_memory(free_ram_gb=14.0, free_vram_mb=5000.0)
    assert p3 == "qwen2.5:7b" and s3 == "qwen2.5:7b" and o3 == "qwen2.5:1.5b"
    _, p4, s4, o4, _, _ = size_models_by_available_memory(free_ram_gb=8.0, free_vram_mb=2500.0)
    assert p4 == "qwen2.5:3b" and s4 == "qwen2.5:3b" and o4 == "qwen2.5:1.5b"
    _, p5, s5, o5, _, _ = size_models_by_available_memory(free_ram_gb=4.0, free_vram_mb=500.0)
    assert p5 == "qwen2.5:1.5b" and s5 == "qwen2.5:1.5b" and o5 == "qwen2.5:0.5b"

    router = DynaRotModelRouter()
    live_decision = router.route()
    assert "-coder" not in live_decision.primary_model
    print(
        f"  [PASS] Live DynaRot Tier: {live_decision.hardware_tier} | Primary: {live_decision.primary_model} | "
        f"Shinka Heavy: {live_decision.shinka_heavy_reasoning_model} | OpenEvolve Fast: {live_decision.openevolve_fast_mutation_model}"
    )

    # 6. Test Paper-to-Python Code Synthesis & Deterministic Execution
    print("[6/9] Testing Paper-to-Python production code generation & deterministic benchmark...")
    apc_paper = fetch_paper_by_id("2608.15888")
    assert apc_paper is not None
    apc_exts, _ = extract_grounded_breakouts(apc_paper)
    py_bundle = synthesize_production_python_code(apc_paper, apc_exts)
    py_src = py_bundle["production_python_code"]
    assert "# EVOLVE-BLOCK-START" in py_src and "# EVOLVE-BLOCK-END" in py_src
    metrics = evaluate_python_candidate(py_src, candidate_id="test-2608-15888")
    assert metrics.error is None, f"Generated Python code failed evaluation: {metrics.error}"
    assert metrics.tests_passed == metrics.tests_total == 4
    print(
        f"  [PASS] Generated Python module passed {metrics.tests_passed}/{metrics.tests_total} tests "
        f"(Fitness: {metrics.total_fitness}/100, Speed: {metrics.execution_time_us} us, Memory: {metrics.peak_memory_kib} KiB)"
    )

    # 7. Test Combined ShinkaEvolve + OpenEvolve (--evolves) Pipeline
    print("[7/9] Testing Combined ShinkaEvolve + OpenEvolve evolutionary search (Sequential & Parallel)...")
    evo_report = run_combined_shinka_openevolve(
        seed_python_code=py_src,
        arxiv_id="2608.15888",
        mode="hybrid",
        generations=4,
        num_islands=3,
        dynarot_decision=live_decision,
    )
    assert evo_report["evolve_block_convention_verified"] is True
    assert evo_report["champion_metrics"]["tests_passed"] == evo_report["champion_metrics"]["tests_total"]
    assert evo_report["champion_metrics"]["total_fitness"] >= evo_report["baseline_seed_metrics"]["total_fitness"]
    assert len(evo_report["cross_pollination_events"]) >= 2
    assert evo_report["openevolve_map_elites_summary"]["occupied_niches_count"] >= 1
    print(
        f"  [PASS] Evolved Champion Fitness: {evo_report['baseline_seed_metrics']['total_fitness']} -> "
        f"{evo_report['champion_metrics']['total_fitness']} "
        f"(Speedup: {evo_report['speedup_ratio']}x, Occupied MAP-Elites Niches: "
        f"{evo_report['openevolve_map_elites_summary']['occupied_niches_count']})"
    )

    # 8. Test 4-Arm HMAC-SHA256 Signed Swarm
    print("[8/9] Testing 4-Arm HMAC-SHA256 Signed ArXiv-to-Code Swarm...")
    swarm_state = run_langgraph_research_swarm(
        query="LLM agent sandbox escape breakout",
        arxiv_id="2603.02277",
        only_novel_enhancements=True,
    )
    assert swarm_state["stopping_reason"] == "Converged_PostEvolutionVerified_100PercentCoverage"
    assert swarm_state["evolution_completed"] is True
    assert swarm_state["verification_completed"] is True
    assert all(t["hmac_verified"] is True for t in swarm_state["edge_transitions"])
    print(
        f"  [PASS] 4-Arm Swarm converged across {swarm_state['visited_nodes']} -> {swarm_state['stopping_reason']}"
    )

    # 9. Test LLM + Deterministic Generated Code Verification Gate
    print("[9/9] Testing Deterministic + LLM Code Verification Gate (valid pass & dangerous code fail-closed)...")
    ver_ok = verify_generated_python_code(
        source_code=evo_report["champion_candidate"]["source_code"],
        paper=apc_paper,
        extractions=apc_exts,
        code_stage="evolved_champion",
        dynarot_decision=live_decision,
        enable_llm=False,
    )
    assert ver_ok.verdict == "PASS" and ver_ok.overall_score >= 95.0
    ver_bad = verify_generated_python_code(
        source_code="import os\nos.system('whoami')\n",
        paper=apc_paper,
        extractions=apc_exts,
        code_stage="custom",
        dynarot_decision=live_decision,
        enable_llm=False,
    )
    assert ver_bad.verdict == "FAIL" and ver_bad.deterministic_checks["zero_dangerous_calls"] is False
    print(
        f"  [PASS] Code Verifier PASS ({ver_ok.overall_score}/100) on valid champion & FAIL-CLOSED on dangerous os.system()"
    )

    print("================================================================================")
    print("[PASS] ALL 9 ARIXV-TO-CODE, LANGEXTRACT, DYNAROT, EVOLVE & LLM VERIFIER TESTS PASSED!")
    print("================================================================================")
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arixv_to_code.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "ArXiv-to-Code All-in-One CLI & Universal MCP Server (arixv_to_code.py v1.0.0)\n"
            "--------------------------------------------------------------------------------\n"
            "Autonomous arXiv Research Discovery, Google LangExtract 3-Class Source Grounding,\n"
            "Zero-Mock Paper-to-Python Code Synthesis (# EVOLVE-BLOCK Convention), ModelFusion\n"
            "DynaRot Hardware-Aware Model Routing, and Combined ShinkaEvolve + OpenEvolve\n"
            "Evolutionary Optimization."
        ),
        epilog=CLI_EPILOG,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="arixv_to_code.py 1.0.0",
        help="Show program's version number and exit",
    )

    g_search = parser.add_argument_group("1. arXiv Discovery & Multi-Factor Ranking (0-100 Score)")
    g_search.add_argument(
        "--search",
        "--searct",
        "--seach",
        "--query",
        dest="query",
        type=str,
        default="LLM agent sandbox escape breakout",
        help="Search query for discovering and ranking arXiv papers (default: 'LLM agent sandbox escape breakout')",
    )
    g_search.add_argument(
        "--arxiv-id",
        "--id",
        dest="arxiv_id",
        type=str,
        default=None,
        help="Specific arXiv paper ID to fetch, ground, and synthesize into Python (e.g., 2603.02277, 2608.15888, 2609.00005)",
    )
    g_search.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Maximum number of ranked novel arXiv papers to retrieve (default: 3)",
    )
    g_search.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum composite rank score (0.0-100.0) filter threshold (default: 0.0)",
    )
    g_search.add_argument(
        "--include-implemented",
        action="store_true",
        help="Disable strict novelty filtering and include papers already in CODEBASE_IMPLEMENTED_PAPERS",
    )

    g_extract = parser.add_argument_group("2. Google LangExtract 3-Class Source Grounding & HTML Reports")
    g_extract.add_argument(
        "--extract-doc",
        "--extract",
        dest="extract_doc",
        type=str,
        default=None,
        help="Path to a local document (.pdf, .txt, .md, .tex) to extract with Google LangExtract and generate Python code",
    )
    g_extract.add_argument(
        "--html-report",
        type=str,
        default=None,
        help="Write self-contained interactive Google LangExtract HTML visualization (langextract.visualize) to this path",
    )
    g_extract.add_argument(
        "--langextract-llm",
        action="store_true",
        help="Invoke Google langextract.extract() LLM-backed 3-class grounding (via local Ollama or GEMINI_API_KEY) in addition to deterministic grounding",
    )
    g_extract.add_argument(
        "--langextract-model",
        type=str,
        default=None,
        help="Optional model ID for Google langextract.extract() (e.g., 'qwen2.5:7b' for Ollama or 'gemini-2.5-flash' for Gemini)",
    )

    g_codegen = parser.add_argument_group("3. Paper-to-Python Production Code Synthesis & Defense Audit")
    g_codegen.add_argument(
        "--generate",
        action="store_true",
        help="Synthesize runnable production Python module with # EVOLVE-BLOCK markers (enabled automatically on search/id/extract)",
    )
    g_codegen.add_argument(
        "--output-dir",
        type=str,
        default="generated",
        help="Root output directory organized into per-paper folders with INDEX.md, MANIFEST.json, src/, eval/, evolved/, and LangExtract HTML/JSONL (default: 'generated')",
    )
    g_codegen.add_argument(
        "--audit",
        "--audit-defense",
        dest="audit",
        action="store_true",
        help="Run multi-target defense & invariant coverage audit across Python runtime, OshoosiClaw, KillSwitch, and AgentOctopus",
    )
    g_codegen.add_argument(
        "--verify-llm",
        "--llm-verify",
        "--verify",
        dest="verify_llm",
        action="store_true",
        help=(
            "Verify generated (and evolved) Python code using a local open-weight LLM judge (Ollama via DynaRot) "
            "combined with fail-closed AST, security invariant, and runtime unit-test checks; saves verification_report.json"
        ),
    )
    g_codegen.add_argument(
        "--verify-model",
        type=str,
        default=None,
        help="Optional LLM model tag for --verify-llm (e.g., 'glm-4.7-flash', 'glm4:9b', 'qwen2.5:7b', 'deepseek-r1:7b')",
    )
    g_codegen.add_argument(
        "--verify-panel",
        "--panel",
        "--fusion-verify",
        dest="verify_panel",
        action="store_true",
        help=(
            "Run ModelFusion Panel-of-Models verification (GLM + Qwen + DeepSeek + Gemma in memory-safe batches) "
            "followed by minority-preserving Judge Analysis + Layer-1 deterministic gate"
        ),
    )
    g_codegen.add_argument(
        "--panel-size",
        type=int,
        default=None,
        help="Number of panelists for --verify-panel (default: auto-derived from live free RAM/VRAM via derive_fusion_model_count)",
    )
    g_codegen.add_argument(
        "--panel-models",
        type=str,
        default=None,
        help="Optional comma-separated model tags for --verify-panel (e.g., 'glm-4.7-flash,qwen2.5:7b,deepseek-r1:7b')",
    )
    g_codegen.add_argument(
        "--panel-mode",
        type=str,
        choices=["multi-model", "multi-sample"],
        default="multi-model",
        help="ModelFusion panel mode: 'multi-model' (distinct open-weight families, default) or 'multi-sample' (diverse temperatures/personas)",
    )

    g_dynarot = parser.add_argument_group("4. ModelFusion DynaRot Hardware Sizing & Auto-Provisioning")
    g_dynarot.add_argument(
        "--dynarot",
        "--sys-info",
        dest="dynarot",
        action="store_true",
        help="Probe live free RAM & free VRAM, compute 4-objective Pareto front & knee-point, and show Shinka/OpenEvolve compute split",
    )
    g_dynarot.add_argument(
        "--install-models",
        "--provision",
        "--install-ollama",
        dest="install_models",
        action="store_true",
        help=(
            "Auto-install Ollama (if missing), start `ollama serve` detached, dynamically select "
            "resource-fitted open-weight panel models (e.g., `glm4:9b` + `deepseek-r1:7b`), and "
            "pull them via streaming `/api/pull` with live tqdm byte progress"
        ),
    )
    g_dynarot.add_argument(
        "--auto-provision",
        action="store_true",
        help=(
            "On-demand auto-provisioning guard for --verify-panel, --verify-llm, and --langextract-llm: "
            "automatically installs/starts Ollama and pulls missing resource-fitted models before execution"
        ),
    )
    g_dynarot.add_argument(
        "--model-family",
        type=str,
        choices=["auto", "glm", "qwen", "gemma"],
        default="auto",
        help="Preferred open-weight Ollama model family calibrated to available RAM/VRAM/disk (QuorumRAG AutoFusion parity: auto | glm | qwen | gemma; default: auto)",
    )

    g_evolve = parser.add_argument_group("5. Combined ShinkaEvolve + OpenEvolve Evolutionary Search")
    g_evolve.add_argument(
        "--evolves",
        "--evolve",
        "--evoves",
        dest="evolve",
        action="store_true",
        help=(
            "Evolve synthesized Python code by combining ShinkaEvolve (UCB1 routing, Meta-Scratchpad, "
            "AST Novelty Rejection) and OpenEvolve (Multi-Island 2D MAP-Elites grid)"
        ),
    )
    g_evolve.add_argument(
        "--evolve-mode",
        type=str,
        choices=["sequential", "parallel", "hybrid"],
        default="hybrid",
        help="Pairing mode for ShinkaEvolve + OpenEvolve: 'sequential', 'parallel', or 'hybrid' (default: hybrid)",
    )
    g_evolve.add_argument(
        "--evolve-count",
        type=int,
        default=1,
        help="Number of full evolutionary rounds/cycles to run OpenEvolve + ShinkaEvolve (default: 1)",
    )
    g_evolve.add_argument(
        "--generations",
        type=int,
        default=4,
        help="Number of evolutionary generations per stage inside each round (default: 4)",
    )
    g_evolve.add_argument(
        "--islands",
        type=int,
        default=3,
        help="Number of OpenEvolve MAP-Elites islands for behavioral niche exploration (default: 3)",
    )
    g_evolve.add_argument(
        "--elites-dir",
        type=str,
        default=None,
        help="Shared filesystem directory for ShinkaEvolve & OpenEvolve cross-pollinated elites",
    )
    g_evolve.add_argument(
        "--evolve-llm",
        action="store_true",
        help="Enable live local Ollama LLM structural synthesis calls during ShinkaEvolve discovery",
    )
    g_evolve.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars and console spinners (for quiet/machine-parsed runs)",
    )

    g_swarm = parser.add_argument_group("6. Agentic Swarm, Universal MCP Server & Test Suites")
    g_swarm.add_argument(
        "--langgraph",
        "--swarm",
        dest="langgraph",
        action="store_true",
        help="Execute the 4-Arm HMAC-SHA256 signed ArXiv-to-Code StateGraph Research & Evolutionary Swarm",
    )
    g_swarm.add_argument(
        "--status",
        action="store_true",
        help="Print full JSON status report of ArXiv-to-Code, LangExtract, DynaRot, and ShinkaEvolve/OpenEvolve",
    )
    g_swarm.add_argument(
        "--list-tools",
        "--mcp-tools",
        dest="list_tools",
        action="store_true",
        help="Print JSON catalog of all 9 registered Universal MCP Server tools and exit",
    )
    g_swarm.add_argument(
        "--mcp",
        action="store_true",
        help="Start the ArXiv-to-Code Universal MCP Server over stdio JSON-RPC 2.0 (9 Tools)",
    )
    g_swarm.add_argument(
        "--test",
        action="store_true",
        help="Run the 9-step automated end-to-end functional self-test suite",
    )
    g_swarm.add_argument(
        "--pytest",
        action="store_true",
        help="Run the full Pytest unit and integration test suite (tests/)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    clean_argv = normalize_cli_argv(argv)
    parser = build_arg_parser()
    args = parser.parse_args(clean_argv)

    if args.list_tools:
        run_mcp_server(["--list-tools"])
        return 0

    if args.mcp:
        run_mcp_server([])
        return 0

    if args.status:
        print(asyncio.run(arxiv_to_code_status()))
        return 0

    if args.pytest:
        repo_root = Path(__file__).resolve().parent.parent.parent
        env = dict(os.environ)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--basetemp", str(repo_root / ".pytest_tmp"), str(repo_root / "tests")],
            cwd=str(repo_root),
            env=env,
        )
        return proc.returncode

    if args.test:
        ok = run_self_test()
        return 0 if ok else 1

    show_ui = not args.no_progress
    router = DynaRotModelRouter()
    explicit_provision_models = (
        [m.strip() for m in args.panel_models.split(",") if m.strip()]
        if args.panel_models
        else ([args.verify_model.strip()] if args.verify_model and args.verify_model.strip() else None)
    )

    if args.install_models:
        print("================================================================================", flush=True)
        print("[OLLAMA PROVISIONER] DYNAMIC OLLAMA AUTO-INSTALL & RESOURCE-BASED MODEL PULL", flush=True)
        print("================================================================================", flush=True)
        prov = dynamically_provision_ollama_and_models(
            model_family=args.model_family,
            explicit_models=explicit_provision_models,
            auto_install_ollama=True,
            auto_pull=True,
            prefer_vram_resident=True,
            show_progress=show_ui,
        )
        print(json.dumps(prov, indent=2), flush=True)
        if not args.arxiv_id and not args.evolve and not args.dynarot and not args.extract_doc and not args.audit and not args.verify_llm and not args.verify_panel:
            return 0
    elif args.auto_provision:
        min_needed = (args.panel_size or 2) if args.verify_panel else 1
        print("================================================================================", flush=True)
        print("[OLLAMA PROVISIONER] ON-DEMAND AUTO-PROVISION CHECK", flush=True)
        print("================================================================================", flush=True)
        auto_res = ensure_models_for_pipeline(
            min_eligible_models=min_needed,
            model_family=args.model_family,
            explicit_models=explicit_provision_models,
            auto_install_ollama=True,
            prefer_vram_resident=True,
            show_progress=show_ui,
        )
        print(
            f"  [PROVISION] Action={auto_res.get('action')} | Online={auto_res.get('ollama_online')} | "
            f"Installed={auto_res.get('installed_models_after', [])}",
            flush=True,
        )

    with spinner("ModelFusion DynaRot: probing live free RAM, GPU VRAM, disk & Pareto knee-point", enabled=show_ui):
        dynarot_decision = router.route(prompt=args.query, model_family=args.model_family)

    if args.dynarot:
        print("================================================================================", flush=True)
        print("[MODELFUSION DYNAROT] HARDWARE SIZING, PARETO FRONT & COMPUTE SPLIT", flush=True)
        print("================================================================================", flush=True)
        print(json.dumps(dynarot_decision.to_dict(), indent=2), flush=True)
        if not args.arxiv_id and not args.evolve and not args.extract_doc and not args.audit and not args.verify_llm and not args.verify_panel:
            return 0

    if args.audit:
        target_id = args.arxiv_id or "2603.02277"
        with spinner(f"Auditing multi-target defense coverage for arXiv:{target_id}", enabled=show_ui):
            audit_res = audit_defense_coverage_multi_target(target_id)
        print(json.dumps(audit_res, indent=2), flush=True)
        if not args.evolve and not args.generate:
            return 0 if audit_res.get("success") else 1

    only_novel = not args.include_implemented

    if args.langgraph:
        print("================================================================================", flush=True)
        print("[SWARM] EXECUTING 4-ARM ARIXV-TO-CODE STATEGRAPH SWARM", flush=True)
        print("================================================================================", flush=True)
        with spinner("Running 4-Arm HMAC-SHA256 Signed LangGraph Research & Evolution Swarm", enabled=show_ui):
            final_state = run_langgraph_research_swarm(
                query=args.query,
                arxiv_id=args.arxiv_id,
                max_results=args.max_results,
                min_score=args.min_score,
                only_novel_enhancements=only_novel,
                evolve_mode=args.evolve_mode,
            )
        print(json.dumps(final_state, indent=2), flush=True)
        return 0

    # Gather target paper(s) with animated console spinner
    if args.extract_doc:
        with spinner(f"Extracting local document '{args.extract_doc}' with Google LangExtract", enabled=show_ui):
            doc_text = extract_text_from_document_path(args.extract_doc)
            doc_path = Path(args.extract_doc)
            paper = {
                "arxiv_id": doc_path.stem,
                "title": f"Extracted Document: {doc_path.name}",
                "published": "2026-01-01",
                "published_year": 2026,
                "url": f"file:///{doc_path.resolve().as_posix()}",
                "pdf_url": "",
                "authors": ["Document Author"],
                "summary": doc_text[:4000],
                "comments": "",
            }
            extractions, annotated_doc = extract_grounded_breakouts(
                paper,
                use_llm_extract=args.langextract_llm,
                llm_model_id=args.langextract_model,
            )
            rank_info = compute_article_rank(paper, extractions)
        if args.html_report:
            with spinner(f"Rendering Google LangExtract interactive HTML -> {args.html_report}", enabled=show_ui):
                saved = generate_interactive_html(annotated_doc, args.html_report)
            print(f"[+] Interactive LangExtract HTML visualization saved to: {saved}", flush=True)
        ranked_articles = [
            {
                "rank_position": 1,
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "published": paper["published"],
                "url": paper["url"],
                "summary": paper.get("summary", ""),
                "rank_score": rank_info["total_score"],
                "priority_tier": rank_info["priority_tier"],
                "is_novel_to_codebase": rank_info["is_novel_to_codebase"],
                "enhances_codebase": rank_info["enhances_codebase"],
                "target_crates_enhanced": rank_info["target_crates_enhanced"],
                "score_breakdown": rank_info["breakdown"],
                "extractions": extractions,
            }
        ]
    elif args.arxiv_id:
        with spinner(f"Fetching arXiv:{args.arxiv_id} & running Google LangExtract 3-class grounding", enabled=show_ui):
            paper = fetch_paper_by_id(args.arxiv_id)
            if not paper:
                print(f"[!] Paper not found for arXiv ID: {args.arxiv_id}", file=sys.stderr, flush=True)
                return 1
            extractions, annotated_doc = extract_grounded_breakouts(
                paper,
                use_llm_extract=args.langextract_llm,
                llm_model_id=args.langextract_model,
            )
            rank_info = compute_article_rank(paper, extractions)
        if args.html_report:
            with spinner(f"Rendering Google LangExtract interactive HTML -> {args.html_report}", enabled=show_ui):
                saved = generate_interactive_html(annotated_doc, args.html_report)
            print(f"[+] Interactive LangExtract HTML visualization saved to: {saved}", flush=True)
        ranked_articles = [
            {
                "rank_position": 1,
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "published": paper.get("published", ""),
                "url": paper.get("url", ""),
                "summary": paper.get("summary", ""),
                "code_repo_url": rank_info.get("code_repo_url", ""),
                "rank_score": rank_info["total_score"],
                "priority_tier": rank_info["priority_tier"],
                "is_novel_to_codebase": rank_info["is_novel_to_codebase"],
                "enhances_codebase": rank_info["enhances_codebase"],
                "target_crates_enhanced": rank_info["target_crates_enhanced"],
                "score_breakdown": rank_info["breakdown"],
                "extractions": extractions,
            }
        ]
    else:
        with spinner(
            f"Searching arXiv for '{args.query}' & grounding spans with Google LangExtract",
            enabled=show_ui,
        ):
            ranked_articles = search_and_rank_articles(
                query=args.query,
                max_results=args.max_results,
                min_score=args.min_score,
                only_novel_enhancements=only_novel,
                use_llm_extract=args.langextract_llm,
                llm_model_id=args.langextract_model,
            )

    print("\n================================================================================", flush=True)
    print(
        f"[DYNAROT] Active Tier: {dynarot_decision.hardware_tier}\n"
        f"          Primary Model: {dynarot_decision.primary_model} | "
        f"ShinkaEvolve (Heavy): {dynarot_decision.shinka_heavy_reasoning_model} | "
        f"OpenEvolve (Fast): {dynarot_decision.openevolve_fast_mutation_model}",
        flush=True,
    )
    print("================================================================================", flush=True)

    output_root = args.output_dir or "generated"
    papers_bar = create_progress_bar(
        total=max(1, len(ranked_articles)),
        desc="ArXiv-to-Code Pipeline",
        unit="paper",
        disable=not show_ui,
    )

    try:
        for item in ranked_articles:
            papers_bar.set_postfix_str(f"arXiv:{item['arxiv_id']} ({item['rank_score']}/100)")
            print(
                f"\nRank #{item['rank_position']} | Score: {item['rank_score']}/100 [{item['priority_tier']}] | "
                f"arXiv:{item['arxiv_id']}",
                flush=True,
            )
            print(f"  Title     : {item['title']}", flush=True)
            for ext in item["extractions"]:
                print(
                    f"  -> [{ext['extraction_class'].upper()}] {ext['attack_vector']} -> "
                    f"'{ext['source_quote']}' [{ext['source_char_start']}:{ext['source_char_end']}]",
                    flush=True,
                )

            paper_paths = prepare_paper_workspace(output_root, item)

            with spinner(f"Synthesizing production Python code for arXiv:{item['arxiv_id']}", enabled=show_ui):
                code_bundle = synthesize_production_python_code(
                    item,
                    item["extractions"],
                    output_dir=None,
                )
            print("\n" + "=" * 80, flush=True)
            print(code_bundle["printable_code_bundle_markdown"], flush=True)
            print("=" * 80, flush=True)

            evo_report = None
            if args.evolve:
                print("\n================================================================================", flush=True)
                print(
                    f"[EVOLVE] Running Combined ShinkaEvolve + OpenEvolve ({args.evolve_mode.upper()} mode, "
                    f"evolve_count={args.evolve_count}, generations={args.generations}, islands={args.islands}) "
                    f"for arXiv:{code_bundle['clean_arxiv_id']}...",
                    flush=True,
                )
                print("================================================================================", flush=True)
                elites_target = args.elites_dir or str(paper_paths["elites_dir"])
                evo_report = run_combined_shinka_openevolve(
                    seed_python_code=code_bundle["production_python_code"],
                    arxiv_id=code_bundle["clean_arxiv_id"],
                    mode=args.evolve_mode,
                    generations=args.generations,
                    num_islands=args.islands,
                    evolve_count=args.evolve_count,
                    elites_dir=elites_target,
                    dynarot_decision=dynarot_decision,
                    show_progress=show_ui,
                    use_ollama_llm=args.evolve_llm,
                )
                print(
                    f"  Baseline Seed Fitness : {evo_report['baseline_seed_metrics']['total_fitness']}/100 "
                    f"({evo_report['baseline_seed_metrics']['execution_time_us']} us, "
                    f"{evo_report['baseline_seed_metrics']['peak_memory_kib']} KiB)",
                    flush=True,
                )
                print(
                    f"  Champion Elite Fitness: {evo_report['champion_metrics']['total_fitness']}/100 "
                    f"({evo_report['champion_metrics']['execution_time_us']} us, "
                    f"{evo_report['champion_metrics']['peak_memory_kib']} KiB)",
                    flush=True,
                )
                print(f"  Evolution Rounds      : {evo_report['evolve_count']}", flush=True)
                print(f"  Speedup Ratio         : {evo_report['speedup_ratio']}x", flush=True)
                print(f"  Champion Origin       : {evo_report['champion_candidate']['engine_origin']}", flush=True)
                print(f"  Champion Operator     : {evo_report['champion_candidate']['mutation_operator']}", flush=True)
                print(
                    f"  ShinkaEvolve UCB1     : {evo_report['shinkaevolve_summary']['ucb1_model_scores']}",
                    flush=True,
                )
                print(
                    f"  MAP-Elites Niches     : {evo_report['openevolve_map_elites_summary']['occupied_niches_count']} "
                    f"occupied niches ({list(evo_report['openevolve_map_elites_summary']['niches'].keys())})",
                    flush=True,
                )
                print("  Cross-Pollination Log :", flush=True)
                for ev in evo_report["cross_pollination_events"]:
                    print(f"    * {ev['direction']} -> {ev}", flush=True)

            verification_report = None
            if args.verify_panel or args.verify_llm:
                target_code = (
                    evo_report["champion_candidate"]["source_code"]
                    if evo_report is not None
                    else code_bundle["production_python_code"]
                )
                code_stage = "evolved_champion" if evo_report is not None else "synthesized"
                print("\n================================================================================", flush=True)
                if args.verify_panel:
                    print(
                        f"[MODELFUSION PANEL VERIFIER] Deliberating {code_stage} Python code for arXiv:{code_bundle['clean_arxiv_id']} ({args.panel_mode})...",
                        flush=True,
                    )
                else:
                    print(
                        f"[LLM CODE VERIFIER] Auditing {code_stage} Python code for arXiv:{code_bundle['clean_arxiv_id']}...",
                        flush=True,
                    )
                print("================================================================================", flush=True)
                if args.verify_panel:
                    explicit_panel = (
                        [m.strip() for m in args.panel_models.split(",") if m.strip()]
                        if args.panel_models
                        else None
                    )
                    with spinner(
                        f"Running ModelFusion Panel-of-Models + Judge Analysis on {code_stage} code",
                        enabled=show_ui,
                    ):
                        ver_obj = run_model_panel_verification(
                            source_code=target_code,
                            paper=item,
                            extractions=item["extractions"],
                            code_stage=code_stage,
                            panel_size=args.panel_size,
                            panel_models=explicit_panel,
                            panel_mode=args.panel_mode,
                            dynarot_decision=dynarot_decision,
                            enable_llm=True,
                        )
                        verification_report = ver_obj.to_dict()
                else:
                    with spinner(
                        f"Verifying {code_stage} code with Deterministic Gate + LLM Judge ({args.verify_model or dynarot_decision.shinka_heavy_reasoning_model})",
                        enabled=show_ui,
                    ):
                        ver_obj = verify_generated_python_code(
                            source_code=target_code,
                            paper=item,
                            extractions=item["extractions"],
                            code_stage=code_stage,
                            model_id=args.verify_model,
                            dynarot_decision=dynarot_decision,
                            enable_llm=True,
                        )
                        verification_report = ver_obj.to_dict()

                det_c = verification_report["deterministic_checks"]
                llm_a = verification_report["llm_audit"]
                print(
                    f"  Final Verdict         : {verification_report['verdict']} ({verification_report['overall_score']}/100)",
                    flush=True,
                )
                print(
                    f"  Verification Mode     : {verification_report['verification_mode']}",
                    flush=True,
                )
                print(
                    f"  Verifier Model        : {verification_report['verifier_model']} "
                    f"(Provider: {verification_report['verifier_provider']} | LLM Executed: {verification_report['llm_verified']})",
                    flush=True,
                )
                print(
                    f"  Deterministic Gate    : AST={det_c['ast_syntax_valid']} | "
                    f"Tests={det_c['runtime_tests_passed']}/{det_c['runtime_tests_total']} | "
                    f"HMAC={det_c['constant_time_hmac_used']} | SafeCalls={det_c['zero_dangerous_calls']}",
                    flush=True,
                )
                panel_a = llm_a.get("panel_audit")
                if panel_a:
                    ja = panel_a.get("judge_analysis", {})
                    print(
                        f"  ModelFusion Panel     : mode={panel_a['panel_mode']} | "
                        f"panelists={panel_a['responded_panelists']}/{panel_a['effective_panel_size']} | "
                        f"batch_size={panel_a['batch_size']} | judge={panel_a['judge_model']} "
                        f"(minority_override={ja.get('minority_override_triggered', False)})",
                        flush=True,
                    )
                    if ja.get("consensus"):
                        print(f"  Judge Consensus       : {ja['consensus'][0]}", flush=True)
                    if ja.get("disagreements"):
                        print(f"  Judge Disagreements   : {ja['disagreements']}", flush=True)
                    if ja.get("risk_flags"):
                        print(f"  Judge Risk Flags      : {ja['risk_flags'][:3]}", flush=True)
                if verification_report["llm_verified"]:
                    print(
                        f"  LLM Judge Scores      : Paper Fidelity={llm_a['paper_fidelity_score']}/100 | "
                        f"Correctness={llm_a['correctness_score']}/100 | Security={llm_a['security_score']}/100",
                        flush=True,
                    )
                    print(f"  LLM Audit Summary     : {llm_a['summary']}", flush=True)
                else:
                    print(f"  LLM Status            : {llm_a['status']} — {llm_a['summary']}", flush=True)
                if verification_report["findings"]:
                    print("  Findings              :", flush=True)
                    for fnd in verification_report["findings"][:6]:
                        print(
                            f"    * [{fnd['severity']}] ({fnd['category']}) {fnd['message']}",
                            flush=True,
                        )

            # Build AnnotatedDocument for LangExtract JSONL + HTML persistence inside organized paper folder
            _, item_doc = extract_grounded_breakouts(item, use_llm_extract=False)
            with spinner(f"Organizing generated artifacts in '{paper_paths['paper_dir']}'", enabled=show_ui):
                saved_artifacts = write_organized_paper_artifacts(
                    output_root=output_root,
                    paper=item,
                    extractions=item["extractions"],
                    annotated_doc=item_doc,
                    code_bundle=code_bundle,
                    evolution_report=evo_report,
                    dynarot_decision=dynarot_decision.to_dict(),
                    verification_report=verification_report,
                )

            print("\n  [ORGANIZED OUTPUT ARTIFACTS]", flush=True)
            print(f"    Paper Folder       : {saved_artifacts['paper_dir']}", flush=True)
            print(f"    Python Module      : {saved_artifacts['python_module']}", flush=True)
            print(f"    Evaluator Script   : {saved_artifacts['evaluator_script']}", flush=True)
            print(f"    LangExtract HTML   : {saved_artifacts['langextract_html']}", flush=True)
            print(f"    LangExtract JSONL  : {saved_artifacts['langextract_jsonl']}", flush=True)
            if saved_artifacts.get("evolved_champion"):
                print(f"    Evolved Champion   : {saved_artifacts['evolved_champion']}", flush=True)
                print(f"    Evolution Report   : {saved_artifacts['evolution_report']}", flush=True)
            if saved_artifacts.get("verification_report"):
                print(f"    Verification Report: {saved_artifacts['verification_report']}", flush=True)
            print(f"    Workspace Index    : {saved_artifacts['index_md']}", flush=True)
            print(f"    Workspace Manifest : {saved_artifacts['manifest_json']}", flush=True)

            papers_bar.update(1)
    finally:
        papers_bar.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
