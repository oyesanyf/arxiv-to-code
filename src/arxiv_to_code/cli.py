#!/usr/bin/env python3
"""
arxiv_to_code/cli.py
======================================================================================
Unified Command-Line Interface for ArXiv-to-Code:
- Search arXiv & rank papers (0-100 multi-factor score with strict novelty gate)
- Extract documents & 3-class character-span grounding with Google LangExtract
- Synthesize runnable, zero-mock production Python code with EVOLVE-BLOCK markers
- ModelFusion DynaRot hardware-aware model routing & resource-based model installation
- Combined `--evolve` / `--evolves` / `--evoves` flag pairing ShinkaEvolve + OpenEvolve
  (Sequential Two-Stage Pipeline & Parallel Cross-Pollinating Ensemble)
"""

from __future__ import annotations

import argparse
import json
import os
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
from .mcp_server import main as run_mcp_server
from .python_code_generator import synthesize_production_python_code
from .swarm import run_langgraph_research_swarm


def run_self_test() -> bool:
    """
    Comprehensive 8-step end-to-end functional self-test verifying:
    1. arXiv discovery & strict novelty filtering
    2. Google LangExtract 3-class character-interval source grounding
    3. Multi-factor 0-100 ranking & code repo bonus
    4. Interactive HTML report generation (`langextract.visualize`)
    5. ModelFusion DynaRot Dynamic Available Memory Law & Compute Budget Split
    6. Paper-to-Python production code generation & deterministic execution
    7. Combined ShinkaEvolve + OpenEvolve (`--evolve`) sequential & parallel cross-pollination
    8. 4-Arm HMAC-SHA256 signed LangGraph Research & Evolutionary Swarm
    """
    print("================================================================================")
    print("[ARXIV-TO-CODE] 8-STEP END-TO-END FUNCTIONAL SELF-TEST SUITE")
    print("================================================================================")

    # 1. Test arXiv discovery & novelty-filtered ranking
    print("[1/8] Testing arXiv discovery & strict novelty-filtered article ranking...")
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
    print("[2/8] Testing Google LangExtract 3-class character-span grounding on arXiv:2603.02277...")
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
    print("[3/8] Verifying strict novelty rejection of already-implemented paper (arXiv:2610.09167)...")
    impl_paper = fetch_paper_by_id("2610.09167")
    assert impl_paper is not None
    impl_exts, _ = extract_grounded_breakouts(impl_paper)
    impl_rank = compute_article_rank(impl_paper, impl_exts)
    assert impl_rank["is_novel_to_codebase"] is False
    assert impl_rank["novelty_delta_score"] == 0.0
    print("  [PASS] Correctly rejected already-implemented paper 2610.09167 (Novelty Delta: 0.0 pts)")

    # 4. Test Interactive HTML Visualization
    print("[4/8] Testing Google LangExtract interactive HTML report generation...")
    test_html = Path("test_arxiv_to_code_audit.html")
    generate_interactive_html(annotated_doc, str(test_html))
    assert test_html.exists() and test_html.stat().st_size > 250
    test_html.unlink()
    print("  [PASS] Generated and verified interactive LangExtract HTML visualization")

    # 5. Test ModelFusion DynaRot Dynamic Available Memory Law across all 5 tiers
    print("[5/8] Testing ModelFusion DynaRot hardware sizing & compute budget split...")
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
    print("[6/8] Testing Paper-to-Python production code generation & deterministic benchmark...")
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

    # 7. Test Combined ShinkaEvolve + OpenEvolve (--evolve) Pipeline
    print("[7/8] Testing Combined ShinkaEvolve + OpenEvolve evolutionary search (Sequential & Parallel)...")
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
    print("[8/8] Testing 4-Arm HMAC-SHA256 Signed ArXiv-to-Code Swarm...")
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

    print("================================================================================")
    print("[PASS] ALL 8 ARXIV-TO-CODE, LANGEXTRACT, DYNAROT & SHINKA+OPENEVOLVE TESTS PASSED!")
    print("================================================================================")
    return True


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
EXECUTABLE CLI EXAMPLES
================================================================================
  # 1. Run 8-step automated end-to-end self-test suite
  python cli.py --test

  # 2. Probe live RAM/VRAM, Pareto knee-point model, and Shinka/OpenEvolve compute split
  python cli.py --dynarot

  # 3. Auto-install/pull optimal Ollama models based on live available RAM & VRAM
  python cli.py --install-models

  # 4. Search arXiv, extract 3-class Google LangExtract spans, and synthesize Python code
  python cli.py --query "LLM container sandbox escape" --max-results 3 --output-dir ./generated

  # 5. Synthesize Python code for arXiv:2608.15888 and evolve with ShinkaEvolve + OpenEvolve
  python cli.py --arxiv-id 2608.15888 --evolves --evolve-mode hybrid --output-dir ./generated

  # 6. Extract a local PDF/LaTeX/Markdown document and write interactive LangExtract HTML
  python cli.py --extract-doc ./paper.pdf --html-report ./audit.html --evolves

  # 7. Run 4-Arm HMAC-SHA256 signed LangGraph Research & Evolutionary Swarm
  python cli.py --arxiv-id 2603.02277 --langgraph

  # 8. Launch Universal MCP Server over stdio JSON-RPC (8 tools)
  python cli.py --mcp
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv-to-code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "ArXiv-to-Code CLI & Universal MCP Server (v1.0.0)\n"
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
        version="arxiv-to-code 1.0.0",
        help="Show program's version number and exit",
    )

    g_search = parser.add_argument_group("1. arXiv Discovery & Multi-Factor Ranking (0-100 Score)")
    g_search.add_argument(
        "--query",
        "--search",
        dest="query",
        type=str,
        default="LLM agent sandbox escape breakout",
        help="Search query for discovering and ranking arXiv papers (default: 'LLM agent sandbox escape breakout')",
    )
    g_search.add_argument(
        "--arxiv-id",
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

    g_codegen = parser.add_argument_group("3. Paper-to-Python Production Code Synthesis")
    g_codegen.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write generated Python modules (arxiv_<id>_module.py) and deterministic scoring scripts",
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
        action="store_true",
        help="Automatically install/pull optimal local Ollama models (ollama pull <model>) based on live available RAM and VRAM",
    )

    g_evolve = parser.add_argument_group("5. Combined ShinkaEvolve + OpenEvolve Evolutionary Search")
    g_evolve.add_argument(
        "--evolve",
        "--evolves",
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
        "--generations",
        type=int,
        default=4,
        help="Number of evolutionary generations per stage (default: 4)",
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

    g_swarm = parser.add_argument_group("6. Agentic Swarm, Universal MCP Server & Self-Test Suite")
    g_swarm.add_argument(
        "--langgraph",
        action="store_true",
        help="Execute the 4-Arm HMAC-SHA256 signed ArXiv-to-Code StateGraph Research & Evolutionary Swarm",
    )
    g_swarm.add_argument(
        "--mcp",
        action="store_true",
        help="Start the ArXiv-to-Code Universal MCP Server over stdio JSON-RPC 2.0 (8 Tools)",
    )
    g_swarm.add_argument(
        "--test",
        action="store_true",
        help="Run the 8-step automated end-to-end functional self-test suite",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mcp:
        run_mcp_server()
        return 0

    if args.test:
        ok = run_self_test()
        return 0 if ok else 1

    router = DynaRotModelRouter()

    if args.install_models:
        print("================================================================================")
        print("[MODELFUSION DYNAROT] RESOURCE-BASED MODEL PROVISIONING & AUTO-INSTALL")
        print("================================================================================")
        prov = router.provision_models_for_resources(auto_pull=True)
        print(json.dumps(prov, indent=2))
        if not args.arxiv_id and not args.evolve and not args.dynarot and not args.extract_doc:
            return 0

    dynarot_decision = router.route(prompt=args.query)
    if args.dynarot:
        print("================================================================================")
        print("[MODELFUSION DYNAROT] HARDWARE SIZING, PARETO FRONT & COMPUTE SPLIT")
        print("================================================================================")
        print(json.dumps(dynarot_decision.to_dict(), indent=2))
        if not args.arxiv_id and not args.evolve and not args.extract_doc:
            return 0

    only_novel = not args.include_implemented

    if args.langgraph:
        print("================================================================================")
        print("[SWARM] EXECUTING 4-ARM ARXIV-TO-CODE STATEGRAPH SWARM")
        print("================================================================================")
        final_state = run_langgraph_research_swarm(
            query=args.query,
            arxiv_id=args.arxiv_id,
            max_results=args.max_results,
            min_score=args.min_score,
            only_novel_enhancements=only_novel,
            evolve_mode=args.evolve_mode,
        )
        print(json.dumps(final_state, indent=2))
        return 0

    # Gather target paper(s)
    if args.extract_doc:
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
        extractions, annotated_doc = extract_grounded_breakouts(paper)
        rank_info = compute_article_rank(paper, extractions)
        if args.html_report:
            saved = generate_interactive_html(annotated_doc, args.html_report)
            print(f"[+] Interactive LangExtract HTML visualization saved to: {saved}")
        ranked_articles = [
            {
                "rank_position": 1,
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "published": paper["published"],
                "url": paper["url"],
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
        paper = fetch_paper_by_id(args.arxiv_id)
        if not paper:
            print(f"[!] Paper not found for arXiv ID: {args.arxiv_id}", file=sys.stderr)
            return 1
        extractions, annotated_doc = extract_grounded_breakouts(paper)
        rank_info = compute_article_rank(paper, extractions)
        if args.html_report:
            saved = generate_interactive_html(annotated_doc, args.html_report)
            print(f"[+] Interactive LangExtract HTML visualization saved to: {saved}")
        ranked_articles = [
            {
                "rank_position": 1,
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "published": paper.get("published", ""),
                "url": paper.get("url", ""),
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
        print(f"[*] Searching arXiv articles for: '{args.query}' (Strict Novelty Gate: {only_novel})...")
        ranked_articles = search_and_rank_articles(
            query=args.query,
            max_results=args.max_results,
            min_score=args.min_score,
            only_novel_enhancements=only_novel,
        )

    print("\n================================================================================")
    print(
        f"[DYNAROT] Active Tier: {dynarot_decision.hardware_tier}\n"
        f"          Primary Model: {dynarot_decision.primary_model} | "
        f"ShinkaEvolve (Heavy): {dynarot_decision.shinka_heavy_reasoning_model} | "
        f"OpenEvolve (Fast): {dynarot_decision.openevolve_fast_mutation_model}"
    )
    print("================================================================================")

    for item in ranked_articles:
        print(
            f"\nRank #{item['rank_position']} | Score: {item['rank_score']}/100 [{item['priority_tier']}] | "
            f"arXiv:{item['arxiv_id']}"
        )
        print(f"  Title     : {item['title']}")
        for ext in item["extractions"]:
            print(
                f"  -> [{ext['extraction_class'].upper()}] {ext['attack_vector']} -> "
                f"'{ext['source_quote']}' [{ext['source_char_start']}:{ext['source_char_end']}]"
            )

        code_bundle = synthesize_production_python_code(
            item,
            item["extractions"],
            output_dir=args.output_dir,
        )
        print("\n" + "=" * 80)
        print(code_bundle["printable_code_bundle_markdown"])
        print("=" * 80)

        if args.evolve:
            print("\n================================================================================")
            print(
                f"[EVOLVE] Running Combined ShinkaEvolve + OpenEvolve ({args.evolve_mode.upper()} mode) "
                f"for arXiv:{code_bundle['clean_arxiv_id']}..."
            )
            print("================================================================================")
            evo_report = run_combined_shinka_openevolve(
                seed_python_code=code_bundle["production_python_code"],
                arxiv_id=code_bundle["clean_arxiv_id"],
                mode=args.evolve_mode,
                generations=args.generations,
                num_islands=args.islands,
                elites_dir=args.elites_dir or args.output_dir,
                dynarot_decision=dynarot_decision,
            )
            print(
                f"  Baseline Seed Fitness : {evo_report['baseline_seed_metrics']['total_fitness']}/100 "
                f"({evo_report['baseline_seed_metrics']['execution_time_us']} us, "
                f"{evo_report['baseline_seed_metrics']['peak_memory_kib']} KiB)"
            )
            print(
                f"  Champion Elite Fitness: {evo_report['champion_metrics']['total_fitness']}/100 "
                f"({evo_report['champion_metrics']['execution_time_us']} us, "
                f"{evo_report['champion_metrics']['peak_memory_kib']} KiB)"
            )
            print(f"  Speedup Ratio         : {evo_report['speedup_ratio']}x")
            print(f"  Champion Origin       : {evo_report['champion_candidate']['engine_origin']}")
            print(f"  Champion Operator     : {evo_report['champion_candidate']['mutation_operator']}")
            print(f"  Champion File Saved   : {evo_report['champion_file_path']}")
            print(
                f"  ShinkaEvolve UCB1     : {evo_report['shinkaevolve_summary']['ucb1_model_scores']}"
            )
            print(
                f"  MAP-Elites Niches     : {evo_report['openevolve_map_elites_summary']['occupied_niches_count']} "
                f"occupied niches ({list(evo_report['openevolve_map_elites_summary']['niches'].keys())})"
            )
            print("  Cross-Pollination Log :")
            for ev in evo_report["cross_pollination_events"]:
                print(f"    * {ev['direction']} -> {ev}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
