#!/usr/bin/env python3
"""
arxiv_to_code/mcp_server.py
======================================================================================
ArXiv-to-Code Universal MCP Server (stdio JSON-RPC 2.0):
- arXiv Research Discovery & Multi-Factor 0-100 Article Ranking
- Google LangExtract 3-Class Character-Span Source Grounding & Interactive HTML Reports
- Paper-to-Python Production Code Synthesis (with EVOLVE-BLOCK markers)
- ModelFusion DynaRot Dynamic Hardware Sizing & Resource-Based Model Provisioning
- Combined ShinkaEvolve + OpenEvolve Evolutionary Optimization (Sequential & Parallel)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure all logging goes strictly to stderr so MCP stdio JSON-RPC on stdout is never polluted
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)
for _logger_name in ("arxiv", "httpx", "httpcore", "urllib3", "langextract", "mcp"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

try:
    from mcp.server.mcpserver import MCPServer as _MCPServerClass  # type: ignore
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as _MCPServerClass  # type: ignore
    except ImportError:

        class _MCPServerClass:  # type: ignore[no-redef]
            def __init__(self, name: str) -> None:
                self.name = name
                self.tools: Dict[str, Callable[..., Any]] = {}

            def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                    self.tools[fn.__name__] = fn
                    return fn

                return decorator

            def run(self, transport: str = "stdio") -> None:
                sys.stderr.write(f"[{self.name}] Minimal stdio MCP fallback loop started ({transport}).\n")
                sys.stderr.flush()

from .dynarot import DynaRotModelRouter
from .evolve_engine import run_combined_shinka_openevolve
from .langextract_engine import (
    CODEBASE_IMPLEMENTED_PAPERS,
    arxiv,
    audit_defense_coverage_multi_target,
    compute_article_rank,
    extract_grounded_breakouts,
    extract_text_from_document_path,
    fetch_paper_by_id,
    generate_interactive_html,
    lx,
    normalize_arxiv_id,
    search_and_rank_articles,
)
from .python_code_generator import synthesize_production_python_code
from .swarm import run_langgraph_research_swarm

server = _MCPServerClass("arxiv-to-code")


# =============================================================================
# Tool 1: arxiv_to_code_status
# =============================================================================

@server.tool()
async def arxiv_to_code_status() -> str:
    """
    Check the health, runtime hardware sizing (ModelFusion DynaRot), Google LangExtract
    3-class grounding configuration, Paper-to-Python code generator, and combined
    ShinkaEvolve + OpenEvolve evolutionary search status.
    """
    router = DynaRotModelRouter()
    routing = router.route()
    return json.dumps(
        {
            "server": "arxiv-to-code",
            "status": "operational",
            "target_code_generation_language": "python",
            "evolve_pipeline": "ShinkaEvolve (UCB + Meta-Scratchpad + Novelty Rejection) + OpenEvolve (Multi-Island MAP-Elites)",
            "evolve_block_convention": "# EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END",
            "python_version": sys.version.split()[0],
            "langextract_version": getattr(lx, "__version__", "1.7.1"),
            "arxiv_version": getattr(arxiv, "__version__", "2.1.0"),
            "modelfusion_dynarot": routing.to_dict(),
            "ranking_factors": {
                "grounded_vector_density_max": 35.0,
                "threat_severity_tier_max": 30.0,
                "empirical_rigor_max": 20.0,
                "temporal_recency_max": 15.0,
                "novelty_delta_max": 15.0,
                "code_repo_priority_bonus": 20.0,
                "total_max_score": 100.0,
            },
            "grounding_classes": [
                "breakout_vector",
                "mathematical_formulation",
                "defense_countermeasure",
            ],
            "codebase_implemented_papers": CODEBASE_IMPLEMENTED_PAPERS,
        },
        indent=2,
    )


# =============================================================================
# Tool 2: dynarot_hardware_route_and_install
# =============================================================================

@server.tool()
async def dynarot_hardware_route_and_install(
    prompt: str = "Synthesize production Python code from arXiv paper and evolve with ShinkaEvolve + OpenEvolve",
    auto_install_models: bool = False,
    override_free_ram_gb: Optional[float] = None,
    override_free_vram_mb: Optional[float] = None,
) -> str:
    """
    Executes ModelFusion DynaRot hardware-aware routing strictly under the Dynamic Available
    Memory Law (free RAM & free VRAM, never total RAM), splits the compute budget between
    heavy reasoning models (for ShinkaEvolve) and fast lightweight models (for OpenEvolve),
    and optionally installs/pulls missing Ollama models based on runtime resources.
    """
    router = DynaRotModelRouter()
    if auto_install_models:
        res = router.provision_models_for_resources(
            override_free_ram_gb=override_free_ram_gb,
            override_free_vram_mb=override_free_vram_mb,
            auto_pull=True,
        )
        return json.dumps(res, indent=2)

    decision = router.route(
        prompt=prompt,
        override_free_ram_gb=override_free_ram_gb,
        override_free_vram_mb=override_free_vram_mb,
    )
    return json.dumps(decision.to_dict(), indent=2)


# =============================================================================
# Tool 3: search_and_rank_arxiv_papers
# =============================================================================

@server.tool()
async def search_and_rank_arxiv_papers(
    query: str = "LLM agent sandbox escape breakout",
    max_results: int = 5,
    min_score: float = 0.0,
    only_novel_enhancements: bool = True,
    evolve: bool = False,
    evolve_mode: str = "hybrid",
) -> str:
    """
    Search arXiv for research papers, extract 3-class source-grounded spans using Google
    LangExtract, rank papers from highest to lowest (0-100 score), synthesize runnable
    production Python code for each paper, and optionally evolve the code with ShinkaEvolve + OpenEvolve.
    """
    try:
        ranked, filtered_out = search_and_rank_articles(
            query=query,
            max_results=max(1, min(int(max_results), 25)),
            min_score=float(min_score),
            only_novel_enhancements=bool(only_novel_enhancements),
            return_filtered=True,
        )
        router = DynaRotModelRouter()
        dynarot_decision = router.route(prompt=query)

        lines = [
            "| Rank | arXiv ID | Score | Priority Tier | Title | Code Repo | Language |",
            "|:---:|:---:|:---:|:---:|:---|:---|:---:|",
        ]
        for item in ranked:
            py_bundle = synthesize_production_python_code(item, item.get("extractions", []))
            item["generated_python_code"] = py_bundle
            if evolve:
                item["evolution_report"] = run_combined_shinka_openevolve(
                    seed_python_code=py_bundle["production_python_code"],
                    arxiv_id=py_bundle["clean_arxiv_id"],
                    mode=evolve_mode,
                    generations=3,
                    num_islands=3,
                    dynarot_decision=dynarot_decision,
                )
            title_trunc = item["title"][:50] + "..." if len(item["title"]) > 50 else item["title"]
            repo_str = item.get("code_repo_url") or "N/A"
            lines.append(
                f"| #{item['rank_position']} | [{item['arxiv_id']}]({item['url']}) | {item['rank_score']}/100 | "
                f"`{item['priority_tier']}` | {title_trunc} | {repo_str} | `Python` |"
            )

        return json.dumps(
            {
                "query": query,
                "total_ranked": len(ranked),
                "only_novel_enhancements": only_novel_enhancements,
                "evolve_enabled": evolve,
                "dynarot_model_routing": dynarot_decision.to_dict(),
                "ranked_markdown_table": "\n".join(lines),
                "ranked_articles": ranked,
                "filtered_out_already_implemented": filtered_out,
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc), "query": query}, indent=2)


# =============================================================================
# Tool 4: extract_document_langextract
# =============================================================================

@server.tool()
async def extract_document_langextract(
    arxiv_id_or_path: str = "2603.02277",
    output_html_path: str = "",
) -> str:
    """
    Extracts 3-class source-grounded spans (`breakout_vector`, `mathematical_formulation`,
    `defense_countermeasure`) with exact character intervals (`CharInterval(start_pos, end_pos)`)
    from an arXiv paper ID or local document path (.pdf, .txt, .md, .tex) using Google LangExtract,
    and optionally generates an interactive HTML visualization report.
    """
    try:
        cand_path = Path(arxiv_id_or_path)
        if cand_path.exists() and cand_path.is_file():
            doc_text = extract_text_from_document_path(str(cand_path))
            paper = {
                "arxiv_id": cand_path.stem,
                "title": f"Local Document: {cand_path.name}",
                "published": "2026-01-01",
                "published_year": 2026,
                "url": f"file:///{cand_path.resolve().as_posix()}",
                "pdf_url": "",
                "authors": ["Local Document Author"],
                "summary": doc_text[:4000],
                "comments": "",
            }
        else:
            clean_id = normalize_arxiv_id(arxiv_id_or_path)
            fetched = fetch_paper_by_id(clean_id)
            if not fetched:
                return json.dumps({"error": f"Paper or file not found: {arxiv_id_or_path}"}, indent=2)
            paper = fetched

        extractions, annotated_doc = extract_grounded_breakouts(paper)
        rank_info = compute_article_rank(paper, extractions)

        saved_html = None
        if output_html_path.strip():
            saved_html = generate_interactive_html(annotated_doc, output_html_path)

        return json.dumps(
            {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "url": paper.get("url", ""),
                "rank_score": rank_info["total_score"],
                "priority_tier": rank_info["priority_tier"],
                "total_grounded_spans": len(extractions),
                "grounded_extractions": extractions,
                "interactive_html_report": saved_html,
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc), "arxiv_id_or_path": arxiv_id_or_path}, indent=2)


# =============================================================================
# Tool 5: generate_python_code_from_paper
# =============================================================================

@server.tool()
async def generate_python_code_from_paper(
    arxiv_id: str = "2608.15888",
    output_dir: str = "",
    evolve: bool = False,
    evolve_mode: str = "hybrid",
    generations: int = 4,
) -> str:
    """
    Fetch an arXiv paper by ID (e.g. '2608.15888' or '2603.02277'), run Google LangExtract
    3-class character-interval source grounding, synthesize complete production Python code
    with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers and unit tests, and optionally
    evolve it with ShinkaEvolve + OpenEvolve.
    """
    try:
        clean_id = normalize_arxiv_id(arxiv_id)
        paper = fetch_paper_by_id(clean_id)
        if not paper:
            return json.dumps({"error": f"Paper not found for arXiv ID: {clean_id}"}, indent=2)

        extractions, _ = extract_grounded_breakouts(paper)
        rank_info = compute_article_rank(paper, extractions)
        paper["code_repo_url"] = rank_info.get("code_repo_url", paper.get("code_repo_url", ""))

        code_bundle = synthesize_production_python_code(
            paper,
            extractions,
            output_dir=output_dir.strip() or None,
        )

        router = DynaRotModelRouter()
        dynarot_decision = router.route(prompt=f"Generate and evolve Python code for arXiv:{clean_id}")

        evolution_report = None
        if evolve:
            evolution_report = run_combined_shinka_openevolve(
                seed_python_code=code_bundle["production_python_code"],
                arxiv_id=clean_id,
                mode=evolve_mode,
                generations=generations,
                num_islands=3,
                elites_dir=output_dir.strip() or None,
                dynarot_decision=dynarot_decision,
            )

        return json.dumps(
            {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "rank_score": rank_info["total_score"],
                "priority_tier": rank_info["priority_tier"],
                "dynarot_model_routing": dynarot_decision.to_dict(),
                "grounded_extractions": extractions,
                "generated_python_code": code_bundle,
                "evolution_report": evolution_report,
                "printable_code_bundle_markdown": code_bundle["printable_code_bundle_markdown"],
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc), "arxiv_id": arxiv_id}, indent=2)


# =============================================================================
# Tool 6: evolve_python_code
# =============================================================================

@server.tool()
async def evolve_python_code(
    arxiv_id: str = "2603.02277",
    python_source_code: str = "",
    evolve_mode: str = "hybrid",
    generations: int = 4,
    num_islands: int = 3,
    elites_dir: str = "",
) -> str:
    """
    Runs the combined ShinkaEvolve + OpenEvolve evolutionary search pipeline on a standalone
    Python file (or synthesizes one from `arxiv_id` first if `python_source_code` is omitted).
    Supports `sequential` (Stage 1 ShinkaEvolve -> Stage 2 OpenEvolve MAP-Elites),
    `parallel` (cross-pollinating ensemble sharing `elites_dir`), and `hybrid` modes.
    """
    try:
        clean_id = normalize_arxiv_id(arxiv_id) or "2603.02277"
        if not python_source_code.strip():
            paper = fetch_paper_by_id(clean_id)
            if not paper:
                return json.dumps({"error": f"Paper not found for arXiv ID: {clean_id}"}, indent=2)
            extractions, _ = extract_grounded_breakouts(paper)
            bundle = synthesize_production_python_code(paper, extractions)
            python_source_code = bundle["production_python_code"]

        router = DynaRotModelRouter()
        dynarot_decision = router.route(
            prompt=f"Evolve Python code for arXiv:{clean_id} in {evolve_mode} mode"
        )
        report = run_combined_shinka_openevolve(
            seed_python_code=python_source_code,
            arxiv_id=clean_id,
            mode=evolve_mode,
            generations=max(1, int(generations)),
            num_islands=max(1, int(num_islands)),
            elites_dir=elites_dir.strip() or None,
            dynarot_decision=dynarot_decision,
        )
        return json.dumps(report, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc), "arxiv_id": arxiv_id}, indent=2)


# =============================================================================
# Tool 7: audit_defense_coverage
# =============================================================================

@server.tool()
async def audit_defense_coverage(
    arxiv_id: str = "2603.02277",
    targets: Optional[List[str]] = None,
) -> str:
    """
    Evaluates grounded extractions from an arXiv paper against active defense boundaries
    and returns a full coverage breakdown alongside the generated Python code bundle.
    """
    try:
        clean_id = normalize_arxiv_id(arxiv_id)
        audit_res = audit_defense_coverage_multi_target(clean_id, targets=targets)
        paper = fetch_paper_by_id(clean_id)
        if paper:
            exts, _ = extract_grounded_breakouts(paper)
            code_bundle = synthesize_production_python_code(paper, exts)
            audit_res["generated_python_code"] = code_bundle
            audit_res["printable_code_bundle_markdown"] = code_bundle["printable_code_bundle_markdown"]
        return json.dumps(audit_res, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc), "arxiv_id": arxiv_id}, indent=2)


# =============================================================================
# Tool 8: run_langgraph_arxiv_to_code_swarm
# =============================================================================

@server.tool()
async def run_langgraph_arxiv_to_code_swarm(
    query: str = "LLM agent sandbox escape breakout",
    arxiv_id: str = "2603.02277",
    max_results: int = 3,
    min_score: float = 0.0,
    only_novel_enhancements: bool = True,
    evolve_mode: str = "hybrid",
) -> str:
    """
    Executes the 4-Arm HMAC-SHA256 signed ArXiv-to-Code Swarm:
    1. `arm_1_arxiv_scout`
    2. `arm_2_langextract_grounder`
    3. `arm_3_python_code_synthesizer`
    4. `arm_4_shinka_openevolve_optimizer`
    """
    try:
        res = run_langgraph_research_swarm(
            query=query,
            arxiv_id=arxiv_id or None,
            max_results=max_results,
            min_score=min_score,
            only_novel_enhancements=only_novel_enhancements,
            evolve_mode=evolve_mode,
        )
        return json.dumps(res, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc), "arxiv_id": arxiv_id}, indent=2)


def main() -> None:
    sys.stderr.write(
        "Starting ArXiv-to-Code + Google LangExtract + ModelFusion DynaRot + ShinkaEvolve/OpenEvolve MCP Server on stdio (8 Tools Registered)...\n"
    )
    sys.stderr.flush()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
