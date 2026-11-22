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

class _LazyMCPServerRegistry:
    """
    Zero-overhead tool decorator registry that avoids importing heavy pydantic/anyio/uvicorn
    dependencies during CLI or pytest startup, and binds to the real MCPServer only when
    `run(transport="stdio")` is invoked.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: Dict[str, Callable[..., Any]] = {}

    def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[fn.__name__] = fn
            return fn

        return decorator

    def run(self, transport: str = "stdio") -> None:
        real_server = None
        try:
            from mcp.server.mcpserver import MCPServer as _RealMCPServer  # type: ignore

            real_server = _RealMCPServer(self.name)
        except ImportError:
            try:
                from mcp.server.fastmcp import FastMCP as _RealFastMCP  # type: ignore

                real_server = _RealFastMCP(self.name)
            except ImportError:
                real_server = None

        if real_server is not None:
            for fn in self.tools.values():
                real_server.tool()(fn)
            real_server.run(transport=transport)
        else:
            sys.stderr.write(f"[{self.name}] Minimal stdio MCP fallback loop started ({transport}).\n")
            sys.stderr.flush()


_MCPServerClass = _LazyMCPServerRegistry

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
from .llm_verifier import verify_generated_python_code as _verify_generated_python_code_impl
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
    model_family: str = "auto",
    auto_install_models: bool = False,
    override_free_ram_gb: Optional[float] = None,
    override_free_vram_mb: Optional[float] = None,
    override_free_disk_gb: Optional[float] = None,
) -> str:
    """
    Executes ModelFusion DynaRot + QuorumRAG AutoFusion hardware-aware routing strictly under
    the Dynamic Available Memory Law (free RAM, free VRAM, and free disk headroom — never total RAM),
    supports GLM (`glm-4.7-flash`, `glm4:9b`), Qwen 2.5, and Gemma families, splits compute budget
    between ShinkaEvolve and OpenEvolve, and optionally pulls missing Ollama models.
    """
    router = DynaRotModelRouter()
    if auto_install_models:
        res = router.provision_models_for_resources(
            override_free_ram_gb=override_free_ram_gb,
            override_free_vram_mb=override_free_vram_mb,
            override_free_disk_gb=override_free_disk_gb,
            model_family=model_family,
            auto_pull=True,
        )
        return json.dumps(res, indent=2)

    decision = router.route(
        prompt=prompt,
        override_free_ram_gb=override_free_ram_gb,
        override_free_vram_mb=override_free_vram_mb,
        override_free_disk_gb=override_free_disk_gb,
        model_family=model_family,
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
    use_google_langextract_llm: bool = False,
    langextract_model_id: str = "",
) -> str:
    """
    Extracts 3-class source-grounded spans (`breakout_vector`, `mathematical_formulation`,
    `defense_countermeasure`) with exact character intervals (`CharInterval(start_pos, end_pos)`)
    from an arXiv paper ID or local document path (.pdf, .txt, .md, .tex) using Google LangExtract
    (`langextract.extract()` when `use_google_langextract_llm=True` + `langextract.visualize()`),
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

        extractions, annotated_doc = extract_grounded_breakouts(
            paper,
            use_llm_extract=bool(use_google_langextract_llm),
            llm_model_id=langextract_model_id.strip() or None,
        )
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
    verify_llm: bool = False,
    verify_model: str = "",
) -> str:
    """
    Fetch an arXiv paper by ID (e.g. '2608.15888' or '2603.02277'), run Google LangExtract
    3-class character-interval source grounding, synthesize complete production Python code
    with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers and unit tests, optionally
    evolve it with ShinkaEvolve + OpenEvolve, and optionally verify it with an LLM judge.
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

        verification_report = None
        if verify_llm:
            target_code = (
                evolution_report["champion_candidate"]["source_code"]
                if evolution_report is not None
                else code_bundle["production_python_code"]
            )
            verification_report = _verify_generated_python_code_impl(
                source_code=target_code,
                paper=paper,
                extractions=extractions,
                code_stage="evolved_champion" if evolution_report is not None else "synthesized",
                model_id=verify_model.strip() or None,
                dynarot_decision=dynarot_decision,
                enable_llm=True,
            ).to_dict()

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
                "verification_report": verification_report,
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
    evolve_count: int = 1,
    elites_dir: str = "",
) -> str:
    """
    Runs `evolve_count` cycles of the combined ShinkaEvolve + OpenEvolve evolutionary search pipeline
    on a standalone Python file (or synthesizes one from `arxiv_id` first if `python_source_code` is omitted).
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
            evolve_count=max(1, int(evolve_count)),
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


# =============================================================================
# Tool 9: verify_generated_python_code
# =============================================================================

@server.tool()
async def verify_generated_python_code(
    arxiv_id: str = "2603.02277",
    python_source_code: str = "",
    verify_model: str = "",
    enable_llm: bool = True,
) -> str:
    """
    Verifies generated or evolved Python code against the arXiv paper's grounded spans,
    mathematical formulations, and security invariants using a fail-closed Deterministic AST +
    Runtime Test Gate combined with an Open-Weight LLM-as-a-Judge (local Ollama via ModelFusion DynaRot).
    """
    try:
        clean_id = normalize_arxiv_id(arxiv_id) or "2603.02277"
        paper = fetch_paper_by_id(clean_id)
        if not paper:
            paper = {
                "arxiv_id": clean_id,
                "title": f"Custom Verification Target ({clean_id})",
                "summary": "",
            }
        extractions, _ = extract_grounded_breakouts(paper)
        code_stage = "custom"
        if not python_source_code.strip():
            bundle = synthesize_production_python_code(paper, extractions)
            python_source_code = bundle["production_python_code"]
            code_stage = "synthesized"

        report = _verify_generated_python_code_impl(
            source_code=python_source_code,
            paper=paper,
            extractions=extractions,
            code_stage=code_stage,
            model_id=verify_model.strip() or None,
            enable_llm=bool(enable_llm),
        )
        return json.dumps(report.to_dict(), indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc), "arxiv_id": arxiv_id}, indent=2)


def build_mcp_arg_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="arxiv-to-code-mcp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "ArXiv-to-Code Universal MCP Server (stdio JSON-RPC 2.0)\n"
            "Exposes 9 specialized tools for arXiv search, 3-class Google LangExtract character-span\n"
            "grounding, Paper-to-Python code synthesis, ModelFusion DynaRot hardware-aware routing,\n"
            "combined ShinkaEvolve + OpenEvolve evolutionary search, and LLM-based code verification."
        ),
        epilog=(
            "Registered MCP Tools (9 Tools):\n"
            "  1. arxiv_to_code_status              : Health, versions, active DynaRot tier & ranking weights\n"
            "  2. dynarot_hardware_route_and_install: Live free RAM/VRAM probe, Pareto knee-point & ollama pull\n"
            "  3. search_and_rank_arxiv_papers      : Search arXiv, 3-class LangExtract grounding & Python codegen\n"
            "  4. extract_document_langextract      : Extract CharInterval spans from arXiv ID or local PDF/text + HTML\n"
            "  5. generate_python_code_from_paper   : Synthesize EVOLVE-BLOCK Python module from arXiv ID (+ evolve)\n"
            "  6. evolve_python_code                : Run ShinkaEvolve + OpenEvolve (sequential | parallel | hybrid)\n"
            "  7. audit_defense_coverage            : Audit paper vectors across Python & multi-ring boundaries\n"
            "  8. run_langgraph_arxiv_to_code_swarm : Execute 4-Arm HMAC-SHA256 signed StateGraph Research Swarm\n"
            "  9. verify_generated_python_code      : Verify generated Python code via Deterministic Gate + LLM Judge\n"
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print JSON catalog of all 9 registered MCP tools and exit",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="arxiv-to-code-mcp 1.0.0",
        help="Show MCP server version and exit",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_mcp_arg_parser()
    args = parser.parse_args(argv)
    if args.list_tools:
        tools_catalog = [
            {"name": "arxiv_to_code_status", "description": arxiv_to_code_status.__doc__.strip()},
            {"name": "dynarot_hardware_route_and_install", "description": dynarot_hardware_route_and_install.__doc__.strip()},
            {"name": "search_and_rank_arxiv_papers", "description": search_and_rank_arxiv_papers.__doc__.strip()},
            {"name": "extract_document_langextract", "description": extract_document_langextract.__doc__.strip()},
            {"name": "generate_python_code_from_paper", "description": generate_python_code_from_paper.__doc__.strip()},
            {"name": "evolve_python_code", "description": evolve_python_code.__doc__.strip()},
            {"name": "audit_defense_coverage", "description": audit_defense_coverage.__doc__.strip()},
            {"name": "run_langgraph_arxiv_to_code_swarm", "description": run_langgraph_arxiv_to_code_swarm.__doc__.strip()},
            {"name": "verify_generated_python_code", "description": verify_generated_python_code.__doc__.strip()},
        ]
        print(json.dumps({"server": "arxiv-to-code", "total_tools": 9, "tools": tools_catalog}, indent=2))
        return

    sys.stderr.write(
        "Starting ArXiv-to-Code + Google LangExtract + ModelFusion DynaRot + ShinkaEvolve/OpenEvolve + LLM Verifier MCP Server on stdio (9 Tools Registered)...\n"
    )
    sys.stderr.flush()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
