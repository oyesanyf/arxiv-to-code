"""
ArXiv-to-Code (`arxiv_to_code`)
======================================================================================
Autonomous arXiv Research Discovery, Google LangExtract 3-Class Source Grounding,
Paper-to-Python Code Synthesis, ModelFusion DynaRot Hardware-Aware Model Routing,
and Combined ShinkaEvolve + OpenEvolve Evolutionary Optimization (CLI & MCP Server).
"""

__version__ = "1.0.0"

from .dynarot import (
    DynaRotModelRouter,
    detect_live_hardware,
    probe_ollama_models,
    sanitize_model_tag,
    size_models_by_available_memory,
)
from .evolve_engine import (
    EVOLVE_END_MARKER,
    EVOLVE_START_MARKER,
    MapElitesGrid,
    OpenEvolveEngine,
    ShinkaEvolveEngine,
    evaluate_python_candidate,
    evaluate_python_candidate_file,
    extract_evolve_block,
    replace_evolve_block,
    run_combined_shinka_openevolve,
)
from .langextract_engine import (
    BREAKOUT_SIGNATURES,
    CODEBASE_IMPLEMENTED_PAPERS,
    DEFENSE_COUNTERMEASURE_SIGNATURES,
    MATH_FORMULATION_SIGNATURES,
    audit_defense_coverage_multi_target,
    compute_article_rank,
    extract_grounded_breakouts,
    extract_text_from_document_path,
    fetch_paper_by_id,
    generate_interactive_html,
    normalize_arxiv_id,
    search_and_rank_articles,
    search_breakout_papers,
)
from .models import (
    CandidateScoreMetrics,
    DynaRotRoutingDecision,
    EvolutionCandidate,
    SystemHardwareResources,
)
from .python_code_generator import (
    synthesize_deterministic_scoring_script,
    synthesize_production_code,
    synthesize_production_python_code,
)
from .swarm import run_langgraph_research_swarm

__all__ = [
    "__version__",
    "BREAKOUT_SIGNATURES",
    "CODEBASE_IMPLEMENTED_PAPERS",
    "CandidateScoreMetrics",
    "DEFENSE_COUNTERMEASURE_SIGNATURES",
    "DynaRotModelRouter",
    "DynaRotRoutingDecision",
    "EVOLVE_END_MARKER",
    "EVOLVE_START_MARKER",
    "EvolutionCandidate",
    "MATH_FORMULATION_SIGNATURES",
    "MapElitesGrid",
    "OpenEvolveEngine",
    "ShinkaEvolveEngine",
    "SystemHardwareResources",
    "audit_defense_coverage_multi_target",
    "compute_article_rank",
    "detect_live_hardware",
    "evaluate_python_candidate",
    "evaluate_python_candidate_file",
    "extract_evolve_block",
    "extract_grounded_breakouts",
    "extract_text_from_document_path",
    "fetch_paper_by_id",
    "generate_interactive_html",
    "normalize_arxiv_id",
    "probe_ollama_models",
    "replace_evolve_block",
    "run_combined_shinka_openevolve",
    "run_langgraph_research_swarm",
    "sanitize_model_tag",
    "search_and_rank_articles",
    "search_breakout_papers",
    "size_models_by_available_memory",
    "synthesize_deterministic_scoring_script",
    "synthesize_production_code",
    "synthesize_production_python_code",
]
