# ArXiv-to-Code Master Documentation Index & Sitemap

Welcome to the **ArXiv-to-Code** (`arxiv-to-code`) technical documentation index. Every module, CLI flag, MCP tool, mathematical formulation, and evolutionary search algorithm is documented and cross-referenced below for fast indexing by search engines, RAG pipelines, and AI coding assistants.

---

## 1. Documentation Sitemap

| Document | Path | Primary Topics Covered |
|:---|:---|:---|
| **Root Overview & Quickstart** | [`README.md`](../README.md) | System architecture, Mermaid pipeline graphs, quickstart commands, mathematical scoring formulas, and IDE setup. |
| **CLI Reference Manual** | [`docs/CLI_REFERENCE.md`](CLI_REFERENCE.md) | Exhaustive reference for all CLI flags (`--query`, `--arxiv-id`, `--extract-doc`, `--dynarot`, `--install-models`, `--evolve` / `--evolves`, `--evolve-mode`, `--langgraph`, `--mcp`, `--test`). |
| **Universal MCP Server Guide** | [`docs/MCP_SERVER_GUIDE.md`](MCP_SERVER_GUIDE.md) | Full JSON-RPC 2.0 specification for all 8 registered MCP tools and copy-paste configurations for Antigravity, Cursor, VS Code, Windsurf, Claude Desktop, and Zed. |
| **ModelFusion DynaRot Routing & Provisioning** | [`docs/DYNAROT_MODEL_ROUTING.md`](DYNAROT_MODEL_ROUTING.md) | Dynamic Available Memory Law (Anti-OOM), 4-objective Pareto front & geometric knee-point selection, UCB1 bandit exploration, and evolutionary compute budget splitting. |
| **ShinkaEvolve + OpenEvolve Architecture** | [`docs/SHINKA_OPENEVOLVE_ARCHITECTURE.md`](SHINKA_OPENEVOLVE_ARCHITECTURE.md) | Shared `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` convention, deterministic scoring harness, Sequential Two-Stage Pipeline, and Parallel Cross-Pollinating Ensemble. |

---

## 2. Source Code Package Index (`src/arxiv_to_code/`)

| Module File | Core Classes & Functions | Responsibility |
|:---|:---|:---|
| [`src/arxiv_to_code/__init__.py`](../src/arxiv_to_code/__init__.py) | Public API exports (`__version__ = "1.0.0"`) | Top-level package interface. |
| [`src/arxiv_to_code/models.py`](../src/arxiv_to_code/models.py) | `SystemHardwareResources`, `DynaRotRoutingDecision`, `CandidateScoreMetrics`, `EvolutionCandidate` | Strictly typed dataclasses for hardware telemetry, routing plans, candidate fitness scores, and evolved programs. |
| [`src/arxiv_to_code/langextract_engine.py`](../src/arxiv_to_code/langextract_engine.py) | `search_breakout_papers`, `fetch_paper_by_id`, `extract_grounded_breakouts`, `compute_article_rank`, `search_and_rank_articles`, `generate_interactive_html`, `extract_text_from_document_path`, `audit_defense_coverage_multi_target` | arXiv search, strict novelty filtering (`CODEBASE_IMPLEMENTED_PAPERS`), 3-class Google LangExtract character-interval grounding, 0–100 multi-factor ranking, and HTML visualization. |
| [`src/arxiv_to_code/dynarot.py`](../src/arxiv_to_code/dynarot.py) | `DynaRotModelRouter`, `detect_live_hardware`, `size_models_by_available_memory`, `compute_pareto_front`, `select_knee_point`, `calculate_ucb1_exploration_bonus`, `probe_ollama_models`, `sanitize_model_tag` | ModelFusion DynaRot hardware sizing by live available RAM/VRAM, Pareto knee-point routing, UCB1 exploration, and `ollama pull` auto-provisioning. |
| [`src/arxiv_to_code/python_code_generator.py`](../src/arxiv_to_code/python_code_generator.py) | `synthesize_production_python_code`, `synthesize_deterministic_scoring_script`, `derive_multi_token_indicators` | Synthesizes standalone, zero-mock, self-tested Python modules with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers, meet-semilattice delegation scopes, HMAC-SHA256 seals, and conformal submartingale kernels. |
| [`src/arxiv_to_code/evolve_engine.py`](../src/arxiv_to_code/evolve_engine.py) | `ShinkaEvolveEngine`, `ShinkaUCBRouter`, `ShinkaMetaScratchpad`, `ShinkaNoveltyArchive`, `OpenEvolveEngine`, `MapElitesGrid`, `evaluate_python_candidate`, `run_combined_shinka_openevolve` | Combined evolutionary engine pairing ShinkaEvolve (structural discovery) and OpenEvolve (multi-island MAP-Elites behavioral fine-tuning) in `sequential`, `parallel`, and `hybrid` modes. |
| [`src/arxiv_to_code/swarm.py`](../src/arxiv_to_code/swarm.py) | `run_langgraph_research_swarm`, `derive_arm_epoch_key`, `sign_and_verify_edge_transition` | 4-Arm Agentic StateGraph Swarm (`arm_1_arxiv_scout` $\to$ `arm_2_langextract_grounder` $\to$ `arm_3_python_code_synthesizer` $\to$ `arm_4_shinka_openevolve_optimizer`) with constant-time HMAC-SHA256 edge verification. |
| [`src/arxiv_to_code/mcp_server.py`](../src/arxiv_to_code/mcp_server.py) | `arxiv_to_code_status`, `dynarot_hardware_route_and_install`, `search_and_rank_arxiv_papers`, `extract_document_langextract`, `generate_python_code_from_paper`, `evolve_python_code`, `audit_defense_coverage`, `run_langgraph_arxiv_to_code_swarm` | Universal stdio JSON-RPC 2.0 MCP server exposing 8 tools. |
| [`src/arxiv_to_code/cli.py`](../src/arxiv_to_code/cli.py) | `build_arg_parser`, `run_self_test`, `main` | Unified CLI dispatcher and 8-step automated end-to-end self-test suite. |

---

## 3. Searchable Keyword & Concept Glossary

- **`3-Class Google LangExtract Grounding`**: Character-exact `CharInterval(start_pos, end_pos)` extraction across `breakout_vector`, `mathematical_formulation`, and `defense_countermeasure`.
- **`Asymmetric Meet-Semilattice (DelegationScope.meet)`**: Algebraic authority attenuation $S_1 \sqcap S_2 = (R_1 \cap R_2, A_1 \cap A_2, D_1 \cap D_2, X_1 \cup X_2)$ synthesized into generated Python modules.
- **`Dynamic Available Memory Law (DynaRot)`**: Invariant that local LLMs are sized strictly against runtime free RAM (`free_ram_gb`) and free GPU VRAM (`free_vram_mb`), never total physical RAM.
- **`EVOLVE-BLOCK Convention`**: Source code markers (`# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`) shared by both `ShinkaEvolve` and `OpenEvolve` to delimit mutable algorithmic kernels inside standalone Python files.
- **`MAP-Elites Behavioral Grid (OpenEvolve)`**: 2D quality-diversity grid indexing evolved programs across Execution Speed ($\mu\text{s}$) and Peak Memory Footprint ($\text{KiB}$ via `tracemalloc`).
- **`Meta-Scratchpad & Novelty Rejection (ShinkaEvolve)`**: Evolutionary memory of stepping-stone breakthroughs paired with canonical AST SHA-256 fingerprinting (`ast.dump`) to reject duplicate mutations before evaluation.
