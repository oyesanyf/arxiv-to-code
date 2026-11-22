# ArXiv-to-Code Master Documentation Index & Sitemap

Welcome to the **ArXiv-to-Code** (`arxiv-to-code`) technical documentation index. Every module, CLI flag, MCP tool, mathematical formulation, and evolutionary search algorithm is documented and cross-referenced below for fast indexing by search engines, RAG pipelines, and AI coding assistants.

---

## 1. Documentation Sitemap

| Document | Path | Primary Topics Covered |
|:---|:---|:---|
| **Root Overview & Quickstart** | [`README.md`](../README.md) | System architecture, Mermaid pipeline graphs, quickstart commands, mathematical scoring formulas, and IDE setup. |
| **CLI Reference Manual** | [`docs/CLI_REFERENCE.md`](CLI_REFERENCE.md) | Exhaustive reference for all CLI flags (`--search`, `--arxiv-id`, `--extract-doc`, `--langextract-llm`, `--dynarot`, `--install-models`, `--evolves`, `--evolve-mode`, `--evolve-count`, `--verify-llm`, `--langgraph`, `--mcp`, `--test`). |
| **Universal MCP Server Guide** | [`docs/MCP_SERVER_GUIDE.md`](MCP_SERVER_GUIDE.md) | Full JSON-RPC 2.0 specification for all 9 registered MCP tools and copy-paste configurations for Antigravity, Cursor, VS Code, Windsurf, Claude Desktop, and Zed. |
| **ModelFusion DynaRot Routing & Provisioning** | [`docs/DYNAROT_MODEL_ROUTING.md`](DYNAROT_MODEL_ROUTING.md) | Dynamic Available Memory Law (Anti-OOM), 4-objective Pareto front & geometric knee-point selection, UCB1 bandit exploration, and evolutionary compute budget splitting. |
| **ShinkaEvolve + OpenEvolve Architecture** | [`docs/SHINKA_OPENEVOLVE_ARCHITECTURE.md`](SHINKA_OPENEVOLVE_ARCHITECTURE.md) | Shared `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` convention, deterministic scoring harness, Sequential Two-Stage Pipeline, and Parallel Cross-Pollinating Ensemble. |

---

## 2. Source Code Package Index (`src/arxiv_to_code/`)

| Module File | Core Classes & Functions | Responsibility |
|:---|:---|:---|
| [`src/arxiv_to_code/__init__.py`](../src/arxiv_to_code/__init__.py) | Public API exports (`__version__ = "1.0.0"`) | Top-level package interface. |
| [`src/arxiv_to_code/models.py`](../src/arxiv_to_code/models.py) | `SystemHardwareResources`, `DynaRotRoutingDecision`, `CandidateScoreMetrics`, `EvolutionCandidate` | Strictly typed dataclasses for hardware telemetry, routing plans, candidate fitness scores, and evolved programs. |
| [`src/arxiv_to_code/langextract_engine.py`](../src/arxiv_to_code/langextract_engine.py) | `search_breakout_papers`, `fetch_paper_by_id`, `extract_grounded_breakouts`, `compute_article_rank`, `search_and_rank_articles`, `generate_interactive_html`, `extract_text_from_document_path`, `audit_defense_coverage_multi_target` | arXiv search, strict novelty filtering (`CODEBASE_IMPLEMENTED_PAPERS`), 3-class Google LangExtract character-interval grounding, 0–100 multi-factor ranking, and HTML visualization. |
| [`src/arxiv_to_code/dynarot.py`](../src/arxiv_to_code/dynarot.py) | `DynaRotModelRouter`, `detect_live_hardware`, `size_models_by_available_memory`, `size_glm_by_hardware`, `calculate_provision_plan`, `compute_pareto_front`, `select_knee_point`, `probe_ollama_models`, `sanitize_model_tag` | ModelFusion DynaRot + QuorumRAG AutoFusion hardware sizing by live available RAM/VRAM/disk, Pareto knee-point routing, UCB1 exploration, and pre-pull downsizing. |
| [`src/arxiv_to_code/ollama_provisioner.py`](../src/arxiv_to_code/ollama_provisioner.py) | `detect_ollama_executable`, `ensure_ollama_installed`, `ensure_ollama_daemon`, `select_dynamic_provision_targets`, `pull_ollama_model_streaming`, `dynamically_provision_ollama_and_models`, `ensure_models_for_pipeline` | Zero-touch-PATH Ollama auto-installer, detached `ollama serve` launcher, dynamic resource-aware panel target selection (`glm4:9b` + `deepseek-r1:7b`), and streaming `/api/pull` with live `tqdm` byte progress bars. |
| [`src/arxiv_to_code/python_code_generator.py`](../src/arxiv_to_code/python_code_generator.py) | `synthesize_production_python_code`, `synthesize_deterministic_scoring_script`, `derive_multi_token_indicators` | Synthesizes standalone, zero-mock, self-tested Python modules with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers, meet-semilattice delegation scopes, HMAC-SHA256 seals, and conformal submartingale kernels. |
| [`src/arxiv_to_code/evolve_engine.py`](../src/arxiv_to_code/evolve_engine.py) | `ShinkaEvolveEngine`, `ShinkaUCBRouter`, `ShinkaMetaScratchpad`, `ShinkaNoveltyArchive`, `OpenEvolveEngine`, `MapElitesGrid`, `evaluate_python_candidate`, `run_combined_shinka_openevolve` | Combined evolutionary engine pairing ShinkaEvolve (structural discovery) and OpenEvolve (multi-island MAP-Elites behavioral fine-tuning) across `--evolve-count` rounds in `sequential`, `parallel`, and `hybrid` modes. |
| [`src/arxiv_to_code/dream_rsi_engine.py`](../src/arxiv_to_code/dream_rsi_engine.py) | `TraceEnv`, `ReplayWorldPool`, `DreamPolicyOptimizer`, `detect_dream_rsi_backend`, `compute_student_t_ci95`, `run_dream_rsi_empirical_benchmark`, `run_dream_rsi_rl_optimization` | Dream-RSI Reinforcement Learning engine (`arXiv:2609.14858`): online `DiscoveryTree` rollouts, evolving zero-cost Replay Worlds (`trace_pool.jsonl`), offline dreaming meta-policy optimization, optional local Ollama `--rl-llm` rewriting, and 5-baseline 95% Student-$t$ CI empirical superiority proof. |
| [`src/arxiv_to_code/llm_verifier.py`](../src/arxiv_to_code/llm_verifier.py) | `CodeVerificationReport`, `run_deterministic_code_verification`, `parse_llm_verification_json`, `verify_generated_python_code` | Fail-closed Layer-1 Deterministic AST/Security/Unit-Test Gate paired with Layer-2 Open-Weight LLM-as-a-Judge (Ollama via DynaRot/QuorumRAG) auditing Paper Fidelity, Correctness, and Security. |
| [`src/arxiv_to_code/panel_verifier.py`](../src/arxiv_to_code/panel_verifier.py) | `run_model_panel_verification`, `assemble_verification_panel`, `derive_fusion_model_count`, `calculate_panel_batch_size`, `assess_glm_for_panel`, `JudgeAnalysis` | ModelFusion Panel-of-Models verification engine (resource-gated GLM + Qwen + DeepSeek + Gemma in VRAM-safe batches + minority-preserving Judge Analysis). |
| [`src/arxiv_to_code/workspace_layout.py`](../src/arxiv_to_code/workspace_layout.py) | `prepare_paper_workspace`, `write_organized_paper_artifacts` | Organizes default `generated/` output into per-paper folders (`src/`, `eval/`, `evolved/`, `verification_report.json`, LangExtract HTML/JSONL) plus root `INDEX.md` and `MANIFEST.json`. |
| [`src/arxiv_to_code/progress.py`](../src/arxiv_to_code/progress.py) | `ConsoleSpinner`, `spinner`, `create_progress_bar` | Animated console spinners and `tqdm` percentage progress bars (`0% -> 100%`). |
| [`src/arxiv_to_code/swarm.py`](../src/arxiv_to_code/swarm.py) | `run_langgraph_research_swarm`, `derive_arm_epoch_key`, `sign_and_verify_edge_transition` | 4-Arm Agentic StateGraph Swarm (`arm_1_arxiv_scout` $\to$ `arm_2_langextract_grounder` $\to$ `arm_3_python_code_synthesizer` $\to$ `arm_4_shinka_openevolve_optimizer`) with constant-time HMAC-SHA256 edge verification. |
| [`src/arxiv_to_code/mcp_server.py`](../src/arxiv_to_code/mcp_server.py) | `arxiv_to_code_status`, `dynarot_hardware_route_and_install`, `search_and_rank_arxiv_papers`, `extract_document_langextract`, `generate_python_code_from_paper`, `evolve_python_code`, `audit_defense_coverage`, `run_langgraph_arxiv_to_code_swarm`, `verify_generated_python_code` | Universal stdio JSON-RPC 2.0 MCP server exposing 9 tools. |
| [`src/arxiv_to_code/cli.py`](../src/arxiv_to_code/cli.py) | `build_arg_parser`, `normalize_cli_argv`, `run_self_test`, `main` | Unified all-in-one CLI dispatcher invoked by [`arixv_to_code.py`](../arixv_to_code.py) and 9-step automated end-to-end self-test suite. |
| [`tests/test_all_cli_flags.py`](../tests/test_all_cli_flags.py) | `test_flag_01_help_and_version` ... `test_flag_12_verify_llm_and_aliases` | Exhaustive verification suite testing every CLI flag, alias, and flag combination on `arixv_to_code.py`. |
| [`tests/test_llm_code_verifier.py`](../tests/test_llm_code_verifier.py) | `test_deterministic_verification_*`, `test_live_loopback_ollama_llm_verifier_*` | End-to-end verification tests for Deterministic AST Gate + real loopback HTTP Ollama LLM Judge + fail-closed override. |

---

## 3. Searchable Keyword & Concept Glossary

- **`3-Class Google LangExtract Grounding`**: Character-exact `CharInterval(start_pos, end_pos)` extraction across `breakout_vector`, `mathematical_formulation`, and `defense_countermeasure`.
- **`Asymmetric Meet-Semilattice (DelegationScope.meet)`**: Algebraic authority attenuation $S_1 \sqcap S_2 = (R_1 \cap R_2, A_1 \cap A_2, D_1 \cap D_2, X_1 \cup X_2)$ synthesized into generated Python modules.
- **`Dynamic Available Memory Law (DynaRot)`**: Invariant that local LLMs are sized strictly against runtime free RAM (`free_ram_gb`) and free GPU VRAM (`free_vram_mb`), never total physical RAM.
- **`EVOLVE-BLOCK Convention`**: Source code markers (`# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`) shared by both `ShinkaEvolve` and `OpenEvolve` to delimit mutable algorithmic kernels inside standalone Python files.
- **`MAP-Elites Behavioral Grid (OpenEvolve)`**: 2D quality-diversity grid indexing evolved programs across Execution Speed ($\mu\text{s}$) and Peak Memory Footprint ($\text{KiB}$ via `tracemalloc`).
- **`Meta-Scratchpad & Novelty Rejection (ShinkaEvolve)`**: Evolutionary memory of stepping-stone breakthroughs paired with canonical AST SHA-256 fingerprinting (`ast.dump`) to reject duplicate mutations before evaluation.
