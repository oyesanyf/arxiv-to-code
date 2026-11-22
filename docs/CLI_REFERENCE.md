# ArXiv-to-Code CLI Reference Manual (`arixv_to_code.py`)

The **ArXiv-to-Code All-in-One CLI** (`python arixv_to_code.py`) provides single-script flag access to every capability in the repository: arXiv search (`--search` / `--searct` / `--seach` / `--query`), Google LangExtract 3-class character-span grounding (`--extract-doc`, `--html-report`), Paper-to-Python code synthesis (`--generate`, `--output-dir`), ModelFusion DynaRot hardware sizing & provisioning (`--dynarot` / `--sys-info`, `--install-models`), combined `ShinkaEvolve` + `OpenEvolve` optimization (`--evolves` / `---evolves` / `--evolve` / `--evoves`, `--evolve-mode`), multi-target defense auditing (`--audit`), 4-Arm HMAC-SHA256 signed LangGraph swarm (`--langgraph` / `--swarm`), status reporting (`--status`), Universal MCP Server (`--mcp`, `--list-tools`), and test suites (`--test`, `--pytest`).

---

## 1. Master Flag Reference Table

| Flag / Alias | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--search`, `--searct`, `--seach`, `--query` | `str` | `"LLM agent sandbox escape breakout"` | Search query for discovering and ranking arXiv papers. |
| `--arxiv-id`, `--id` | `str` | `None` | Specific arXiv paper ID to fetch, ground, and synthesize into Python (e.g., `2608.15888`, `2603.02277`, `2609.00005`). |
| `--max-results` | `int` | `3` | Maximum number of ranked novel arXiv papers to return. |
| `--min-score` | `float` | `0.0` | Minimum composite rank score (`0.0–100.0`) threshold. |
| `--include-implemented` | `flag` | `False` | Disable strict novelty filtering and include papers listed in `CODEBASE_IMPLEMENTED_PAPERS`. |
| `--extract-doc`, `--extract` | `str` | `None` | Path to a local document (`.pdf`, `.txt`, `.md`, `.tex`) to extract with Google LangExtract and synthesize into Python code. |
| `--html-report` | `str` | `None` | Output path for self-contained Google LangExtract interactive HTML visualization (`langextract.visualize()`). |
| `--langextract-llm` | `flag` | `False` | Invoke Google's real `langextract.extract()` LLM-backed 3-class grounding (via local Ollama or `GEMINI_API_KEY`) merged with deterministic character-interval grounding. |
| `--langextract-model` | `str` | `None` | Optional model ID for `langextract.extract()` (e.g., `qwen2.5:7b` for local Ollama or `gemini-2.5-flash` for Gemini). |
| `--evolve-llm` | `flag` | `False` | Enable live local Ollama LLM structural synthesis calls during `ShinkaEvolve` discovery. |
| `--no-progress` | `flag` | `False` | Disable `tqdm` percentage progress bars and animated console spinners for quiet scripted piping. |
| `--generate` | `flag` | `False` | Synthesize runnable production Python module with `# EVOLVE-BLOCK` markers (enabled automatically on `--search`/`--arxiv-id`/`--extract-doc`). |
| `--output-dir` | `str` | `"generated"` | Organized root output directory (`INDEX.md`, `MANIFEST.json`, and per-paper folders with `src/`, `eval/`, `evolved/`, `langextract_report.html`, `extraction_results.jsonl`, `metadata.json`, `README.md`). |
| `--audit`, `--audit-defense` | `flag` | `False` | Run multi-target defense & invariant coverage audit across Python runtime, OshoosiClaw, KillSwitch, and AgentOctopus. |
| `--verify-llm`, `--llm-verify`, `--verify`, `--verify-code` | `flag` | `False` | Verify generated (and evolved) Python code via Layer 1 Deterministic AST/Security/Unit-Test Gate + Layer 2 Local Open-Weight LLM-as-a-Judge (Ollama via DynaRot) and save `verification_report.json`. |
| `--verify-model` | `str` | `None` | Optional model tag for `--verify-llm` (e.g., `glm-4.7-flash`, `glm4:9b`, `qwen2.5:7b`, `deepseek-r1:7b`, or `gemini-2.5-flash`). |
| `--verify-panel`, `--panel`, `--fusion-verify` | `flag` | `False` | Run **ModelFusion Panel-of-Models** verification (GLM + Qwen + DeepSeek + Gemma in memory-safe VRAM batches) followed by minority-preserving Judge Analysis + Layer-1 deterministic gate. |
| `--panel-size` | `int` | `auto` | Number of panelists for `--verify-panel` (auto-derived via `derive_fusion_model_count()` from live free RAM/VRAM: `2–7`). |
| `--panel-models` | `str` | `None` | Explicit comma-separated Ollama panelist list (e.g., `glm-4.7-flash,qwen2.5:7b,deepseek-r1:7b`). |
| `--panel-mode` | `str` | `"multi-model"` | ModelFusion panel mode: `multi-model` (distinct families + persona variations) or `multi-sample` (diverse temperatures/personas of top model). |
| `--dynarot`, `--sys-info` | `flag` | `False` | Probe live free RAM, free GPU VRAM, and free disk headroom, compute 4-objective Pareto front & QuorumRAG-calibrated GLM/Qwen/Gemma provision plan. |
| `--model-family` | `str` | `"auto"` | Preferred open-weight Ollama model family calibrated to available RAM/VRAM/disk (`auto`, `glm` [`glm-4.7-flash`/`glm4:9b`], `qwen`, `gemma`). |
| `--install-models`, `--provision`, `--install-ollama` | `flag` | `False` | Auto-install Ollama (if missing, zero registry PATH mutation), start `ollama serve` detached, dynamically select resource-fitted complementary panel models (e.g., `glm4:9b` + `deepseek-r1:7b` on 8 GB VRAM / 12+ GB RAM hosts), and stream `/api/pull` with live `tqdm` byte progress bars. |
| `--auto-provision` | `flag` | `False` | On-demand auto-provisioning guard for `--verify-panel`, `--verify-llm`, and `--langextract-llm`: automatically installs/starts Ollama and pulls missing resource-fitted models before execution. |
| `--evolves`, `--evolve`, `--evoves` *(also `---evolves`)* | `flag` | `False` | Activate the combined **ShinkaEvolve + OpenEvolve** evolutionary optimization pipeline on the synthesized Python code. |
| `--evolve-mode` | `str` | `"hybrid"` | Pairing mode for `ShinkaEvolve` + `OpenEvolve`: `sequential`, `parallel`, or `hybrid`. |
| `--evolve-count` | `int` | `1` | Number of full evolutionary cycles/rounds to execute `OpenEvolve` + `ShinkaEvolve` (carrying forward champions & MAP-Elites grid across every round). |
| `--generations` | `int` | `4` | Number of evolutionary generations per stage inside each round. |
| `--islands` | `int` | `3` | Number of `OpenEvolve` MAP-Elites islands for behavioral niche exploration. |
| `--elites-dir` | `str` | `None` | Shared filesystem directory for cross-pollinated elites between `ShinkaEvolve` and `OpenEvolve`. |
| `--langgraph`, `--swarm` | `flag` | `False` | Execute the 4-Arm HMAC-SHA256 signed StateGraph Research & Evolution Swarm. |
| `--status` | `flag` | `False` | Print full JSON status report of ArXiv-to-Code, LangExtract, DynaRot, and ShinkaEvolve/OpenEvolve. |
| `--list-tools`, `--mcp-tools` | `flag` | `False` | Print JSON catalog of all 9 registered Universal MCP Server tools and exit. |
| `--mcp` | `flag` | `False` | Start the ArXiv-to-Code stdio JSON-RPC 2.0 Universal MCP Server. |
| `--test` | `flag` | `False` | Run the 9-step automated end-to-end functional self-test suite. |
| `--pytest` | `flag` | `False` | Run the full Pytest unit and integration suite (`tests/`). |
| `-V`, `--version` | `flag` | `—` | Show program version (`arixv_to_code.py 1.0.0`) and exit. |

---

## 2. Complete Flag Combination Matrix (`arixv_to_code.py`)

| Goal | Command |
|:---|:---|
| **Show all flags, DynaRot tiers, pairing modes & examples** | `python arixv_to_code.py --help` |
| **Search arXiv + Hybrid Evolution (3 rounds) + Local LLM Code Verification** | `python arixv_to_code.py --search "LLM container sandbox escape" --max-results 2 --evolves --evolve-mode hybrid --evolve-count 3 --generations 4 --islands 3 --verify-llm` |
| **Search arXiv + Local Open-Weight Ollama Google LangExtract + LLM Verify** | `python arixv_to_code.py --search "multi-agent delegation security" --langextract-llm --langextract-model gemma2:2b --evolves --evolve-count 2 --verify-llm --verify-model qwen2.5:7b` |
| **Specific arXiv ID + Sequential Two-Stage (`ShinkaEvolve` $\to$ `OpenEvolve`)** | `python arixv_to_code.py --arxiv-id 2608.15888 --evolves --evolve-mode sequential --generations 4 --islands 3 --output-dir .\out_seq` |
| **Specific arXiv ID + Parallel Cross-Pollinating Ensemble** | `python arixv_to_code.py --arxiv-id 2603.02277 --evolves --evolve-mode parallel --elites-dir .\shared_elites --output-dir .\out_par` |
| **Specific arXiv ID + LLM-as-a-Judge Verification** | `python arixv_to_code.py --arxiv-id 2608.15888 --verify-llm` |
| **Search arXiv with score threshold & include implemented papers** | `python arixv_to_code.py --search "conformal jailbreak detection" --min-score 60 --max-results 3 --include-implemented` |
| **Extract local PDF / Markdown / Text + Interactive LangExtract HTML + Evolve** | `python arixv_to_code.py --extract-doc .\paper.pdf --html-report .\langextract_audit.html --evolves --output-dir .\out_doc` |
| **Specific arXiv ID + Interactive LangExtract HTML Report** | `python arixv_to_code.py --arxiv-id 2603.02277 --html-report .\audit_2603_02277.html --generate --output-dir .\out_html` |
| **ModelFusion DynaRot Hardware Sizing & Compute Split** | `python arixv_to_code.py --dynarot` |
| **Auto-Install Ollama & Dynamically Pull Resource-Fitted Models (`glm4:9b` + `deepseek-r1:7b`)** | `python arixv_to_code.py --install-models` |
| **Auto-Provision Missing Models On-Demand + Hybrid Evolve + ModelFusion Panel Verify** | `python arixv_to_code.py --arxiv-id 2608.15888 --auto-provision --langextract-llm --evolves --evolve-count 2 --verify-panel` |
| **Multi-Target Defense Coverage Audit** | `python arixv_to_code.py --arxiv-id 2603.02277 --audit` |
| **4-Arm HMAC-Signed LangGraph Research & Evolution Swarm** | `python arixv_to_code.py --arxiv-id 2603.02277 --langgraph --evolve-mode hybrid` |
| **Full JSON Status Report** | `python arixv_to_code.py --status` |
| **List All 9 Universal MCP Tools** | `python arixv_to_code.py --list-tools` |
| **Start Universal MCP Server (stdio JSON-RPC)** | `python arixv_to_code.py --mcp` |
| **Run Built-In 9-Step End-to-End Self-Test** | `python arixv_to_code.py --test` |
| **Run Full Pytest Suite (All Flags + LLM Verifier + MCP + DynaRot + Evolution)** | `python arixv_to_code.py --pytest` |

---

## 3. Testing Every Flag & Executing Generated Champion Code

### A. Run the Exhaustive CLI Flag Combination & LLM Verifier Test Suites
Tests every single flag and alias (`--help`, `--version`, `--search` / `--searct`, `--arxiv-id`, `--max-results`, `--min-score`, `--include-implemented`, `--extract-doc`, `--html-report`, `--langextract-llm`, `--generate`, `--output-dir`, `--audit`, `--verify-llm` / `--verify-model`, `--dynarot` / `--sys-info`, `--evolves` / `---evolves`, all three `--evolve-mode` values, `--evolve-count`, `--generations`, `--islands`, `--elites-dir`, `--no-progress`, `--langgraph`, `--status`, `--list-tools`, and `--test`):

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -v tests/test_all_cli_flags.py tests/test_llm_code_verifier.py
```

### B. Run the Full Pytest Suite & Built-In 9-Step Self-Test
```powershell
python arixv_to_code.py --pytest
python arixv_to_code.py --test
```

### C. Execute Synthesized & Evolved Python Modules Directly
Every generated and evolved Python module is stored inside the organized `generated/` workspace (`INDEX.md` + `MANIFEST.json`) and includes its own deterministic benchmark (`run_deterministic_benchmark()`) and `unittest` test suite:
```powershell
# 1. Generate, evolve (2 rounds), and verify Python code for arXiv:2603.02277
python arixv_to_code.py --arxiv-id 2603.02277 --evolves --evolve-mode hybrid --evolve-count 2 --verify-llm

# 2. Execute the baseline synthesized module and the evolved champion directly
python .\generated\arxiv_2603_02277_quantifying_frontier_llm\src\arxiv_2603_02277_module.py
python .\generated\arxiv_2603_02277_quantifying_frontier_llm\evolved\champion_arxiv_2603_02277.py
```

---

## 4. Verbatim `--help` Output (`python arixv_to_code.py --help` & `python mcp_server.py --help`)

### `python arixv_to_code.py --help`
```text
usage: arixv_to_code.py [-h] [-V] [--search QUERY] [--arxiv-id ARXIV_ID]
                        [--max-results MAX_RESULTS] [--min-score MIN_SCORE]
                        [--include-implemented] [--extract-doc EXTRACT_DOC]
                        [--html-report HTML_REPORT] [--langextract-llm]
                        [--langextract-model LANGEXTRACT_MODEL] [--generate]
                        [--output-dir OUTPUT_DIR] [--audit] [--verify-llm]
                        [--verify-model VERIFY_MODEL] [--dynarot]
                        [--install-models] [--evolves]
                        [--evolve-mode {sequential,parallel,hybrid}]
                        [--evolve-count EVOLVE_COUNT]
                        [--generations GENERATIONS] [--islands ISLANDS]
                        [--elites-dir ELITES_DIR] [--evolve-llm]
                        [--no-progress] [--langgraph] [--status]
                        [--list-tools] [--mcp] [--test] [--pytest]

ArXiv-to-Code All-in-One CLI & Universal MCP Server (arixv_to_code.py v1.0.0)
--------------------------------------------------------------------------------
Autonomous arXiv Research Discovery, Google LangExtract 3-Class Source Grounding,
Zero-Mock Paper-to-Python Code Synthesis (# EVOLVE-BLOCK Convention), ModelFusion
DynaRot Hardware-Aware Model Routing, Combined ShinkaEvolve + OpenEvolve
Evolutionary Optimization, and Fail-Closed LLM Code Verification.

options:
  -h, --help            show this help message and exit
  -V, --version         Show program's version number and exit

1. arXiv Discovery & Multi-Factor Ranking (0-100 Score):
  --search QUERY, --searct QUERY, --seach QUERY, --query QUERY
                        Search query for discovering and ranking arXiv papers
                        (default: 'LLM agent sandbox escape breakout')
  --arxiv-id ARXIV_ID, --id ARXIV_ID
                        Specific arXiv paper ID to fetch, ground, and
                        synthesize into Python (e.g., 2603.02277, 2608.15888,
                        2609.00005)
  --max-results MAX_RESULTS
                        Maximum number of ranked novel arXiv papers to
                        retrieve (default: 3)
  --min-score MIN_SCORE
                        Minimum composite rank score (0.0-100.0) filter
                        threshold (default: 0.0)
  --include-implemented
                        Disable strict novelty filtering and include papers
                        already in CODEBASE_IMPLEMENTED_PAPERS

2. Google LangExtract 3-Class Source Grounding & HTML Reports:
  --extract-doc EXTRACT_DOC, --extract EXTRACT_DOC
                        Path to a local document (.pdf, .txt, .md, .tex) to
                        extract with Google LangExtract and generate Python
                        code
  --html-report HTML_REPORT
                        Write self-contained interactive Google LangExtract
                        HTML visualization (langextract.visualize) to this
                        path
  --langextract-llm     Invoke Google langextract.extract() LLM-backed 3-class
                        grounding (via local Ollama or GEMINI_API_KEY)
  --langextract-model LANGEXTRACT_MODEL
                        Optional model ID for Google langextract.extract()
                        (e.g., 'gemma2:2b', 'qwen2.5:7b', or 'gemini-2.5-flash')

3. Paper-to-Python Production Code Synthesis, Verification & Defense Audit:
  --generate            Synthesize runnable production Python module with #
                        EVOLVE-BLOCK markers (enabled automatically on
                        search/id/extract)
  --output-dir OUTPUT_DIR
                        Root output directory organized into per-paper folders
                        with INDEX.md, MANIFEST.json, src/, eval/, evolved/,
                        verification_report.json, and LangExtract HTML/JSONL
                        (default: 'generated')
  --audit, --audit-defense
                        Run multi-target defense & invariant coverage audit
                        across Python runtime, OshoosiClaw, KillSwitch, and
                        AgentOctopus
  --verify-llm, --llm-verify, --verify
                        Verify generated (and evolved) Python code using a
                        local open-weight LLM judge (Ollama via DynaRot)
                        combined with fail-closed AST, security invariant, and
                        runtime unit-test checks; saves verification_report.json
  --verify-model VERIFY_MODEL
                        Optional LLM model tag for --verify-llm (e.g.,
                        'qwen2.5:7b', 'gemma2:2b', 'deepseek-r1:7b')

4. ModelFusion DynaRot Hardware Sizing & Auto-Provisioning:
  --dynarot, --sys-info
                        Probe live free RAM & free VRAM, compute 4-objective
                        Pareto front & knee-point, and show Shinka/OpenEvolve
                        compute split
  --install-models      Automatically install/pull optimal local Ollama models
                        (ollama pull <model>) based on live available RAM and
                        VRAM

5. Combined ShinkaEvolve + OpenEvolve Evolutionary Search:
  --evolves, --evolve, --evoves
                        Evolve synthesized Python code by combining
                        ShinkaEvolve (UCB1 routing, Meta-Scratchpad, AST
                        Novelty Rejection) and OpenEvolve (Multi-Island 2D
                        MAP-Elites grid)
  --evolve-mode {sequential,parallel,hybrid}
                        Pairing mode for ShinkaEvolve + OpenEvolve:
                        'sequential', 'parallel', or 'hybrid' (default:
                        hybrid)
  --evolve-count EVOLVE_COUNT
                        Number of full evolutionary rounds/cycles to run
                        OpenEvolve + ShinkaEvolve (default: 1)
  --generations GENERATIONS
                        Number of evolutionary generations per stage inside
                        each round (default: 4)
  --islands ISLANDS     Number of OpenEvolve MAP-Elites islands for behavioral
                        niche exploration (default: 3)
  --elites-dir ELITES_DIR
                        Shared filesystem directory for ShinkaEvolve &
                        OpenEvolve cross-pollinated elites
  --evolve-llm          Enable live local Ollama LLM structural synthesis
                        calls during ShinkaEvolve discovery
  --no-progress         Disable tqdm progress bars and console spinners

6. Agentic Swarm, Universal MCP Server & Test Suites:
  --langgraph, --swarm  Execute the 4-Arm HMAC-SHA256 signed ArXiv-to-Code
                        StateGraph Research & Evolutionary Swarm
  --status              Print full JSON status report of ArXiv-to-Code,
                        LangExtract, DynaRot, and ShinkaEvolve/OpenEvolve
  --list-tools, --mcp-tools
                        Print JSON catalog of all 9 registered Universal MCP
                        Server tools and exit
  --mcp                 Start the ArXiv-to-Code Universal MCP Server over
                        stdio JSON-RPC 2.0 (9 Tools)
  --test                Run the 9-step automated end-to-end functional self-
                        test suite
  --pytest              Run the full Pytest unit and integration test suite
                        (tests/)
```

### `python mcp_server.py --help`
```text
usage: arxiv-to-code-mcp [-h] [--list-tools] [-V]

ArXiv-to-Code Universal MCP Server (stdio JSON-RPC 2.0)
Exposes 9 specialized tools for arXiv search, 3-class Google LangExtract character-span
grounding, Paper-to-Python code synthesis, ModelFusion DynaRot hardware-aware routing,
combined ShinkaEvolve + OpenEvolve evolutionary search, and LLM-based code verification.

options:
  -h, --help     show this help message and exit
  --list-tools   Print JSON catalog of all 9 registered MCP tools and exit
  -V, --version  Show MCP server version and exit
```
