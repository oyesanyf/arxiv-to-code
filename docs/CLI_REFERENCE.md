# ArXiv-to-Code CLI Reference Manual

The **ArXiv-to-Code CLI** (`python cli.py` or `arxiv-to-code` after `pip install -e .`) provides command-line access to arXiv discovery, Google LangExtract 3-class character-span grounding, Paper-to-Python code synthesis, ModelFusion DynaRot hardware sizing/provisioning, and combined `ShinkaEvolve` + `OpenEvolve` optimization.

---

## 1. Master Flag Reference Table

| Flag / Alias | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--query`, `--search` | `str` | `"LLM agent sandbox escape breakout"` | Search query for discovering and ranking arXiv papers. |
| `--arxiv-id` | `str` | `None` | Specific arXiv paper ID to fetch, ground, and synthesize into Python (e.g., `2608.15888`, `2603.02277`, `2609.00005`). |
| `--extract-doc` | `str` | `None` | Path to a local document (`.pdf`, `.txt`, `.md`, `.tex`) to extract with Google LangExtract and synthesize into Python code. |
| `--max-results` | `int` | `3` | Maximum number of ranked novel arXiv papers to return. |
| `--min-score` | `float` | `0.0` | Minimum composite rank score (`0.0–100.0`) threshold. |
| `--include-implemented` | `flag` | `False` | Disable strict novelty filtering and include papers listed in `CODEBASE_IMPLEMENTED_PAPERS`. |
| `--html-report` | `str` | `None` | Output path for self-contained Google LangExtract interactive HTML visualization (`langextract.visualize()`). |
| `--output-dir` | `str` | `None` | Directory to write generated Python modules (`arxiv_<id>_module.py`) and deterministic scoring scripts (`evaluate_arxiv_<id>.py`). |
| `--dynarot`, `--sys-info` | `flag` | `False` | Probe live free RAM & free VRAM, compute 4-objective Pareto front & knee-point, and print the ShinkaEvolve/OpenEvolve compute split. |
| `--install-models` | `flag` | `False` | Automatically provision/pull (`ollama pull <model>`) the DynaRot hardware-sized models for current available memory. |
| `--evolve`, `--evolves`, `--evoves` | `flag` | `False` | Activate the combined **ShinkaEvolve + OpenEvolve** evolutionary optimization pipeline on the synthesized Python code. |
| `--evolve-mode` | `str` | `"hybrid"` | Pairing mode for `ShinkaEvolve` + `OpenEvolve`: `sequential`, `parallel`, or `hybrid`. |
| `--generations` | `int` | `4` | Number of evolutionary generations per stage. |
| `--islands` | `int` | `3` | Number of `OpenEvolve` MAP-Elites islands for behavioral niche exploration. |
| `--elites-dir` | `str` | `None` | Shared filesystem directory for cross-pollinated elites between `ShinkaEvolve` and `OpenEvolve`. |
| `--langgraph` | `flag` | `False` | Execute the 4-Arm HMAC-SHA256 signed StateGraph Research & Evolution Swarm. |
| `--mcp` | `flag` | `False` | Start the ArXiv-to-Code stdio JSON-RPC 2.0 Universal MCP Server. |
| `--test` | `flag` | `False` | Run the 8-step automated end-to-end functional self-test suite. |

---

## 2. Executable CLI Workflows

### A. Automated 8-Step Self-Test
```powershell
python cli.py --test
```

### B. Search arXiv & Synthesize Python Modules
```powershell
python cli.py --query "multi-agent delegation security sandbox" --max-results 3 --output-dir .\generated_code
```

### C. Specific Paper-to-Python + Combined Evolutionary Optimization (`--evolves`)
```powershell
# 1. Two-Stage Sequential Pipeline (ShinkaEvolve discovery -> seeds OpenEvolve MAP-Elites)
python cli.py --arxiv-id 2608.15888 --evolves --evolve-mode sequential --generations 4 --islands 3 --output-dir .\out_seq

# 2. Parallel Cross-Pollinating Ensemble (Shared filesystem elites directory)
python cli.py --arxiv-id 2603.02277 --evolves --evolve-mode parallel --elites-dir .\shared_elites

# 3. Hybrid Pipeline (Sequential discovery + fine-tuning followed by parallel cross-pollination)
python cli.py --arxiv-id 2609.00005 --evolves --evolve-mode hybrid --output-dir .\out_hybrid
```

### D. Hardware Inspection & Model Auto-Installation (`--dynarot` / `--install-models`)
```powershell
# Print live RAM/VRAM telemetry, Pareto front, UCB1 scores, and Shinka/OpenEvolve model split
python cli.py --dynarot

# Auto-pull optimal models in Ollama for your current free RAM/VRAM
python cli.py --install-models
```

### E. Local PDF / Text Document LangExtract Grounding + HTML Report
```powershell
python cli.py --extract-doc .\my_paper.pdf --html-report .\langextract_report.html --evolves
```

---

## 3. Verbatim `--help` Output (`python cli.py --help` & `python mcp_server.py --help`)

### `python cli.py --help`
```text
usage: arxiv-to-code [-h] [-V] [--query QUERY] [--arxiv-id ARXIV_ID]
                     [--max-results MAX_RESULTS] [--min-score MIN_SCORE]
                     [--include-implemented] [--extract-doc EXTRACT_DOC]
                     [--html-report HTML_REPORT] [--output-dir OUTPUT_DIR]
                     [--dynarot] [--install-models] [--evolve]
                     [--evolve-mode {sequential,parallel,hybrid}]
                     [--generations GENERATIONS] [--islands ISLANDS]
                     [--elites-dir ELITES_DIR] [--langgraph] [--mcp] [--test]

ArXiv-to-Code CLI & Universal MCP Server (v1.0.0)
--------------------------------------------------------------------------------
Autonomous arXiv Research Discovery, Google LangExtract 3-Class Source Grounding,
Zero-Mock Paper-to-Python Code Synthesis (# EVOLVE-BLOCK Convention), ModelFusion
DynaRot Hardware-Aware Model Routing, and Combined ShinkaEvolve + OpenEvolve
Evolutionary Optimization.

options:
  -h, --help            show this help message and exit
  -V, --version         Show program's version number and exit

1. arXiv Discovery & Multi-Factor Ranking (0-100 Score):
  --query QUERY, --search QUERY
                        Search query for discovering and ranking arXiv papers
                        (default: 'LLM agent sandbox escape breakout')
  --arxiv-id ARXIV_ID   Specific arXiv paper ID to fetch, ground, and
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
  --extract-doc EXTRACT_DOC
                        Path to a local document (.pdf, .txt, .md, .tex) to
                        extract with Google LangExtract and generate Python
                        code
  --html-report HTML_REPORT
                        Write self-contained interactive Google LangExtract
                        HTML visualization (langextract.visualize) to this
                        path

3. Paper-to-Python Production Code Synthesis:
  --output-dir OUTPUT_DIR
                        Directory to write generated Python modules
                        (arxiv_<id>_module.py) and deterministic scoring
                        scripts

4. ModelFusion DynaRot Hardware Sizing & Auto-Provisioning:
  --dynarot, --sys-info
                        Probe live free RAM & free VRAM, compute 4-objective
                        Pareto front & knee-point, and show Shinka/OpenEvolve
                        compute split
  --install-models      Automatically install/pull optimal local Ollama models
                        (ollama pull <model>) based on live available RAM and
                        VRAM

5. Combined ShinkaEvolve + OpenEvolve Evolutionary Search:
  --evolve, --evolves, --evoves
                        Evolve synthesized Python code by combining
                        ShinkaEvolve (UCB1 routing, Meta-Scratchpad, AST
                        Novelty Rejection) and OpenEvolve (Multi-Island 2D
                        MAP-Elites grid)
  --evolve-mode {sequential,parallel,hybrid}
                        Pairing mode for ShinkaEvolve + OpenEvolve:
                        'sequential', 'parallel', or 'hybrid' (default:
                        hybrid)
  --generations GENERATIONS
                        Number of evolutionary generations per stage (default:
                        4)
  --islands ISLANDS     Number of OpenEvolve MAP-Elites islands for behavioral
                        niche exploration (default: 3)
  --elites-dir ELITES_DIR
                        Shared filesystem directory for ShinkaEvolve &
                        OpenEvolve cross-pollinated elites

6. Agentic Swarm, Universal MCP Server & Self-Test Suite:
  --langgraph           Execute the 4-Arm HMAC-SHA256 signed ArXiv-to-Code
                        StateGraph Research & Evolutionary Swarm
  --mcp                 Start the ArXiv-to-Code Universal MCP Server over
                        stdio JSON-RPC 2.0 (8 Tools)
  --test                Run the 8-step automated end-to-end functional self-
                        test suite
```

### `python mcp_server.py --help`
```text
usage: arxiv-to-code-mcp [-h] [--list-tools] [-V]

ArXiv-to-Code Universal MCP Server (stdio JSON-RPC 2.0)
Exposes 8 specialized tools for arXiv search, 3-class Google LangExtract character-span
grounding, Paper-to-Python code synthesis, ModelFusion DynaRot hardware-aware routing,
and combined ShinkaEvolve + OpenEvolve evolutionary search.

options:
  -h, --help     show this help message and exit
  --list-tools   Print JSON catalog of all 8 registered MCP tools and exit
  -V, --version  Show MCP server version and exit
```
