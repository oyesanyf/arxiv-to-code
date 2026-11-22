# ArXiv-to-Code Universal MCP Server Guide

The **ArXiv-to-Code MCP Server** (`mcp_server.py` or `python arixv_to_code.py --mcp`) exposes **9 specialized tools** over the Model Context Protocol (stdio JSON-RPC 2.0), enabling any AI IDE or desktop agent (Google Antigravity, Cursor, VS Code, Windsurf, Claude Desktop, Zed) to search arXiv, run 3-class Google LangExtract grounding, synthesize production Python code, route/install local models via ModelFusion DynaRot, evolve code with `ShinkaEvolve` + `OpenEvolve`, and verify generated code using a fail-closed Deterministic AST + Local Open-Weight LLM Judge.

---

## 1. IDE & Agent Configuration Snippets

### Google Antigravity / Claude Desktop / Cursor / Windsurf (`mcp.json`)
```json
{
  "mcpServers": {
    "arxiv-to-code": {
      "command": "C:\\Users\\oyesanyf\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "args": ["d:\\harfile\\arixiv-to-code\\mcp_server.py"],
      "env": {
        "LOCAL_OLLAMA_ENDPOINT": "http://127.0.0.1:11434"
      }
    }
  }
}
```

---

## 2. Complete 9-Tool Reference Catalog

### Tool 1: `arxiv_to_code_status`
Returns server health, Python/LangExtract/arXiv package versions, active ModelFusion DynaRot hardware tier and model split, ranking factor weights, and novelty blacklist.
- **Arguments**: None

### Tool 2: `dynarot_hardware_route_and_install`
Probes live runtime free RAM (`free_ram_gb`) and free GPU VRAM (`free_vram_mb`), computes the 4-objective Pareto front and geometric knee-point model, splits compute between `shinka_heavy_reasoning_model` and `openevolve_fast_mutation_model`, and optionally executes `ollama pull` for missing models.
- **Arguments**:
  - `prompt` (`str`, default: `"Synthesize production Python code from arXiv paper and evolve with ShinkaEvolve + OpenEvolve"`)
  - `auto_install_models` (`bool`, default: `False`)
  - `override_free_ram_gb` (`float | None`, default: `None`)
  - `override_free_vram_mb` (`float | None`, default: `None`)

### Tool 3: `search_and_rank_arxiv_papers`
Searches arXiv, applies strict novelty filtering (`only_novel_enhancements=True`), extracts 3-class character-grounded spans, ranks papers `0–100`, synthesizes production Python code for each paper, and optionally evolves each module when `evolve=True`.
- **Arguments**:
  - `query` (`str`, default: `"LLM agent sandbox escape breakout"`)
  - `max_results` (`int`, default: `5`)
  - `min_score` (`float`, default: `0.0`)
  - `only_novel_enhancements` (`bool`, default: `True`)
  - `evolve` (`bool`, default: `False`)
  - `evolve_mode` (`str`, default: `"hybrid"`)

### Tool 4: `extract_document_langextract`
Extracts 3-class source-grounded spans (`breakout_vector`, `mathematical_formulation`, `defense_countermeasure`) with exact `CharInterval(start_pos, end_pos)` character offsets from an arXiv ID or local file path (`.pdf`, `.txt`, `.md`, `.tex`) and optionally writes an interactive HTML visualization report.
- **Arguments**:
  - `arxiv_id_or_path` (`str`, default: `"2603.02277"`)
  - `output_html_path` (`str`, default: `""`)

### Tool 5: `generate_python_code_from_paper`
Synthesizes a complete, runnable, zero-mock Python module (with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers, deterministic benchmark harness, and `unittest` test suite) from an arXiv ID, optionally writing files to `output_dir` and running `ShinkaEvolve` + `OpenEvolve` when `evolve=True`.
- **Arguments**:
  - `arxiv_id` (`str`, default: `"2608.15888"`)
  - `output_dir` (`str`, default: `""`)
  - `evolve` (`bool`, default: `False`)
  - `evolve_mode` (`str`, default: `"hybrid"`)
  - `generations` (`int`, default: `4`)

### Tool 6: `evolve_python_code`
Runs the combined **ShinkaEvolve + OpenEvolve** evolutionary search pipeline on a standalone Python program (or synthesizes one from `arxiv_id` if `python_source_code` is empty).
- **Arguments**:
  - `arxiv_id` (`str`, default: `"2603.02277"`)
  - `python_source_code` (`str`, default: `""`)
  - `evolve_mode` (`str`, default: `"hybrid"` — accepts `"sequential"`, `"parallel"`, or `"hybrid"`)
  - `generations` (`int`, default: `4`)
  - `num_islands` (`int`, default: `3`)
  - `elites_dir` (`str`, default: `""`)

### Tool 7: `audit_defense_coverage`
Evaluates grounded extractions from an arXiv paper across Python and multi-ring containment boundaries (`python_runtime`, `oshoosiclaw`, `killswitch`, `octopus`) and returns coverage metrics + synthesized Python code.
- **Arguments**:
  - `arxiv_id` (`str`, default: `"2603.02277"`)
  - `targets` (`list[str] | None`, default: `None`)

### Tool 8: `run_langgraph_arxiv_to_code_swarm`
Executes the 4-Arm HMAC-SHA256 signed ArXiv-to-Code StateGraph Swarm (`arm_1_arxiv_scout` $\to$ `arm_2_langextract_grounder` $\to$ `arm_3_python_code_synthesizer` $\to$ `arm_4_shinka_openevolve_optimizer` $\to$ post-evolution verification).
- **Arguments**:
  - `query` (`str`, default: `"LLM agent sandbox escape breakout"`)
  - `arxiv_id` (`str`, default: `"2603.02277"`)
  - `max_results` (`int`, default: `3`)
  - `min_score` (`float`, default: `0.0`)
  - `only_novel_enhancements` (`bool`, default: `True`)
  - `evolve_mode` (`str`, default: `"hybrid"`)

### Tool 9: `verify_generated_python_code`
Verifies generated or evolved Python code against the arXiv paper's grounded spans, mathematical formulations, and security invariants using a fail-closed Deterministic AST + Runtime Test Gate combined with an Open-Weight LLM-as-a-Judge (local Ollama via ModelFusion DynaRot, or optional Gemini API).
- **Arguments**:
  - `arxiv_id` (`str`, default: `"2603.02277"`)
  - `python_source_code` (`str`, default: `""` — auto-synthesizes from `arxiv_id` if omitted)
  - `verify_model` (`str`, default: `""` — auto-sized by ModelFusion DynaRot if omitted)
  - `enable_llm` (`bool`, default: `True`)
