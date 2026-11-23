"""
Tests for Layer-1 Strict Static Type Checking (`mypy --strict`), Randomized
Property-Based Fuzzing (`run_property_based_fuzzing`), and Iterative Auto-Repair
Retry Loop (`auto_repair_generated_code` / `--auto-repair` / MCP `auto_repair=True`).
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, List, Tuple

from arxiv_to_code.cli import main as cli_main, normalize_cli_argv
from arxiv_to_code.code_repair import (
    auto_repair_generated_code,
    is_verification_clean_pass,
)
from arxiv_to_code.dream_rsi_engine import run_dream_rsi_rl_optimization
from arxiv_to_code.dynarot import _OLLAMA_PROBE_CACHE
from arxiv_to_code.evolve_engine import run_combined_shinka_openevolve
from arxiv_to_code.langextract_engine import extract_grounded_breakouts, fetch_paper_by_id
from arxiv_to_code.llm_verifier import (
    run_deterministic_code_verification,
    run_mypy_type_checker,
    run_property_based_fuzzing,
    run_static_code_linters,
    verify_generated_python_code,
)
from arxiv_to_code.mcp_server import verify_generated_python_code as mcp_verify_tool
from arxiv_to_code.python_code_generator import synthesize_production_python_code


def _start_loopback_ollama_server(
    models: List[str],
    responder: Callable[[str, str], str],
) -> Tuple[HTTPServer, str]:
    """Starts a real loopback HTTP Ollama server supporting `/api/tags` and `/api/generate`."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/api/tags":
                payload = json.dumps({"models": [{"name": m} for m in models]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
            body = json.loads(raw_body) if raw_body else {}
            model_name = str(body.get("model", models[0] if models else "glm4:9b"))
            prompt = str(body.get("prompt", ""))
            reply_text = responder(model_name, prompt)
            payload = json.dumps({"model": model_name, "response": reply_text, "done": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, f"http://127.0.0.1:{port}"


def _build_base_2608_module() -> Tuple[dict, list, str]:
    paper = fetch_paper_by_id("2608.15888")
    assert paper is not None
    extractions, _ = extract_grounded_breakouts(paper)
    bundle = synthesize_production_python_code(paper, extractions)
    return paper, extractions, bundle["production_python_code"]


def test_generated_evolved_and_rl_modules_pass_mypy_strict_and_property_fuzzing() -> None:
    paper, _, base_code = _build_base_2608_module()

    evo_report = run_combined_shinka_openevolve(
        seed_python_code=base_code,
        arxiv_id="2608.15888",
        mode="hybrid",
        generations=2,
        num_islands=2,
        evolve_count=1,
        show_progress=False,
    )
    evo_code = evo_report["champion_candidate"]["source_code"]

    rl_report = run_dream_rsi_rl_optimization(
        seed_python_code=evo_code,
        arxiv_id="2608.15888",
        rl_cycles=2,
        rl_budget=8,
        dream_candidates=10,
        use_llm_dreaming=False,
        benchmark_seeds=3,
        show_progress=False,
    )
    rl_code = rl_report["champion_candidate"]["source_code"]

    for stage_label, src in (
        ("synthesized", base_code),
        ("evolved_champion", evo_code),
        ("rl_dream_rsi_champion", rl_code),
    ):
        det = run_deterministic_code_verification(src, arxiv_id=paper["arxiv_id"])
        assert det["ast_syntax_valid"] is True, stage_label
        assert det["linter_passed"] is True, f"{stage_label}: {det['linter_issues']}"
        assert det["mypy_executed"] is True, stage_label
        assert det["mypy_passed"] is True, f"{stage_label}: {det['mypy_issues']}"
        assert det["mypy_issue_count"] == 0, stage_label
        assert det["fuzz_executed"] is True, stage_label
        assert det["fuzz_passed"] is True, f"{stage_label}: {det['fuzz_failures']}"
        assert det["fuzz_properties_checked"] >= 250, stage_label
        assert det["pytest_executed"] is True and det["pytest_passed"] is True, stage_label
        assert det["deterministic_score"] == 100.0, stage_label


def test_mypy_strict_fails_closed_on_untyped_and_type_mismatched_code() -> None:
    paper, extractions, base_code = _build_base_2608_module()

    broken_type_code = (
        base_code
        + "\n\ndef untyped_helper(a, b):\n    bad_num: int = 'not_an_int'\n    return a + b + bad_num\n"
    )
    mypy_res = run_mypy_type_checker(broken_type_code)
    assert mypy_res["mypy_executed"] is True
    assert mypy_res["mypy_passed"] is False
    assert mypy_res["issue_count"] >= 2

    ver_report = verify_generated_python_code(
        source_code=broken_type_code,
        paper=paper,
        extractions=extractions,
        code_stage="custom",
        enable_llm=False,
    )
    assert ver_report.verdict == "FAIL"
    assert ver_report.deterministic_checks["mypy_passed"] is False


def test_property_based_fuzzing_catches_semilattice_cache_collision_and_zero_alpha_bugs() -> None:
    paper, extractions, base_code = _build_base_2608_module()

    # Bug A: Broken wildcard top-element semilattice (`*` & S -> empty set instead of S)
    broken_lattice = base_code.replace(
        "self._meet_set(self.resources, narrower.resources)",
        "self.resources & narrower.resources",
    )
    fuzz_lattice = run_property_based_fuzzing(broken_lattice)
    assert fuzz_lattice["fuzz_passed"] is False
    assert any(f["property"] == "semilattice_top_identity" for f in fuzz_lattice["failures"])
    ver_lattice = verify_generated_python_code(
        source_code=broken_lattice,
        paper=paper,
        extractions=extractions,
        enable_llm=False,
    )
    assert ver_lattice.verdict == "FAIL"

    # Bug B: Stale cache-key length collision `(payload_corpus, len(indicators))`
    stale_cache_kernel = (
        "_CACHE_BAD: Dict[Tuple[str, int], Tuple[str, ...]] = {}\n\n"
        "def scan_and_score_payload(\n"
        "    payload_corpus: str,\n"
        "    indicators: Sequence[str],\n"
        "    prior_wealth: float = 1.0,\n"
        "    alpha: float = 0.01,\n"
        ") -> Tuple[Tuple[str, ...], float, float, bool]:\n"
        "    cache_key = (payload_corpus, len(indicators))\n"
        "    cached = _CACHE_BAD.get(cache_key)\n"
        "    if cached is None:\n"
        "        corpus_lc = payload_corpus.lower()\n"
        "        cached = tuple(pat for pat in indicators if pat.lower() in corpus_lc)\n"
        "        _CACHE_BAD[cache_key] = cached\n"
        "    safe_prior = prior_wealth if (0.0 <= prior_wealth < 1e12) else 1.0\n"
        "    safe_alpha = alpha if alpha > 1e-6 else 1e-6\n"
        "    nc = min(1.0, 0.05 + 0.35 * len(cached))\n"
        "    w = safe_prior * (1.0 + 3.0 * len(cached)) if cached else max(0.05, safe_prior * 0.9)\n"
        "    return cached, round(nc, 4), round(w, 4), (bool(cached) or w >= (1.0 / safe_alpha))\n"
    )
    s_idx = base_code.index("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
    e_idx = base_code.index("# EVOLVE-BLOCK-END")
    broken_cache = base_code[:s_idx] + "\n" + stale_cache_kernel + "\n" + base_code[e_idx:]
    fuzz_cache = run_property_based_fuzzing(broken_cache)
    assert fuzz_cache["fuzz_passed"] is False
    assert any(f["property"] == "scan_soundness_and_cache_isolation" for f in fuzz_cache["failures"])
    lint_cache = run_static_code_linters(broken_cache)
    assert lint_cache["linter_passed"] is False
    assert any(i["code"] == "AST-BUG001" for i in lint_cache["issues"])

    # Bug C: Unguarded `1.0 / alpha` ZeroDivisionError on `alpha=0.0`
    zero_div_code = re.sub(
        r"safe_alpha\s*=\s*[^\n]+",
        "safe_alpha = alpha",
        base_code,
    )
    fuzz_zero = run_property_based_fuzzing(zero_div_code)
    assert fuzz_zero["fuzz_passed"] is False
    assert any(f["property"] == "scan_no_crash" for f in fuzz_zero["failures"])


def test_auto_repair_deterministic_structural_patcher_heals_multi_defect_module_to_pass() -> None:
    paper, extractions, base_code = _build_base_2608_module()

    # Inject 4 simultaneous real defects:
    # 1. Unused import (`import math`)
    # 2. Bare `except:` inside a helper
    # 3. Broken wildcard semilattice (`self.resources & narrower.resources`)
    # 4. Stale cache-key length collision + unsafe `1.0 / alpha`
    buggy_code = base_code.replace("import hashlib\n", "import hashlib\nimport math\n", 1)
    buggy_code = buggy_code.replace(
        "self._meet_set(self.resources, narrower.resources)",
        "self.resources & narrower.resources",
    )
    stale_and_unsafe_kernel = (
        "_CACHE_BAD: Dict[Tuple[str, int], Tuple[str, ...]] = {}\n\n"
        "def scan_and_score_payload(\n"
        "    payload_corpus: str,\n"
        "    indicators: Sequence[str],\n"
        "    prior_wealth: float = 1.0,\n"
        "    alpha: float = 0.01,\n"
        ") -> Tuple[Tuple[str, ...], float, float, bool]:\n"
        "    try:\n"
        "        cache_key = (payload_corpus, len(indicators))\n"
        "    except:\n"
        "        cache_key = (payload_corpus, len(indicators))\n"
        "    cached = _CACHE_BAD.get(cache_key)\n"
        "    if cached is None:\n"
        "        corpus_lc = payload_corpus.lower()\n"
        "        cached = tuple(pat for pat in indicators if pat.lower() in corpus_lc)\n"
        "        _CACHE_BAD[cache_key] = cached\n"
        "    nc = min(1.0, 0.05 + 0.35 * len(cached))\n"
        "    w = prior_wealth * (1.0 + 3.0 * len(cached)) if cached else max(0.05, prior_wealth * 0.9)\n"
        "    return cached, round(nc, 4), round(w, 4), (bool(cached) or w >= (1.0 / alpha))\n"
    )
    s_idx = buggy_code.index("# EVOLVE-BLOCK-START") + len("# EVOLVE-BLOCK-START")
    e_idx = buggy_code.index("# EVOLVE-BLOCK-END")
    buggy_code = buggy_code[:s_idx] + "\n" + stale_and_unsafe_kernel + "\n" + buggy_code[e_idx:]

    # Confirm buggy code genuinely fails before repair
    initial_rep = verify_generated_python_code(
        source_code=buggy_code,
        paper=paper,
        extractions=extractions,
        enable_llm=False,
    )
    assert initial_rep.verdict == "FAIL"
    assert initial_rep.deterministic_checks["linter_passed"] is False
    assert initial_rep.deterministic_checks["fuzz_passed"] is False

    # Run iterative auto-repair (deterministic structural patcher)
    repair_res = auto_repair_generated_code(
        source_code=buggy_code,
        paper=paper,
        extractions=extractions,
        initial_report=initial_rep,
        max_repair_attempts=3,
        enable_llm=False,
    )
    assert repair_res.repaired is True
    assert repair_res.converged_clean is True
    assert repair_res.final_verdict == "PASS"
    assert repair_res.final_score == 100.0
    assert is_verification_clean_pass(repair_res.final_verification_report) is True
    final_det = repair_res.final_verification_report.deterministic_checks
    assert final_det["linter_passed"] is True
    assert final_det["mypy_passed"] is True
    assert final_det["fuzz_passed"] is True
    assert final_det["pytest_passed"] is True
    assert len(repair_res.repair_history) >= 1
    assert repair_res.repair_history[-1].accepted is True


def test_auto_repair_live_loopback_ollama_heals_mypy_error_and_rejects_regression() -> None:
    paper, extractions, clean_code = _build_base_2608_module()

    # Introduce a subtle mypy --strict type error that requires semantic code repair
    buggy_type_code = (
        clean_code
        + "\n\ndef compute_aux_metric(count: int) -> int:\n"
        + "    broken: int = 'wrong_type_literal'\n"
        + "    return count + len(broken)\n"
    )
    seen_prompts: List[str] = []

    def _healing_responder(model_name: str, prompt: str) -> str:
        seen_prompts.append(prompt)
        # If this is an LLM verification audit prompt, return structured PASS JSON
        if '"paper_fidelity_score"' in prompt or "paper_fidelity_score" in prompt:
            return json.dumps(
                {
                    "verdict": "PASS",
                    "paper_fidelity_score": 96,
                    "correctness_score": 97,
                    "security_score": 98,
                    "summary": f"{model_name} verified repaired module cleanly.",
                    "grounded_span_coverage": ["meet-semilattice", "HMAC-SHA256"],
                    "findings": [],
                }
            )
        # Otherwise this is the repair prompt: return cleanly fixed Python inside <think> + fence
        fixed_code = (
            clean_code
            + "\n\ndef compute_aux_metric(count: int) -> int:\n"
            + "    label: str = 'typed_literal'\n"
            + "    return count + len(label)\n"
        )
        return f"<think>Fixing mypy assignment type error on broken: int = str...</think>\n```python\n{fixed_code}```\n"

    srv, endpoint = _start_loopback_ollama_server(["glm4:9b"], _healing_responder)
    try:
        _OLLAMA_PROBE_CACHE.pop(endpoint, None)
        res = auto_repair_generated_code(
            source_code=buggy_type_code,
            paper=paper,
            extractions=extractions,
            max_repair_attempts=2,
            model_id="glm4:9b",
            ollama_endpoint=endpoint,
            enable_llm=True,
        )
        assert res.initial_verdict == "FAIL"
        assert res.repaired is True
        assert res.converged_clean is True
        assert res.final_verdict == "PASS"
        assert res.final_verification_report.deterministic_checks["mypy_passed"] is True
        assert res.final_verification_report.deterministic_checks["fuzz_passed"] is True
        assert res.final_verification_report.llm_verified is True
        assert any("MYPY" in p for p in seen_prompts), "Repair prompt must include exact mypy diagnostics!"
    finally:
        srv.shutdown()

    # Scenario 2: Monotonic Hill-Climbing Rejects Regressive Patch (e.g., LLM injects os.system)
    def _regressive_responder(_model_name: str, prompt: str) -> str:
        if "paper_fidelity_score" in prompt:
            return json.dumps(
                {
                    "verdict": "FAIL",
                    "paper_fidelity_score": 40,
                    "correctness_score": 30,
                    "security_score": 20,
                    "summary": "Defective code.",
                    "grounded_span_coverage": [],
                    "findings": [],
                }
            )
        return "```python\nimport os\nos.system('whoami')\n```\n"

    srv_bad, ep_bad = _start_loopback_ollama_server(["glm4:9b"], _regressive_responder)
    try:
        _OLLAMA_PROBE_CACHE.pop(ep_bad, None)
        res_bad = auto_repair_generated_code(
            source_code=buggy_type_code,
            paper=paper,
            extractions=extractions,
            max_repair_attempts=1,
            model_id="glm4:9b",
            ollama_endpoint=ep_bad,
            enable_llm=True,
        )
        assert res_bad.repaired is False
        assert "os.system" not in res_bad.repaired_source_code
        assert res_bad.repair_history[0].accepted is False
    finally:
        srv_bad.shutdown()


def test_cli_auto_repair_flags_aliases_workspace_and_mcp_tool(tmp_path: Path, capsys, monkeypatch) -> None:
    assert normalize_cli_argv(["--repair", "--fix-code", "--auto-fix"]) == [
        "--auto-repair",
        "--auto-repair",
        "--auto-repair",
    ]

    # Point CLI at a real local loopback Ollama server so the full CLI + HTTP roundtrip runs deterministically fast
    def _cli_pass_responder(model_name: str, _prompt: str) -> str:
        return json.dumps(
            {
                "verdict": "PASS",
                "paper_fidelity_score": 95,
                "correctness_score": 96,
                "security_score": 97,
                "summary": f"{model_name} verified arXiv:2608.15888 module.",
                "grounded_span_coverage": ["meet-semilattice", "HMAC-SHA256"],
                "findings": [],
            }
        )

    srv_cli, ep_cli = _start_loopback_ollama_server(["glm4:9b"], _cli_pass_responder)
    try:
        _OLLAMA_PROBE_CACHE.pop(ep_cli, None)
        monkeypatch.setenv("LOCAL_OLLAMA_ENDPOINT", ep_cli)

        # CLI end-to-end with --auto-repair --max-repairs 2
        out_dir = tmp_path / "repair_out"
        rc = cli_main(
            [
                "--arxiv-id",
                "2608.15888",
                "--verify-llm",
                "--auto-repair",
                "--max-repairs",
                "2",
                "--no-auto-provision",
                "--output-dir",
                str(out_dir),
                "--no-progress",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Mypy=PASS" in out
        assert "Fuzz=PASS" in out
        assert "Auto-Repair Loop" in out

        reports = list(out_dir.glob("arxiv_2608_15888_*/verification_report.json"))
        assert len(reports) == 1
        rep_data = json.loads(reports[0].read_text(encoding="utf-8"))
        assert rep_data["deterministic_checks"]["mypy_passed"] is True
        assert rep_data["deterministic_checks"]["fuzz_passed"] is True
        assert rep_data["auto_repair"]["converged_clean"] is True
    finally:
        srv_cli.shutdown()

    # MCP Tool 9 with auto_repair=True repairs a deliberately buggy custom snippet
    _, _, base_code = _build_base_2608_module()
    buggy_snippet = base_code.replace("import hashlib\n", "import hashlib\nimport math\n", 1).replace(
        "self._meet_set(self.resources, narrower.resources)",
        "self.resources & narrower.resources",
    )
    mcp_raw = asyncio.run(
        mcp_verify_tool(
            arxiv_id="2608.15888",
            python_source_code=buggy_snippet,
            enable_llm=False,
            auto_repair=True,
            max_repair_attempts=3,
        )
    )
    mcp_data = json.loads(mcp_raw)
    assert mcp_data["verdict"] == "PASS"
    assert mcp_data["deterministic_checks"]["linter_passed"] is True
    assert mcp_data["deterministic_checks"]["mypy_passed"] is True
    assert mcp_data["deterministic_checks"]["fuzz_passed"] is True
    assert mcp_data["auto_repair"]["repaired"] is True
    assert mcp_data["auto_repair"]["converged_clean"] is True
    assert "repaired_source_code" in mcp_data
