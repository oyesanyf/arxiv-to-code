"""
End-to-end verification tests for LLM-Powered Generated Code Verification (`llm_verifier.py`).

Tests:
  1. Deterministic Layer-1 AST, security invariant, anti-mock, and runtime unit-test gate
  2. Strict JSON response parsing (`parse_llm_verification_json`)
  3. Live local HTTP Ollama endpoint integration (real loopback TCP server, zero mocks):
     - LLM PASS on valid synthesized code
     - Fail-closed override: LLM PASS cannot override dangerous `os.system()` / failing tests
     - LLM FAIL on valid code propagates findings & recommendations
     - Offline Ollama fallback returns explicit `ollama pull <model>` guidance
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple

from arxiv_to_code.dynarot import _OLLAMA_PROBE_CACHE
from arxiv_to_code.langextract_engine import extract_grounded_breakouts, fetch_paper_by_id
from arxiv_to_code.llm_verifier import (
    parse_llm_verification_json,
    run_deterministic_code_verification,
    verify_generated_python_code,
)
from arxiv_to_code.python_code_generator import synthesize_production_python_code


def _start_local_ollama_test_server(
    installed_models: List[str],
    llm_json_response: Dict[str, Any],
) -> Tuple[HTTPServer, str]:
    """Starts a real loopback HTTP server implementing Ollama /api/tags and /api/generate."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/tags":
                payload = json.dumps(
                    {"models": [{"name": m} for m in installed_models]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/generate":
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                payload = json.dumps(
                    {
                        "model": installed_models[0] if installed_models else "qwen2.5:7b",
                        "response": json.dumps(llm_json_response),
                        "done": True,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{port}"


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_deterministic_verification_passes_valid_synthesized_code() -> None:
    paper = fetch_paper_by_id("2608.15888")
    assert paper is not None
    extractions, _ = extract_grounded_breakouts(paper)
    bundle = synthesize_production_python_code(paper, extractions)

    det = run_deterministic_code_verification(
        source_code=bundle["production_python_code"],
        arxiv_id="2608.15888",
    )
    assert det["ast_syntax_valid"] is True
    assert det["evolve_block_present"] is True
    assert det["constant_time_hmac_used"] is True
    assert det["meet_semilattice_present"] is True
    assert det["zero_dangerous_calls"] is True
    assert det["zero_mock_single_token_indicators"] is True
    assert det["runtime_tests_passed"] == det["runtime_tests_total"] == 4
    assert det["deterministic_score"] == 100.0


def test_deterministic_verification_rejects_dangerous_and_mock_code() -> None:
    bad_code = (
        "import os, subprocess\n"
        "MULTI_TOKEN_TECHNICAL_INDICATORS = ('powershell.exe', 'bash')\n"
        "os.system('echo pwned')\n"
        "subprocess.run('dir', shell=True)\n"
        "eval('1+1')\n"
    )
    det = run_deterministic_code_verification(bad_code, arxiv_id="bad-test")
    assert det["ast_syntax_valid"] is True
    assert det["zero_dangerous_calls"] is False
    assert "os.system" in det["dangerous_calls_found"]
    assert "eval" in det["dangerous_calls_found"]
    assert "subprocess.run(shell=True)" in det["dangerous_calls_found"]
    assert det["zero_mock_single_token_indicators"] is False
    assert "powershell.exe" in det["mock_indicators_found"]

    syntax_bad = run_deterministic_code_verification("def broken(:\n  pass\n", arxiv_id="syn-err")
    assert syntax_bad["ast_syntax_valid"] is False
    assert syntax_bad["runtime_error"] is not None


def test_parse_llm_verification_json_variants() -> None:
    fenced = """```json
    {
      "verdict": "pass",
      "paper_fidelity_score": 115,
      "correctness_score": 92.5,
      "security_score": -10,
      "summary": "Faithfully implements meet-semilattice and conformal bounds.",
      "grounded_span_coverage": ["meet-semilattice", "conformal bound"],
      "findings": [
        {"severity": "LOW", "category": "performance", "message": "Minor alloc", "recommendation": "Use tuple"}
      ]
    }
    ```"""
    parsed = parse_llm_verification_json(fenced)
    assert parsed is not None
    assert parsed["verdict"] == "PASS"
    assert parsed["paper_fidelity_score"] == 100.0
    assert parsed["correctness_score"] == 92.5
    assert parsed["security_score"] == 0.0
    assert len(parsed["findings"]) == 1

    assert parse_llm_verification_json("not valid json at all") is None


def test_live_loopback_ollama_llm_verifier_pass_fail_and_fail_closed() -> None:
    paper = fetch_paper_by_id("2603.02277")
    assert paper is not None
    extractions, _ = extract_grounded_breakouts(paper)
    bundle = synthesize_production_python_code(paper, extractions)

    # 1. Live loopback Ollama PASS on valid code
    srv_pass, endpoint_pass = _start_local_ollama_test_server(
        installed_models=["qwen2.5:7b"],
        llm_json_response={
            "verdict": "PASS",
            "paper_fidelity_score": 94,
            "correctness_score": 96,
            "security_score": 98,
            "summary": "Verified constant-time HMAC, container escape indicators, and martingale barrier.",
            "grounded_span_coverage": ["container", "sandbox", "Docker/OCI"],
            "findings": [],
        },
    )
    try:
        _OLLAMA_PROBE_CACHE.pop(endpoint_pass, None)
        rep = verify_generated_python_code(
            source_code=bundle["production_python_code"],
            paper=paper,
            extractions=extractions,
            code_stage="synthesized",
            model_id="qwen2.5:7b",
            ollama_endpoint=endpoint_pass,
            enable_llm=True,
        )
        assert rep.llm_verified is True
        assert rep.verifier_provider == "ollama"
        assert rep.verifier_model == "qwen2.5:7b"
        assert rep.verification_mode == "llm_and_deterministic_verified"
        assert rep.verdict == "PASS"
        assert rep.overall_score >= 95.0
        assert "LLM Paper Fidelity Score" in rep.to_markdown()

        # 2. Fail-closed override: even if LLM says PASS, dangerous os.system() MUST force FAIL
        rep_fail_closed = verify_generated_python_code(
            source_code="import os\nos.system('id')\n",
            paper=paper,
            extractions=extractions,
            code_stage="custom",
            model_id="qwen2.5:7b",
            ollama_endpoint=endpoint_pass,
            enable_llm=True,
        )
        assert rep_fail_closed.llm_verified is True
        assert rep_fail_closed.verdict == "FAIL"
        assert rep_fail_closed.overall_score <= 45.0
    finally:
        srv_pass.shutdown()

    # 3. Live loopback Ollama FAIL on valid code propagates findings & recommendations
    srv_fail, endpoint_fail = _start_local_ollama_test_server(
        installed_models=["gemma2:2b"],
        llm_json_response={
            "verdict": "FAIL",
            "paper_fidelity_score": 40,
            "correctness_score": 65,
            "security_score": 80,
            "summary": "Missing explicit seccomp syscall filter mapping from paper section 4.",
            "grounded_span_coverage": ["sandbox"],
            "findings": [
                {
                    "severity": "HIGH",
                    "category": "fidelity",
                    "message": "Seccomp BPF filter table omitted.",
                    "recommendation": "Add explicit seccomp syscall allowlist check.",
                }
            ],
        },
    )
    try:
        _OLLAMA_PROBE_CACHE.pop(endpoint_fail, None)
        rep_llm_fail = verify_generated_python_code(
            source_code=bundle["production_python_code"],
            paper=paper,
            extractions=extractions,
            model_id="gemma2:2b",
            ollama_endpoint=endpoint_fail,
            enable_llm=True,
        )
        assert rep_llm_fail.llm_verified is True
        assert rep_llm_fail.verdict == "FAIL"
        assert any("Seccomp BPF" in f["message"] for f in rep_llm_fail.findings)
        assert any("seccomp syscall allowlist" in r for r in rep_llm_fail.recommendations)
    finally:
        srv_fail.shutdown()

    # 4. Offline Ollama fallback gives actionable `ollama pull` guidance while still passing deterministic gate
    offline_ep = f"http://127.0.0.1:{_unused_local_port()}"
    _OLLAMA_PROBE_CACHE.pop(offline_ep, None)
    rep_offline = verify_generated_python_code(
        source_code=bundle["production_python_code"],
        paper=paper,
        extractions=extractions,
        ollama_endpoint=offline_ep,
        enable_llm=True,
    )
    assert rep_offline.llm_verified is False
    assert rep_offline.verification_mode == "deterministic_static_and_runtime_fallback"
    assert rep_offline.verdict == "PASS"
    assert rep_offline.llm_audit["status"] == "ollama_offline_fallback"
    assert "ollama pull" in rep_offline.llm_audit["ollama_guidance"]["pull_command"]
