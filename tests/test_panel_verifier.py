"""
End-to-end verification tests for ModelFusion Panel-of-Models Code Verification (`panel_verifier.py`).

Tests:
  1. Dynamic panel sizing (`derive_fusion_model_count`) across all 5 RAM/VRAM tiers
  2. Memory-aware local batching (`calculate_panel_batch_size`: >=20GB -> 4, >=11GB -> 2, <11GB -> 1)
  3. Resource-gated GLM panel activation (`assess_glm_for_panel`: >=8GB RAM or >=6GB VRAM floor)
  4. Panel assembly (`assemble_verification_panel`): multi-model, multi-sample, Bespoke-Nimble exclusion
  5. `<think>...</think>` stripping & ModelFusion `JudgeAnalysis` JSON parsing
  6. Live loopback HTTP multi-model Ollama server (3 distinct models: `glm-4.7-flash`, `qwen2.5:7b`, `deepseek-r1:7b`):
     - Unanimous PASS panel + LLM Judge Analysis (`consensus`, `unique_insights`)
     - Minority-Quality Veto: 2 models vote PASS, 1 minority DeepSeek-R1 (`<think>`) reports CRITICAL FAIL -> FAIL + `minority_override_triggered=True`
     - Fail-closed deterministic override: all 3 panel models + Judge vote PASS on `os.system()` -> FAIL
  7. CLI `--verify-panel --panel-size 3 --panel-mode multi-model` & MCP Tool 9 (`use_panel=True`)
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from arxiv_to_code.cli import main as cli_main, normalize_cli_argv
from arxiv_to_code.dynarot import DynaRotModelRouter, _OLLAMA_PROBE_CACHE
from arxiv_to_code.langextract_engine import extract_grounded_breakouts, fetch_paper_by_id
from arxiv_to_code.mcp_server import verify_generated_python_code as mcp_verify_tool
from arxiv_to_code.models import SystemHardwareResources
from arxiv_to_code.panel_verifier import (
    assess_glm_for_panel,
    assemble_verification_panel,
    calculate_panel_batch_size,
    derive_fusion_model_count,
    parse_judge_analysis_json,
    run_model_panel_verification,
    strip_think_tags,
)
from arxiv_to_code.python_code_generator import synthesize_production_python_code


def _start_multi_model_ollama_server(
    installed_models: List[str],
    responder: Callable[[str, str], str],
) -> Tuple[HTTPServer, str]:
    """Starts a real loopback HTTP server simulating multi-model Ollama /api/tags + /api/generate."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/tags":
                payload = json.dumps({"models": [{"name": m} for m in installed_models]}).encode("utf-8")
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
                req_data = json.loads(self.rfile.read(length).decode("utf-8"))
                model_name = str(req_data.get("model", ""))
                prompt_text = str(req_data.get("prompt", ""))
                raw_resp = responder(model_name, prompt_text)
                payload = json.dumps(
                    {"model": model_name, "response": raw_resp, "done": True}
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


def _hw(free_ram: float, free_vram_gb: float = 0.0, gpu_name: str | None = None) -> SystemHardwareResources:
    return SystemHardwareResources(
        total_ram_gb=max(16.0, free_ram),
        free_ram_gb=free_ram,
        cpu_cores=16,
        gpu_name=gpu_name or ("RTX GPU" if free_vram_gb > 0 else None),
        gpu_vram_total_gb=free_vram_gb,
        gpu_vram_free_gb=free_vram_gb,
        free_disk_gb=200.0,
        gpu_count=1 if free_vram_gb > 0 else 0,
    )


def test_dynamic_panel_count_batch_size_and_glm_gate() -> None:
    # ModelFusion derive_fusion_model_count exact tiers
    assert derive_fusion_model_count(_hw(68.0, 0.0)) == 7
    assert derive_fusion_model_count(_hw(16.0, 24.0)) == 7
    assert derive_fusion_model_count(_hw(36.0, 0.0)) == 5
    assert derive_fusion_model_count(_hw(18.0, 0.0)) == 4
    assert derive_fusion_model_count(_hw(10.0, 0.0)) == 3
    assert derive_fusion_model_count(_hw(5.0, 0.0)) == 2

    # ModelFusion calculate_panel_batch_size exact VRAM tiers
    assert calculate_panel_batch_size(5, _hw(32.0, 24.0, "RTX 4090")) == 4
    assert calculate_panel_batch_size(5, _hw(32.0, 12.0, "RTX 4070")) == 2
    assert calculate_panel_batch_size(5, _hw(32.0, 7.0, "Quadro RTX 4000")) == 1
    assert calculate_panel_batch_size(5, _hw(32.0, 0.0, None)) == 1

    # Resource-gated GLM panel activation floor (Free RAM >= 8 GB or Free VRAM >= 6 GB)
    assert assess_glm_for_panel(_hw(6.0, 4.0))["activated"] is False
    glm_mid = assess_glm_for_panel(_hw(10.0, 0.0))
    assert glm_mid["activated"] is True and glm_mid["ollama_tag"] == "glm4:9b"
    glm_hi = assess_glm_for_panel(_hw(24.0, 0.0))
    assert glm_hi["activated"] is True and glm_hi["ollama_tag"] == "glm-4.7-flash"


def test_panel_assembly_multi_model_multi_sample_and_think_parsing() -> None:
    router = DynaRotModelRouter()
    decision = router.route(override_free_ram_gb=32.0, override_free_vram_mb=8000.0)
    hw = _hw(32.0, 8.0, "Quadro RTX 4000")

    installed = [
        "hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q4_K_M",  # decision-only -> excluded
        "glm4:9b",
        "qwen2.5:7b",
    ]
    members, fb, judge_model, meta = assemble_verification_panel(
        hw=hw,
        installed_models=installed,
        dynarot_decision=decision,
        requested_panel_size=3,
        panel_mode="multi-model",
    )
    assert len(members) == 3
    # Bespoke-Nimble-9B excluded; slot 1 & 2 are distinct installed models, slot 3 supplements with variation
    assert all("nimble" not in m.model_tag.lower() for m in members)
    assert {m.model_tag for m in members[:2]} == {"glm4:9b", "qwen2.5:7b"}
    assert meta["batch_size"] == 1
    assert judge_model in ("glm4:9b", "qwen2.5:7b")

    # Multi-sample mode: 3 diverse temperature/persona samples of top model
    ms_members, _, _, ms_meta = assemble_verification_panel(
        hw=hw,
        installed_models=["glm4:9b"],
        dynarot_decision=decision,
        requested_panel_size=3,
        panel_mode="multi-sample",
    )
    assert len(ms_members) == 3
    assert len({m.temperature for m in ms_members}) == 3
    assert ms_meta["panel_mode"] == "multi-sample"

    # <think> stripping & JudgeAnalysis JSON parsing
    raw_think = (
        "<think>DeepSeek chain of thought comparing GLM and Qwen...</think>\n"
        '```json\n{"judge_verdict": "PASS", "minority_override_triggered": false, '
        '"consensus": ["HMAC verified"], "disagreements": [], "unique_insights": ["GLM verified 128K scope"], '
        '"blind_spots": [], "risk_flags": [], "recommended_final_position": "Code verified."}\n```'
    )
    assert "<think>" not in strip_think_tags(raw_think)
    parsed_j = parse_judge_analysis_json(raw_think)
    assert parsed_j is not None
    assert parsed_j["judge_verdict"] == "PASS"
    assert parsed_j["consensus"] == ["HMAC verified"]


def test_live_loopback_3_model_panel_unanimous_minority_veto_and_fail_closed() -> None:
    paper = fetch_paper_by_id("2608.15888")
    assert paper is not None
    extractions, _ = extract_grounded_breakouts(paper)
    bundle = synthesize_production_python_code(paper, extractions)

    models = ["glm-4.7-flash", "qwen2.5:7b", "deepseek-r1:7b"]

    # Scenario 1: Unanimous 3-model PASS + LLM Judge Analysis
    def _unanimous_responder(model_name: str, prompt: str) -> str:
        if "Expert Judge Model" in prompt:
            return (
                "<think>Comparing all three panelist audits.</think>\n"
                + json.dumps(
                    {
                        "judge_verdict": "PASS",
                        "minority_override_triggered": False,
                        "consensus": ["All 3 models verified meet-semilattice & HMAC-SHA256."],
                        "disagreements": [],
                        "unique_insights": ["GLM-4.7-Flash verified pairwise composition lineage."],
                        "blind_spots": [],
                        "risk_flags": [],
                        "recommended_final_position": "Unanimously verified across GLM, Qwen, and DeepSeek.",
                    }
                )
            )
        return json.dumps(
            {
                "verdict": "PASS",
                "paper_fidelity_score": 95,
                "correctness_score": 96,
                "security_score": 97,
                "summary": f"{model_name} verified meet-semilattice and constant-time HMAC.",
                "grounded_span_coverage": ["meet-semilattice", "delegation"],
                "findings": [],
            }
        )

    srv1, ep1 = _start_multi_model_ollama_server(models, _unanimous_responder)
    try:
        _OLLAMA_PROBE_CACHE.pop(ep1, None)
        rep = run_model_panel_verification(
            source_code=bundle["production_python_code"],
            paper=paper,
            extractions=extractions,
            panel_size=3,
            panel_models=models,
            ollama_endpoint=ep1,
            enable_llm=True,
        )
        assert rep.llm_verified is True
        assert rep.verification_mode == "panel_llm_and_deterministic_verified"
        assert rep.verdict == "PASS"
        assert rep.overall_score >= 95.0
        pa = rep.llm_audit["panel_audit"]
        assert pa["responded_panelists"] == 3
        assert pa["judge_analysis"]["judged_by_llm"] is True
        assert "Unanimously verified" in pa["judge_analysis"]["recommended_final_position"]

        # Fail-closed override: even if all 3 panel models + Judge say PASS, os.system() MUST FAIL
        rep_bad = run_model_panel_verification(
            source_code="import os\nos.system('whoami')\n",
            paper=paper,
            extractions=extractions,
            panel_size=3,
            panel_models=models,
            ollama_endpoint=ep1,
            enable_llm=True,
        )
        assert rep_bad.verdict == "FAIL"
        assert rep_bad.overall_score <= 45.0
    finally:
        srv1.shutdown()

    # Scenario 2: Minority-Quality Veto (2 PASS vs 1 CRITICAL FAIL -> MUST override majority to FAIL)
    def _minority_veto_responder(model_name: str, prompt: str) -> str:
        if "Expert Judge Model" in prompt:
            # Even if Judge initially says PASS, fail-closed minority CRITICAL veto enforces FAIL
            return json.dumps(
                {
                    "judge_verdict": "PASS",
                    "minority_override_triggered": False,
                    "consensus": ["Majority voted PASS"],
                    "disagreements": ["DeepSeek flagged critical replay flaw"],
                    "unique_insights": ["DeepSeek-R1 found replay attack"],
                    "blind_spots": [],
                    "risk_flags": ["Approval token replay"],
                    "recommended_final_position": "Majority passed.",
                }
            )
        if "deepseek" in model_name:
            return "<think>Inspecting approval token replay...</think>" + json.dumps(
                {
                    "verdict": "FAIL",
                    "paper_fidelity_score": 60,
                    "correctness_score": 55,
                    "security_score": 30,
                    "summary": "Detected CRITICAL nonce replay window in delegation envelope.",
                    "grounded_span_coverage": ["delegation"],
                    "findings": [
                        {
                            "severity": "CRITICAL",
                            "category": "security",
                            "message": "Missing monotonic nonce check on delegation approval token.",
                            "recommendation": "Track monotonic nonce per session.",
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "verdict": "PASS",
                "paper_fidelity_score": 92,
                "correctness_score": 94,
                "security_score": 90,
                "summary": f"{model_name} approved.",
                "grounded_span_coverage": ["delegation"],
                "findings": [],
            }
        )

    srv2, ep2 = _start_multi_model_ollama_server(models, _minority_veto_responder)
    try:
        _OLLAMA_PROBE_CACHE.pop(ep2, None)
        rep_veto = run_model_panel_verification(
            source_code=bundle["production_python_code"],
            paper=paper,
            extractions=extractions,
            panel_size=3,
            panel_models=models,
            ollama_endpoint=ep2,
            enable_llm=True,
        )
        assert rep_veto.verdict == "FAIL"
        ja = rep_veto.llm_audit["panel_audit"]["judge_analysis"]
        assert ja["minority_override_triggered"] is True
        assert any("CRITICAL" in f["severity"] for f in rep_veto.findings)
    finally:
        srv2.shutdown()


def test_cli_verify_panel_flag_and_mcp_tool_panel_mode(tmp_path: Path, capsys) -> None:
    assert normalize_cli_argv(["--panel", "--fusion-verify"]) == ["--verify-panel", "--verify-panel"]

    out_dir = tmp_path / "panel_out"
    rc = cli_main(
        [
            "--arxiv-id",
            "2603.02277",
            "--verify-panel",
            "--panel-size",
            "3",
            "--panel-mode",
            "multi-model",
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "[MODELFUSION PANEL VERIFIER]" in out
    assert "ModelFusion Panel" in out
    reports = list(out_dir.glob("arxiv_2603_02277_*/verification_report.json"))
    assert len(reports) == 1
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    assert "panel_audit" in data["llm_audit"]
    assert data["llm_audit"]["panel_audit"]["effective_panel_size"] == 3

    # MCP Tool 9 with use_panel=True
    mcp_raw = asyncio.run(
        mcp_verify_tool(
            arxiv_id="2608.15888",
            use_panel=True,
            panel_size=3,
            enable_llm=False,
        )
    )
    mcp_data = json.loads(mcp_raw)
    assert mcp_data["verdict"] == "PASS"
    assert mcp_data["llm_audit"]["panel_audit"]["effective_panel_size"] == 3
