"""Tests for Dynamic Ollama Auto-Installer, Detached Daemon Launcher & Streaming Model Provisioner."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

from arxiv_to_code.cli import build_arg_parser, normalize_cli_argv
from arxiv_to_code.dynarot import DynaRotModelRouter
from arxiv_to_code.models import SystemHardwareResources
from arxiv_to_code.ollama_provisioner import (
    detect_ollama_executable,
    dynamically_provision_ollama_and_models,
    ensure_models_for_pipeline,
    ensure_ollama_installed,
    pull_ollama_model_streaming,
    select_dynamic_provision_targets,
)


def _hw(ram: float, vram_mb: float, disk: float = 500.0) -> SystemHardwareResources:
    return SystemHardwareResources(
        total_ram_gb=max(ram + 4.0, 16.0),
        free_ram_gb=ram,
        cpu_cores=16,
        gpu_name="Quadro RTX 4000" if vram_mb > 0 else None,
        gpu_vram_total_gb=round(vram_mb / 1024.0 + 1.0, 2) if vram_mb > 0 else 0.0,
        gpu_vram_free_gb=round(vram_mb / 1024.0, 2),
        free_disk_gb=disk,
    )


def test_select_dynamic_provision_targets_across_hardware_tiers() -> None:
    # 1. 8 GB GPU Workstation (Quadro RTX 4000: ~6.9 GB free VRAM, 182 GB free RAM)
    #    -> Fast VRAM/hybrid panel: glm4:9b + deepseek-r1:7b
    tags_8g, plans_8g = select_dynamic_provision_targets(
        hw=_hw(ram=182.93, vram_mb=7075.8, disk=947.5),
        installed_models=["hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q4_K_M"],
        model_family="auto",
        prefer_vram_resident=True,
    )
    assert tags_8g == ["glm4:9b", "deepseek-r1:7b"]
    assert all(p["disk_sufficient"] for p in plans_8g)

    # 2. 24 GB High-VRAM GPU (RTX 4090 / A5000): glm-4.7-flash + deepseek-r1:14b + qwen2.5:14b
    tags_24g, _ = select_dynamic_provision_targets(
        hw=_hw(ram=64.0, vram_mb=24_576.0, disk=500.0),
        installed_models=[],
        model_family="auto",
        prefer_vram_resident=True,
    )
    assert tags_24g == ["glm-4.7-flash", "deepseek-r1:14b", "qwen2.5:14b"]

    # 3. Edge laptop (8 GB free RAM, 2 GB VRAM): qwen2.5:3b + deepseek-r1:1.5b
    tags_edge, _ = select_dynamic_provision_targets(
        hw=_hw(ram=8.0, vram_mb=2048.0, disk=60.0),
        installed_models=[],
        model_family="auto",
    )
    assert tags_edge == ["qwen2.5:3b", "deepseek-r1:1.5b"]

    # 4. Ultra-constrained host (4 GB free RAM, no GPU): qwen2.5:1.5b
    tags_min, _ = select_dynamic_provision_targets(
        hw=_hw(ram=4.0, vram_mb=0.0, disk=25.0),
        installed_models=[],
        model_family="auto",
    )
    assert tags_min == ["qwen2.5:1.5b"]

    # 5. Tight disk headroom automatically downsizes oversized explicit targets
    tags_tight, plans_tight = select_dynamic_provision_targets(
        hw=_hw(ram=32.0, vram_mb=8000.0, disk=3.0),
        installed_models=[],
        explicit_models=["glm-4.7-flash", "deepseek-r1:7b"],
    )
    assert "glm-4.7-flash" not in tags_tight
    assert all(p["downsized_for_resources"] for p in plans_tight)


def test_detect_ollama_and_dry_run_install_safety() -> None:
    # Must never throw and never touch PATH registry keys
    binary = detect_ollama_executable()
    assert binary is None or isinstance(binary, str)

    info = ensure_ollama_installed(auto_install_ollama=False)
    assert "installed" in info and "action" in info
    if info["installed"]:
        assert info["action"] == "already_installed"
        assert info["binary"]
    else:
        assert info["action"] == "manual_install_required"
        assert "install_command" in info


def test_live_loopback_ndjson_streaming_pull_and_dynamic_provisioning() -> None:
    """Spins up a real local HTTP server speaking Ollama's /api/tags + streaming NDJSON /api/pull."""
    state: Dict[str, Any] = {
        "installed": ["hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q4_K_M"],
        "pulled_requests": [],
    }

    class OllamaPullHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/api/tags":
                payload = json.dumps({"models": [{"name": m} for m in state["installed"]]}).encode("utf-8")
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
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            req_json = json.loads(raw) if raw else {}
            if self.path == "/api/pull":
                model_name = str(req_json.get("name", ""))
                state["pulled_requests"].append(model_name)
                events: List[Dict[str, Any]] = [
                    {"status": "pulling manifest"},
                    {"status": "pulling sha256:layer1", "digest": "sha256:layer1", "total": 1000, "completed": 400},
                    {"status": "pulling sha256:layer1", "digest": "sha256:layer1", "total": 1000, "completed": 1000},
                    {"status": "pulling sha256:layer2", "digest": "sha256:layer2", "total": 250, "completed": 250},
                    {"status": "verifying sha256 digest"},
                    {"status": "writing manifest"},
                    {"status": "success"},
                ]
                if model_name and model_name not in state["installed"]:
                    state["installed"].append(model_name)
                body = ("\n".join(json.dumps(e) for e in events) + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), OllamaPullHandler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{port}"

    try:
        # 1. Direct streaming pull of glm4:9b verifies multi-layer NDJSON tracking
        single = pull_ollama_model_streaming("glm4:9b", endpoint=endpoint, show_progress=False)
        assert single["status"] == "installed"
        assert single["model"] == "glm4:9b"
        assert single["total_bytes"] == 1000
        assert "glm4:9b" in state["installed"]

        # 2. Dynamic provisioner sees glm4:9b is already installed and pulls deepseek-r1:7b
        prov = dynamically_provision_ollama_and_models(
            endpoint=endpoint,
            model_family="auto",
            auto_install_ollama=False,
            auto_pull=True,
            prefer_vram_resident=True,
            override_free_ram_gb=180.0,
            override_free_vram_mb=7000.0,
            override_free_disk_gb=500.0,
            show_progress=False,
        )
        assert prov["ollama_online"] is True
        assert prov["target_models"] == ["glm4:9b", "deepseek-r1:7b"]
        statuses = {r["model"]: r["status"] for r in prov["provisioning_results"]}
        assert statuses["glm4:9b"] == "already_installed"
        assert statuses["deepseek-r1:7b"] == "installed"
        assert "deepseek-r1:7b" in prov["installed_models_after"]

        # 3. Pipeline guard returns already_satisfied immediately with zero new pulls
        pull_count_before = len(state["pulled_requests"])
        guard = ensure_models_for_pipeline(
            min_eligible_models=2,
            endpoint=endpoint,
            auto_install_ollama=False,
            show_progress=False,
        )
        assert guard["action"] == "already_satisfied"
        assert len(state["pulled_requests"]) == pull_count_before

        # 4. DynaRotModelRouter.provision_models_for_resources integrates cleanly with loopback server
        router = DynaRotModelRouter(endpoint=endpoint)
        r_prov = router.provision_models_for_resources(
            override_free_ram_gb=180.0,
            override_free_vram_mb=7000.0,
            override_free_disk_gb=500.0,
            use_dynamic_panel_targets=True,
            auto_pull=True,
            show_progress=False,
        )
        assert r_prov["ollama_online"] is True
        assert r_prov["target_models"] == ["glm4:9b", "deepseek-r1:7b"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_provision_and_auto_provision_flags() -> None:
    assert normalize_cli_argv(["--provision"]) == ["--install-models"]
    assert normalize_cli_argv(["--install-ollama"]) == ["--install-models"]
    assert normalize_cli_argv(["--pull-models"]) == ["--install-models"]

    parser = build_arg_parser()
    ns = parser.parse_args(["--auto-provision", "--verify-panel", "--panel-models", "glm4:9b,deepseek-r1:7b"])
    assert ns.auto_provision is True
    assert ns.no_auto_provision is False
    assert ns.verify_panel is True
    assert ns.panel_models == "glm4:9b,deepseek-r1:7b"

    ns_off = parser.parse_args(["--no-auto-provision", "--verify-llm"])
    assert ns_off.no_auto_provision is True
    assert ns_off.verify_llm is True
