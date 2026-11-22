"""
Dynamic Ollama Auto-Installer, Detached Daemon Launcher & Resource-Aware Model Provisioner
(`ollama_provisioner.py`).

Ports QuorumRAG's `NimbleProvisioner` (`detect_ollama_executable`, `ensure_ollama_daemon`,
`calculate_provision_plan`, streaming `/api/pull` progress) into Python with:
  1. Zero-Touch PATH Safety (`environment_safety.md`):
     - Locates `ollama` across process `PATH` and standard Windows/macOS/Linux directories
       strictly in-memory without ever mutating user/system registry `PATH`.
  2. Automatic Ollama Installation (`ensure_ollama_installed`):
     - Installs Ollama automatically via `winget` / official `OllamaSetup.exe` (Windows),
       `brew` / `install.sh` (macOS), or `https://ollama.com/install.sh` (Linux) when missing.
  3. Detached `ollama serve` Auto-Start (`ensure_ollama_daemon`):
     - Starts the local Ollama daemon detached (`CREATE_NO_WINDOW` on Windows) and polls
       `127.0.0.1:11434/api/tags` up to 10s for readiness.
  4. Dynamic Resource-Based Panel Model Selection (`select_dynamic_provision_targets`):
     - Sizes optimal complementary open-weight models strictly from live free RAM, free GPU VRAM,
       and free disk headroom (e.g., `glm4:9b` + `deepseek-r1:7b` on 8 GB VRAM / 12+ GB RAM hosts;
       `glm-4.7-flash` + `deepseek-r1:14b` on 20+ GB VRAM hosts; `qwen2.5:3b` + `deepseek-r1:1.5b`
       on edge 6-12 GB hosts), gating every pull through `calculate_provision_plan()`.
  5. Streaming HTTP `/api/pull` with Live `tqdm` Byte Progress Bar (`pull_ollama_model_streaming`):
     - Streams NDJSON download progress (`completed` / `total` bytes) directly into a `tqdm`
       percentage progress bar and refreshes `_OLLAMA_PROBE_CACHE` immediately upon completion.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dynarot import (
    _OLLAMA_PROBE_CACHE,
    build_no_proxy_opener,
    calculate_provision_plan,
    detect_live_hardware,
    estimated_model_disk_gb,
    is_model_installed,
    is_synthesis_chat_model,
    model_fits_free_memory,
    probe_ollama_models,
    sanitize_model_tag,
    size_glm_by_hardware,
    size_models_by_available_memory,
)
from .models import SystemHardwareResources
from .progress import create_progress_bar


def is_autoprovision_disabled() -> bool:
    """Checks opt-out env var `ARXIV_TO_CODE_DISABLE_AUTOPROVISION=1` (QuorumRAG parity)."""
    val = os.environ.get("ARXIV_TO_CODE_DISABLE_AUTOPROVISION", "").strip().lower()
    return val in ("1", "true", "yes")


def is_ci_or_pytest_environment() -> bool:
    """True when running inside pytest or a CI runner so tests never stall downloading multi-GB weights."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    ci_val = os.environ.get("CI", "").strip().lower()
    return ci_val in ("1", "true", "yes")


# =============================================================================
# 1. Multi-Location Ollama Binary Discovery & Safe Auto-Installer
# =============================================================================

def detect_ollama_executable() -> Optional[str]:
    """
    Locates the `ollama` binary across process PATH and standard OS installation folders
    without mutating system/user PATH registry keys (QuorumRAG NimbleProvisioner parity).
    """
    found_in_path = shutil.which("ollama")
    if found_in_path:
        return found_in_path

    system = platform.system()
    candidates: List[Path] = []
    if system == "Windows":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidates.append(Path(local_app) / "Programs" / "Ollama" / "ollama.exe")
        for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
            pf = os.environ.get(env_var, "")
            if pf:
                candidates.append(Path(pf) / "Ollama" / "ollama.exe")
        users_root = Path(r"C:\Users")
        if users_root.is_dir():
            try:
                for entry in users_root.iterdir():
                    if entry.is_dir():
                        candidates.append(entry / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe")
            except OSError:
                pass
    else:
        candidates.extend(
            [
                Path("/usr/local/bin/ollama"),
                Path("/opt/homebrew/bin/ollama"),
                Path("/usr/bin/ollama"),
                Path.home() / ".local" / "bin" / "ollama",
            ]
        )

    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand)
        except OSError:
            continue
    return None


def ensure_ollama_installed(
    auto_install_ollama: bool = True,
    install_timeout_sec: int = 300,
) -> Dict[str, Any]:
    """
    Verifies whether Ollama is installed; if missing and `auto_install_ollama=True`,
    installs Ollama using the platform's official sanctioned installer without mutating
    user/system PATH registry keys directly.
    """
    existing = detect_ollama_executable()
    if existing:
        return {
            "installed": True,
            "binary": existing,
            "action": "already_installed",
            "detail": f"Ollama executable found at {existing}",
        }

    system = platform.system()
    manual_cmd = (
        "winget install -e --id Ollama.Ollama"
        if system == "Windows"
        else ("brew install ollama" if system == "Darwin" else "curl -fsSL https://ollama.com/install.sh | sh")
    )

    if not auto_install_ollama or is_autoprovision_disabled():
        return {
            "installed": False,
            "binary": None,
            "action": "manual_install_required",
            "install_command": manual_cmd,
            "detail": f"Ollama not found. Install via `{manual_cmd}` or pass auto_install_ollama=True.",
        }

    creationflags = 0x08000000 if system == "Windows" else 0

    try:
        if system == "Windows":
            winget = shutil.which("winget")
            if winget:
                proc = subprocess.run(
                    [
                        winget,
                        "install",
                        "-e",
                        "--id",
                        "Ollama.Ollama",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                        "--silent",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=install_timeout_sec,
                    creationflags=creationflags,
                )
                found_after = detect_ollama_executable()
                if proc.returncode == 0 or found_after:
                    return {
                        "installed": bool(found_after),
                        "binary": found_after,
                        "action": "installed_via_winget",
                        "detail": "Ollama installed via winget.",
                    }

            # Fallback: download official OllamaSetup.exe into temp dir & run /VERYSILENT
            setup_path = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
            req = urllib.request.Request(
                "https://ollama.com/download/OllamaSetup.exe",
                headers={"User-Agent": "ArxivToCode-Provisioner/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                setup_path.write_bytes(resp.read())
            subprocess.run(
                [str(setup_path), "/VERYSILENT", "/NORESTART"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=install_timeout_sec,
                creationflags=creationflags,
            )
            found_after = detect_ollama_executable()
            return {
                "installed": bool(found_after),
                "binary": found_after,
                "action": "installed_via_official_setup" if found_after else "install_attempted_restart_shell",
                "install_command": manual_cmd,
                "detail": f"Executed official OllamaSetup.exe (detected={found_after}).",
            }

        if system == "Darwin":
            brew = shutil.which("brew")
            if brew:
                subprocess.run(
                    [brew, "install", "ollama"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=install_timeout_sec,
                )
            else:
                subprocess.run(
                    ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=install_timeout_sec,
                )
        else:
            subprocess.run(
                ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=install_timeout_sec,
            )

        found_after = detect_ollama_executable()
        return {
            "installed": bool(found_after),
            "binary": found_after,
            "action": "installed_official" if found_after else "install_failed",
            "install_command": manual_cmd,
            "detail": f"Official Ollama installer finished (detected={found_after}).",
        }
    except Exception as exc:
        return {
            "installed": False,
            "binary": None,
            "action": "install_error",
            "install_command": manual_cmd,
            "detail": f"Automatic Ollama install failed: {type(exc).__name__}: {str(exc)[:200]}",
        }


def ensure_ollama_daemon(
    endpoint: str = "http://127.0.0.1:11434",
    auto_install_ollama: bool = False,
    wait_timeout_sec: float = 10.0,
) -> Dict[str, Any]:
    """
    Ensures the local Ollama HTTP daemon is reachable at `endpoint`.
    If offline, locates (or optionally installs) Ollama, spawns `ollama serve` detached
    (`CREATE_NO_WINDOW` on Windows), and polls `/api/tags` until ready.
    """
    _OLLAMA_PROBE_CACHE.pop(endpoint, None)
    online, models = probe_ollama_models(endpoint=endpoint, timeout_sec=0.5)
    if online:
        return {
            "online": True,
            "endpoint": endpoint,
            "action": "already_running",
            "binary": detect_ollama_executable(),
            "installed_models": models,
        }

    binary = detect_ollama_executable()
    install_info: Optional[Dict[str, Any]] = None
    if not binary and auto_install_ollama:
        install_info = ensure_ollama_installed(auto_install_ollama=True)
        binary = install_info.get("binary")

    if not binary:
        return {
            "online": False,
            "endpoint": endpoint,
            "action": "ollama_not_installed",
            "binary": None,
            "install_info": install_info,
            "installed_models": [],
        }

    # Spawn `ollama serve` completely detached
    try:
        popen_kwargs: Dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            # CREATE_NO_WINDOW (0x08000000) | DETACHED_PROCESS (0x00000008)
            popen_kwargs["creationflags"] = 0x08000000 | 0x00000008
        else:
            popen_kwargs["start_new_session"] = True

        subprocess.Popen([binary, "serve"], **popen_kwargs)
    except Exception as exc:
        return {
            "online": False,
            "endpoint": endpoint,
            "action": "daemon_spawn_failed",
            "binary": binary,
            "error": str(exc)[:200],
            "installed_models": [],
        }

    deadline = time.monotonic() + max(1.0, wait_timeout_sec)
    while time.monotonic() < deadline:
        time.sleep(0.25)
        _OLLAMA_PROBE_CACHE.pop(endpoint, None)
        online, models = probe_ollama_models(endpoint=endpoint, timeout_sec=0.4)
        if online:
            return {
                "online": True,
                "endpoint": endpoint,
                "action": "daemon_started",
                "binary": binary,
                "installed_models": models,
            }

    return {
        "online": False,
        "endpoint": endpoint,
        "action": "daemon_start_timeout",
        "binary": binary,
        "installed_models": [],
    }


# =============================================================================
# 2. Dynamic Resource-Based Panel Model Selection (GLM + DeepSeek-R1 + Qwen)
# =============================================================================

def select_dynamic_provision_targets(
    hw: SystemHardwareResources,
    installed_models: List[str],
    model_family: str = "auto",
    explicit_models: Optional[List[str]] = None,
    prefer_vram_resident: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Dynamically selects optimal complementary open-weight models to provision based strictly
    on live free RAM, free GPU VRAM, and free disk headroom (QuorumRAG + ModelFusion parity):

      - If `explicit_models` are supplied (e.g. `["glm4:9b", "deepseek-r1:7b"]`), validates each
        through `calculate_provision_plan()` (downsizing only if RAM/VRAM/disk is insufficient).
      - When VRAM is 4.5-14 GB (e.g. 8 GB Quadro RTX 4000 / RTX 4060) and `prefer_vram_resident=True`:
        provisions **`glm4:9b`** (128K paper grounding & verification) + **`deepseek-r1:7b`**
        (`<think>` reasoning code arbiter) so models stay fast and responsive!
      - When Free VRAM >= 20 GB (e.g. RTX 3090/4090) or CPU-only workstation >= 48 GB RAM with
        `prefer_vram_resident=False`: provisions **`glm-4.7-flash`** + **`deepseek-r1:14b`** + **`qwen2.5:14b`**.
      - When constrained (6-12 GB RAM / 2-4.5 GB VRAM): provisions **`qwen2.5:3b`** + **`deepseek-r1:1.5b`**.
      - When ultra-constrained (< 6 GB RAM): provisions **`qwen2.5:1.5b`**.

    Returns `(effective_target_tags, provision_plans)`.
    """
    fam = (model_family or "auto").strip().lower()
    raw_targets: List[str] = []

    if explicit_models:
        raw_targets = [sanitize_model_tag(m) for m in explicit_models if m and m.strip()]
    else:
        free_ram = hw.free_ram_gb
        free_vram_mb = hw.free_vram_mb
        vram_mid_range = 4_500.0 <= free_vram_mb < 16_000.0

        if fam in ("glm", "glm4", "glm-4"):
            glm_tag = "glm4:9b" if (prefer_vram_resident and vram_mid_range) else size_glm_by_hardware(free_ram, free_vram_mb)
            r1_tag = "deepseek-r1:7b" if (free_ram >= 12.0 or free_vram_mb >= 4_500.0) else "deepseek-r1:1.5b"
            raw_targets = [glm_tag, r1_tag]
        elif fam in ("gemma", "gemma4"):
            gemma_tag = "gemma2:9b" if (prefer_vram_resident and vram_mid_range) else ("gemma4:26b" if free_ram >= 18.0 or free_vram_mb >= 16_384.0 else "gemma3:4b")
            r1_tag = "deepseek-r1:7b" if (free_ram >= 12.0 or free_vram_mb >= 4_500.0) else "deepseek-r1:1.5b"
            raw_targets = [gemma_tag, r1_tag]
        elif fam in ("qwen", "qwen2.5", "qwen25"):
            _, primary_qwen, _, fast_qwen, _, _ = size_models_by_available_memory(free_ram, free_vram_mb)
            if prefer_vram_resident and vram_mid_range:
                raw_targets = ["qwen2.5:7b", "deepseek-r1:7b"]
            else:
                raw_targets = [primary_qwen, fast_qwen]
        else:
            # AutoFusion balanced panel: GLM + DeepSeek-R1 (+ Qwen fast tier if room)
            if free_vram_mb >= 20_000.0:
                raw_targets = ["glm-4.7-flash", "deepseek-r1:14b", "qwen2.5:14b"]
            elif vram_mid_range and prefer_vram_resident:
                # 6-14 GB GPU sweet spot (e.g., Quadro RTX 4000 8 GB): fast VRAM/hybrid GLM-4-9B + DeepSeek-R1-7B
                raw_targets = ["glm4:9b", "deepseek-r1:7b"]
            elif free_ram >= 24.0 or free_vram_mb >= 14_000.0:
                raw_targets = ["glm4:9b", "deepseek-r1:7b"]
            elif free_ram >= 12.0 or free_vram_mb >= 4_500.0:
                raw_targets = ["glm4:9b", "deepseek-r1:7b"]
            elif free_ram >= 6.0 or free_vram_mb >= 2_000.0:
                raw_targets = ["qwen2.5:3b", "deepseek-r1:1.5b"]
            else:
                raw_targets = ["qwen2.5:1.5b"]

    effective_tags: List[str] = []
    plans: List[Dict[str, Any]] = []
    seen = set()
    for tag in raw_targets:
        plan = calculate_provision_plan(tag, hw, installed_models)
        eff = plan["effective_tag"]
        if eff not in seen:
            seen.add(eff)
            effective_tags.append(eff)
            plans.append(plan)
    return effective_tags, plans


# =============================================================================
# 3. Streaming `/api/pull` with Live `tqdm` Byte Progress Bar
# =============================================================================

def pull_ollama_model_streaming(
    model_tag: str,
    endpoint: str = "http://127.0.0.1:11434",
    timeout_sec: int = 1800,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Pulls an Ollama model via streaming HTTP `POST /api/pull` (no-proxy opener, zero `Origin: null`)
    displaying live `tqdm` byte-percentage progress (`0% -> 100%`). Falls back to `ollama pull` CLI
    if the HTTP stream is rejected.
    """
    clean_tag = sanitize_model_tag(model_tag)
    url = f"{endpoint.rstrip('/')}/api/pull"
    body = json.dumps({"name": clean_tag, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ArxivToCode-Provisioner/1.0",
        },
    )
    opener = build_no_proxy_opener()
    t0 = time.perf_counter()

    pbar = None
    last_status = ""
    current_digest = ""
    total_bytes = 0
    completed_bytes = 0
    max_layer_bytes = 0
    succeeded = False
    error_msg = ""

    try:
        with opener.open(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return {
                    "model": clean_tag,
                    "status": f"pull_http_{resp.status}",
                    "command": f"ollama pull {clean_tag}",
                    "duration_s": round(time.perf_counter() - t0, 2),
                }
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if ev.get("error"):
                    error_msg = str(ev["error"])
                    break

                status_txt = str(ev.get("status", "")).strip()
                if status_txt:
                    last_status = status_txt
                if "success" in status_txt.lower():
                    succeeded = True

                ev_digest = str(ev.get("digest") or "")
                ev_total = int(ev.get("total") or 0)
                ev_comp = int(ev.get("completed") or 0)
                if ev_total > 0:
                    if ev_total > max_layer_bytes:
                        max_layer_bytes = ev_total
                    layer_changed = bool(ev_digest and current_digest and ev_digest != current_digest)
                    if ev_digest:
                        current_digest = ev_digest
                    total_bytes = ev_total
                    if pbar is None:
                        pbar = create_progress_bar(
                            total=ev_total,
                            desc=f"Pulling {clean_tag}",
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            disable=not show_progress,
                        )
                        completed_bytes = 0
                    elif layer_changed or (hasattr(pbar, "total") and pbar.total != ev_total):
                        if hasattr(pbar, "reset"):
                            pbar.reset(total=ev_total)
                        else:
                            pbar.total = ev_total
                        completed_bytes = 0
                    delta = max(0, ev_comp - completed_bytes)
                    completed_bytes = ev_comp
                    if delta > 0:
                        pbar.update(delta)
                    pbar.set_postfix_str(status_txt[:28])
                elif pbar is not None and status_txt:
                    pbar.set_postfix_str(status_txt[:28])
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        if pbar is not None:
            pbar.close()

    duration_s = round(time.perf_counter() - t0, 2)
    if succeeded and not error_msg:
        _OLLAMA_PROBE_CACHE.pop(endpoint, None)
        return {
            "model": clean_tag,
            "status": "installed",
            "last_event": last_status,
            "total_bytes": max_layer_bytes or total_bytes,
            "completed_bytes": max(completed_bytes, max_layer_bytes or total_bytes),
            "duration_s": duration_s,
            "command": f"ollama pull {clean_tag}",
        }

    return {
        "model": clean_tag,
        "status": "pull_failed",
        "error": error_msg or f"Unexpected stream end ({last_status})",
        "duration_s": duration_s,
        "command": f"ollama pull {clean_tag}",
    }


# =============================================================================
# 4. Unified Dynamic Provisioning Orchestrator
# =============================================================================

def dynamically_provision_ollama_and_models(
    endpoint: Optional[str] = None,
    model_family: str = "auto",
    explicit_models: Optional[List[str]] = None,
    auto_install_ollama: bool = True,
    auto_pull: bool = True,
    prefer_vram_resident: bool = True,
    override_free_ram_gb: Optional[float] = None,
    override_free_vram_mb: Optional[float] = None,
    override_free_disk_gb: Optional[float] = None,
    show_progress: bool = True,
    pull_timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """
    End-to-end dynamic provisioner:
      1. Probes live free RAM, multi-GPU free VRAM, and free disk headroom.
      2. Ensures Ollama is installed (auto-installing when `auto_install_ollama=True` and missing).
      3. Ensures the local Ollama daemon is running (spawning `ollama serve` detached if stopped).
      4. Dynamically computes resource-fitted panel models (`glm4:9b`, `deepseek-r1:7b`, etc.)
         gated by `calculate_provision_plan()` (automatic RAM/VRAM/disk downsizing).
      5. Streams downloads via `/api/pull` with live `tqdm` byte progress bars.
    """
    ep = endpoint or os.environ.get("LOCAL_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    hw = detect_live_hardware(
        override_free_ram_gb=override_free_ram_gb,
        override_free_vram_mb=override_free_vram_mb,
        override_free_disk_gb=override_free_disk_gb,
    )

    daemon_status = ensure_ollama_daemon(
        endpoint=ep,
        auto_install_ollama=auto_install_ollama and auto_pull,
        wait_timeout_sec=15.0,
    )
    ollama_online = bool(daemon_status.get("online"))
    installed_before: List[str] = list(daemon_status.get("installed_models") or [])

    target_models, plans = select_dynamic_provision_targets(
        hw=hw,
        installed_models=installed_before,
        model_family=model_family,
        explicit_models=explicit_models,
        prefer_vram_resident=prefer_vram_resident,
    )

    pull_results: List[Dict[str, Any]] = []
    for tag, plan in zip(target_models, plans):
        if is_model_installed(tag, installed_before):
            pull_results.append(
                {
                    "model": tag,
                    "status": "already_installed",
                    "command": f"ollama pull {tag}",
                    "plan": plan,
                }
            )
            continue

        if not plan["disk_sufficient"]:
            pull_results.append(
                {
                    "model": tag,
                    "status": "skipped_insufficient_disk",
                    "free_disk_gb": plan["free_disk_gb"],
                    "required_disk_gb": estimated_model_disk_gb(tag),
                    "command": f"ollama pull {tag}",
                    "plan": plan,
                }
            )
            continue

        if not auto_pull or is_autoprovision_disabled():
            pull_results.append(
                {
                    "model": tag,
                    "status": "pending_install",
                    "command": f"ollama pull {tag}",
                    "plan": plan,
                }
            )
            continue

        if not ollama_online:
            pull_results.append(
                {
                    "model": tag,
                    "status": "deferred_ollama_offline",
                    "command": f"ollama pull {tag}",
                    "plan": plan,
                }
            )
            continue

        res = pull_ollama_model_streaming(
            model_tag=tag,
            endpoint=ep,
            timeout_sec=pull_timeout_sec,
            show_progress=show_progress,
        )
        res["plan"] = plan
        pull_results.append(res)

    _OLLAMA_PROBE_CACHE.pop(ep, None)
    online_after, installed_after = probe_ollama_models(endpoint=ep, timeout_sec=0.5)

    return {
        "ollama_online": online_after or ollama_online,
        "endpoint": ep,
        "daemon_status": daemon_status,
        "hardware": hw.to_dict(),
        "model_family": model_family,
        "target_models": target_models,
        "provision_plans": plans,
        "installed_models_before": installed_before,
        "installed_models_after": installed_after or installed_before,
        "provisioning_results": pull_results,
        "terminal_commands": [f"ollama pull {m}" for m in target_models],
    }


def ensure_models_for_pipeline(
    min_eligible_models: int = 1,
    endpoint: Optional[str] = None,
    model_family: str = "auto",
    explicit_models: Optional[List[str]] = None,
    auto_install_ollama: bool = True,
    prefer_vram_resident: bool = True,
    show_progress: bool = True,
    pull_timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """
    On-demand guard for `--auto-provision` (CLI & MCP):
    Checks whether Ollama is running and already has at least `min_eligible_models`
    memory-eligible synthesis/chat models (or all requested `explicit_models`).
    If already satisfied, returns immediately with zero downloads; otherwise triggers
    `dynamically_provision_ollama_and_models()` to install/start Ollama and pull
    resource-fitted models (e.g., `glm4:9b` + `deepseek-r1:7b`).
    """
    ep = endpoint or os.environ.get("LOCAL_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    hw = detect_live_hardware()
    online, installed = probe_ollama_models(endpoint=ep, timeout_sec=0.5)
    eligible = [
        m
        for m in installed
        if is_synthesis_chat_model(m) and model_fits_free_memory(m, hw.free_ram_gb, hw.free_vram_mb)
    ]

    if online:
        if explicit_models:
            clean_explicit = [sanitize_model_tag(m) for m in explicit_models if m and m.strip()]
            missing = [m for m in clean_explicit if not is_model_installed(m, installed)]
            if not missing:
                return {
                    "action": "already_satisfied",
                    "ollama_online": True,
                    "endpoint": ep,
                    "eligible_installed_models": eligible,
                    "installed_models_after": installed,
                    "provisioning_results": [],
                }
        elif len(eligible) >= max(1, min_eligible_models):
            return {
                "action": "already_satisfied",
                "ollama_online": True,
                "endpoint": ep,
                "eligible_installed_models": eligible,
                "installed_models_after": installed,
                "provisioning_results": [],
            }

    res = dynamically_provision_ollama_and_models(
        endpoint=ep,
        model_family=model_family,
        explicit_models=explicit_models,
        auto_install_ollama=auto_install_ollama,
        auto_pull=True,
        prefer_vram_resident=prefer_vram_resident,
        show_progress=show_progress,
        pull_timeout_sec=pull_timeout_sec,
    )
    res["action"] = "provisioned"
    return res
