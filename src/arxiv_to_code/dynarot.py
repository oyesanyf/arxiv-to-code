"""
ModelFusion DynaRot (Dynamic Router & Hardware-Aware Model Provisioner) in Python.

Implements:
1. Dynamic Available Memory Law (Anti-OOM):
   - Evaluates runtime free/available RAM (free_ram_gb) and free GPU VRAM (free_vram_mb),
     NEVER total physical RAM:
     * Free RAM >= 48 GB OR Free VRAM >= 24 GB (24,000 MB) -> qwen2.5:32b
     * Free RAM >= 24 GB OR Free VRAM >= 14 GB (14,000 MB) -> qwen2.5:14b
     * Free RAM >= 12 GB OR Free VRAM >= 4.5 GB (4,500 MB) -> qwen2.5:7b
     * Free RAM >= 6 GB  OR Free VRAM >= 2 GB   (2,000 MB) -> qwen2.5:3b
     * Free RAM <  6 GB                                    -> qwen2.5:1.5b
   - Never appends '-coder' suffixes that trigger silent 404 aborts.
2. Evolutionary Compute Budget Split:
   - Allocates heavier reasoning models to ShinkaEvolve for structural discovery and UCB routing.
   - Allocates fast, lightweight models to OpenEvolve for high-volume MAP-Elites micro-mutations.
3. Multi-Objective Pareto Front & Geometric Knee-Point Selection + UCB1 Exploration Bonus.
4. Fast 400ms Ollama Probe & Resource-Based Model Auto-Installation (`ollama pull <model>`).
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .models import DynaRotRoutingDecision, SystemHardwareResources

SAFETY_FACTOR = 0.70
EPS = 1e-9

HIGH_SIGNAL_RESEARCH_ORGS = (
    "recursivesi",
    "qwen",
    "deepseek-ai",
    "zai-org",
    "zhipu",
    "google",
    "meta-llama",
    "mistralai",
    "microsoft",
    "allenai",
    "nvidia",
)

GEMMA4_26B_MIN_FREE_RAM_GB = 18.0
GEMMA4_26B_MIN_FREE_VRAM_MB = 16_384.0
NIMBLE_MIN_FREE_RAM_GB = 12.0
NIMBLE_MIN_FREE_VRAM_MB = 6_000.0

# Curated open-weight catalog with parameter counts, capability scores, and popularity metrics
# Includes Z.ai GLM (`glm-4.7-flash`, `glm4:9b`), Qwen 2.5, DeepSeek-R1, and Gemma families
CURATED_MODEL_CATALOG: List[Dict[str, Any]] = [
    {
        "model_id": "qwen2.5:32b",
        "params_b": 32.0,
        "capability_score": 9.6,
        "decision_score": 9.5,
        "efficiency_score": 8.0,
        "downloads": 4_800_000,
        "trending_score": 920.0,
        "author": "Qwen",
        "role": "heavy_reasoning",
    },
    {
        "model_id": "deepseek-r1:32b",
        "params_b": 32.0,
        "capability_score": 9.7,
        "decision_score": 9.6,
        "efficiency_score": 7.9,
        "downloads": 5_200_000,
        "trending_score": 960.0,
        "author": "deepseek-ai",
        "role": "heavy_reasoning",
    },
    {
        "model_id": "glm-4.7-flash",
        "params_b": 30.0,
        "capability_score": 9.5,
        "decision_score": 9.4,
        "efficiency_score": 9.1,
        "downloads": 4_500_000,
        "trending_score": 955.0,
        "author": "zai-org",
        "role": "heavy_reasoning",
    },
    {
        "model_id": "gemma4:26b",
        "params_b": 26.0,
        "capability_score": 9.4,
        "decision_score": 9.3,
        "efficiency_score": 8.6,
        "downloads": 4_200_000,
        "trending_score": 945.0,
        "author": "google",
        "role": "heavy_reasoning",
    },
    {
        "model_id": "qwen2.5:14b",
        "params_b": 14.0,
        "capability_score": 9.1,
        "decision_score": 9.0,
        "efficiency_score": 8.6,
        "downloads": 6_100_000,
        "trending_score": 890.0,
        "author": "Qwen",
        "role": "heavy_reasoning",
    },
    {
        "model_id": "deepseek-r1:14b",
        "params_b": 14.0,
        "capability_score": 9.2,
        "decision_score": 9.1,
        "efficiency_score": 8.5,
        "downloads": 5_900_000,
        "trending_score": 930.0,
        "author": "deepseek-ai",
        "role": "heavy_reasoning",
    },
    {
        "model_id": "glm4:9b",
        "params_b": 9.4,
        "capability_score": 8.9,
        "decision_score": 8.8,
        "efficiency_score": 9.1,
        "downloads": 7_800_000,
        "trending_score": 915.0,
        "author": "zai-org",
        "role": "balanced",
    },
    {
        "model_id": "gemma2:9b",
        "params_b": 9.0,
        "capability_score": 8.7,
        "decision_score": 8.5,
        "efficiency_score": 8.9,
        "downloads": 8_100_000,
        "trending_score": 880.0,
        "author": "google",
        "role": "balanced",
    },
    {
        "model_id": "qwen2.5:7b",
        "params_b": 7.0,
        "capability_score": 8.6,
        "decision_score": 8.5,
        "efficiency_score": 9.2,
        "downloads": 12_500_000,
        "trending_score": 950.0,
        "author": "Qwen",
        "role": "balanced",
    },
    {
        "model_id": "deepseek-r1:7b",
        "params_b": 7.0,
        "capability_score": 8.7,
        "decision_score": 8.6,
        "efficiency_score": 9.0,
        "downloads": 9_400_000,
        "trending_score": 940.0,
        "author": "deepseek-ai",
        "role": "heavy_reasoning",
    },
    {
        "model_id": "gemma3:4b",
        "params_b": 4.0,
        "capability_score": 8.1,
        "decision_score": 8.0,
        "efficiency_score": 9.4,
        "downloads": 6_900_000,
        "trending_score": 895.0,
        "author": "google",
        "role": "fast_mutation",
    },
    {
        "model_id": "qwen2.5:3b",
        "params_b": 3.0,
        "capability_score": 7.9,
        "decision_score": 7.8,
        "efficiency_score": 9.5,
        "downloads": 7_200_000,
        "trending_score": 870.0,
        "author": "Qwen",
        "role": "fast_mutation",
    },
    {
        "model_id": "gemma2:2b",
        "params_b": 2.0,
        "capability_score": 7.6,
        "decision_score": 7.5,
        "efficiency_score": 9.65,
        "downloads": 8_600_000,
        "trending_score": 875.0,
        "author": "google",
        "role": "fast_mutation",
    },
    {
        "model_id": "deepseek-r1:1.5b",
        "params_b": 1.5,
        "capability_score": 7.6,
        "decision_score": 7.7,
        "efficiency_score": 9.7,
        "downloads": 8_300_000,
        "trending_score": 910.0,
        "author": "deepseek-ai",
        "role": "fast_mutation",
    },
    {
        "model_id": "qwen2.5:1.5b",
        "params_b": 1.5,
        "capability_score": 7.4,
        "decision_score": 7.4,
        "efficiency_score": 9.8,
        "downloads": 9_100_000,
        "trending_score": 860.0,
        "author": "Qwen",
        "role": "fast_mutation",
    },
    {
        "model_id": "qwen2.5:0.5b",
        "params_b": 0.5,
        "capability_score": 6.8,
        "decision_score": 6.9,
        "efficiency_score": 9.95,
        "downloads": 6_400_000,
        "trending_score": 820.0,
        "author": "Qwen",
        "role": "fast_router",
    },
]


def sanitize_model_tag(model_tag: str) -> str:
    """
    Enforces the Anti-OOM / Anti-404 Model Tag Invariant:
    Never append '-coder' suffixes that trigger silent 404 aborts.
    """
    cleaned = model_tag.strip()
    if "-coder:" in cleaned:
        cleaned = cleaned.replace("-coder:", ":")
    elif cleaned.endswith("-coder"):
        cleaned = cleaned[: -len("-coder")]
    return cleaned


def _detect_windows_ram_gb() -> Tuple[float, float]:
    """Detects live (total_ram_gb, free_ram_gb) on Windows via GlobalMemoryStatusEx."""
    if platform.system() == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                total_gb = stat.ullTotalPhys / (1024.0 ** 3)
                free_gb = stat.ullAvailPhys / (1024.0 ** 3)
                return round(total_gb, 2), round(free_gb, 2)
        except Exception:
            pass

    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return round(vm.total / (1024.0 ** 3), 2), round(vm.available / (1024.0 ** 3), 2)
    except Exception:
        return 16.0, 12.0


def build_no_proxy_opener() -> urllib.request.OpenerDirector:
    """
    Returns an urllib opener that explicitly bypasses OS/Windows proxy auto-discovery
    (`ProxyHandler({})`) so local loopback requests to `127.0.0.1:11434` never stall
    on WPAD/WinReg lookup or route through a corporate proxy.
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def probe_free_disk_gb() -> float:
    """
    Measures available free disk space (GB) on the filesystem hosting Ollama models
    (`OLLAMA_MODELS`, `USERPROFILE`, or `HOME`), matching QuorumRAG's `HardwareTelemetry::probe_free_disk_gb`.
    """
    target = (
        os.environ.get("OLLAMA_MODELS")
        or os.environ.get("USERPROFILE")
        or os.environ.get("HOME")
        or os.getcwd()
    )
    try:
        usage = shutil.disk_usage(target)
        return round(max(0.0, usage.free / (1024.0 ** 3)), 2)
    except Exception:
        return 100.0


def parse_nvidia_smi_output(stdout: str) -> Tuple[Optional[str], float, float, int]:
    """
    Parses multi-line `nvidia-smi` CSV output (3-column or 2-column) summing VRAM across
    multi-GPU setups and formatting compound GPU names, matching QuorumRAG's `HardwareTelemetry`.
    Returns `(gpu_name, total_vram_gb, free_vram_gb, gpu_count)`.
    """
    names: List[str] = []
    total_mb = 0.0
    free_mb = 0.0
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            try:
                names.append(parts[0])
                total_mb += float(parts[1].split()[0])
                free_mb += float(parts[2].split()[0])
            except ValueError:
                continue
        elif len(parts) == 2 and parts[0]:
            try:
                names.append(parts[0])
                free_mb += float(parts[1].split()[0])
            except ValueError:
                continue
    if not names:
        return None, 0.0, 0.0, 0
    if len(names) == 1:
        formatted = names[0]
    elif all(n == names[0] for n in names):
        formatted = f"{len(names)}x {names[0]}"
    else:
        formatted = " + ".join(names)
    return formatted, round(total_mb / 1024.0, 2), round(free_mb / 1024.0, 2), len(names)


def _detect_gpu_vram() -> Tuple[Optional[str], float, float, int]:
    """
    Detects GPU name, total VRAM (GB), free VRAM (GB), and GPU count via nvidia-smi with a hard
    1.0s non-blocking daemon timeout so Windows pipe handle stalls can never hang startup.
    """
    candidates = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    binary = next((b for b in candidates if (os.path.exists(b) if "\\" in b else shutil.which(b))), None)
    if not binary:
        return None, 0.0, 0.0, 0

    result_box: List[Tuple[Optional[str], float, float, int]] = []
    proc_holder: List[Any] = []

    def _worker() -> None:
        try:
            creationflags = 0x08000000 if platform.system() == "Windows" else 0
            proc = subprocess.Popen(
                [
                    binary,
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
            proc_holder.append(proc)
            stdout, _ = proc.communicate(timeout=0.9)
            if proc.returncode == 0 and stdout and stdout.strip():
                parsed = parse_nvidia_smi_output(stdout)
                if parsed[3] > 0:
                    result_box.append(parsed)
        except Exception:
            if proc_holder:
                try:
                    proc_holder[0].kill()
                except Exception:
                    pass

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=1.0)
    if thread.is_alive() and proc_holder:
        try:
            proc_holder[0].kill()
        except Exception:
            pass
    return result_box[0] if result_box else (None, 0.0, 0.0, 0)


_HARDWARE_CACHE: Optional[Tuple[float, SystemHardwareResources]] = None
_OLLAMA_PROBE_CACHE: Dict[str, Tuple[float, bool, List[str]]] = {}


def detect_live_hardware(
    override_free_ram_gb: Optional[float] = None,
    override_free_vram_mb: Optional[float] = None,
    override_free_disk_gb: Optional[float] = None,
) -> SystemHardwareResources:
    """
    Probes live available system RAM, GPU VRAM, and free disk headroom at runtime.
    Caches hardware detection for 30 seconds within the process (matching ModelFusion & QuorumRAG).
    """
    global _HARDWARE_CACHE
    now = time.monotonic()
    if (
        override_free_ram_gb is None
        and override_free_vram_mb is None
        and override_free_disk_gb is None
        and _HARDWARE_CACHE is not None
        and (now - _HARDWARE_CACHE[0]) < 30.0
    ):
        return _HARDWARE_CACHE[1]

    if _HARDWARE_CACHE is not None and (now - _HARDWARE_CACHE[0]) < 30.0:
        base_hw = _HARDWARE_CACHE[1]
        total_ram_gb, free_ram_gb = base_hw.total_ram_gb, base_hw.free_ram_gb
        gpu_name, gpu_vram_total_gb, gpu_vram_free_gb, gpu_count = (
            base_hw.gpu_name,
            base_hw.gpu_vram_total_gb,
            base_hw.gpu_vram_free_gb,
            base_hw.gpu_count,
        )
        cpu_cores = base_hw.cpu_cores
        free_disk_gb = base_hw.free_disk_gb
    else:
        total_ram_gb, free_ram_gb = _detect_windows_ram_gb()
        gpu_name, gpu_vram_total_gb, gpu_vram_free_gb, gpu_count = _detect_gpu_vram()
        cpu_cores = os.cpu_count() or 4
        free_disk_gb = probe_free_disk_gb()
        _HARDWARE_CACHE = (
            now,
            SystemHardwareResources(
                total_ram_gb=total_ram_gb,
                free_ram_gb=free_ram_gb,
                cpu_cores=cpu_cores,
                gpu_name=gpu_name,
                gpu_vram_total_gb=gpu_vram_total_gb,
                gpu_vram_free_gb=gpu_vram_free_gb,
                free_disk_gb=free_disk_gb,
                gpu_count=gpu_count,
            ),
        )

    if override_free_ram_gb is not None:
        free_ram_gb = float(override_free_ram_gb)
        total_ram_gb = max(total_ram_gb, free_ram_gb)
    if override_free_vram_mb is not None:
        gpu_vram_free_gb = round(float(override_free_vram_mb) / 1024.0, 2)
        gpu_vram_total_gb = max(gpu_vram_total_gb, gpu_vram_free_gb)
        if gpu_vram_free_gb > 0 and gpu_name is None:
            gpu_name = "Simulated/Override GPU"
            gpu_count = max(1, gpu_count)
    if override_free_disk_gb is not None:
        free_disk_gb = float(override_free_disk_gb)

    return SystemHardwareResources(
        total_ram_gb=total_ram_gb,
        free_ram_gb=free_ram_gb,
        cpu_cores=cpu_cores,
        gpu_name=gpu_name,
        gpu_vram_total_gb=gpu_vram_total_gb,
        gpu_vram_free_gb=gpu_vram_free_gb,
        free_disk_gb=free_disk_gb,
        gpu_count=gpu_count,
    )


def estimate_runtime_memory_gb(params_b: float, backend: str = "ollama") -> float:
    """
    Estimates runtime memory in GB:
    - Ollama (Q4_0): params_b * 0.6 * 1.2 = params_b * 0.72 GB
    - OpenVINO (INT4): params_b * 0.5 * 1.2 = params_b * 0.60 GB
    - Transformers (FP16): params_b * 2.0 * 1.2 = params_b * 2.40 GB
    """
    b_low = backend.lower()
    if b_low == "ollama":
        bytes_per_param = 0.6
    elif b_low == "openvino":
        bytes_per_param = 0.5
    else:
        bytes_per_param = 2.0
    return round(params_b * bytes_per_param * 1.2, 2)


def size_models_by_available_memory(
    free_ram_gb: float,
    free_vram_mb: float,
    requires_reasoning: bool = True,
) -> Tuple[str, str, str, str, int, List[str]]:
    """
    Strictly enforces the Dynamic Available Memory Law:
      - Free RAM >= 48 GB OR Free VRAM >= 24 GB (24,000 MB) -> qwen2.5:32b
      - Free RAM >= 24 GB OR Free VRAM >= 14 GB (14,000 MB) -> qwen2.5:14b
      - Free RAM >= 12 GB OR Free VRAM >= 4.5 GB (4,500 MB) -> qwen2.5:7b
      - Free RAM >= 6 GB  OR Free VRAM >= 2 GB   (2,000 MB) -> qwen2.5:3b
      - Free RAM <  6 GB                                    -> qwen2.5:1.5b

    Also splits compute budget for --evolve (ShinkaEvolve vs OpenEvolve):
      - shinka_heavy_reasoning_model: Heavy reasoning tier for ShinkaEvolve structural synthesis
      - openevolve_fast_mutation_model: Fast lightweight tier for OpenEvolve high-volume MAP-Elites mutations

    Returns:
      (hardware_tier, primary_model, shinka_heavy_model, openevolve_fast_model, max_ensemble, candidate_pool)
    """
    if free_ram_gb >= 48.0 or free_vram_mb >= 24_000.0:
        tier = "Tier 1: Frontier 32B Reasoning (Free RAM >= 48 GB or Free VRAM >= 24 GB)"
        primary = "qwen2.5:32b"
        shinka_heavy = "qwen2.5:32b"
        openevolve_fast = "qwen2.5:7b"
        max_ensemble = 7
        pool = [
            "qwen2.5:32b",
            "glm-4.7-flash",
            "deepseek-r1:32b",
            "gemma4:26b",
            "qwen2.5:14b",
            "glm4:9b",
            "qwen2.5:7b",
        ]
    elif free_ram_gb >= 24.0 or free_vram_mb >= 14_000.0:
        tier = "Tier 2: High-Throughput 14B (Free RAM >= 24 GB or Free VRAM >= 14 GB)"
        primary = "qwen2.5:14b"
        shinka_heavy = "qwen2.5:14b"
        openevolve_fast = "qwen2.5:3b"
        max_ensemble = 5
        pool = [
            "qwen2.5:14b",
            "glm-4.7-flash",
            "deepseek-r1:14b",
            "glm4:9b",
            "qwen2.5:7b",
        ]
    elif free_ram_gb >= 12.0 or free_vram_mb >= 4_500.0:
        tier = "Tier 3: GPU-Resident 7B/9B (Free RAM >= 12 GB or Free VRAM >= 4.5 GB)"
        primary = "qwen2.5:7b"
        shinka_heavy = "qwen2.5:7b"
        openevolve_fast = "qwen2.5:1.5b"
        max_ensemble = 4
        pool = [
            "qwen2.5:7b",
            "glm4:9b",
            "deepseek-r1:7b",
            "qwen2.5:3b",
            "qwen2.5:1.5b",
        ]
    elif free_ram_gb >= 6.0 or free_vram_mb >= 2_000.0:
        tier = "Tier 4: Low-Power Edge 3B (Free RAM >= 6 GB or Free VRAM >= 2 GB)"
        primary = "qwen2.5:3b"
        shinka_heavy = "qwen2.5:3b"
        openevolve_fast = "qwen2.5:1.5b"
        max_ensemble = 3
        pool = [
            "qwen2.5:3b",
            "gemma3:4b",
            "deepseek-r1:1.5b",
            "qwen2.5:1.5b",
        ]
    else:
        tier = "Tier 5: Compact Minimal Footprint 1.5B (Free RAM < 6 GB)"
        primary = "qwen2.5:1.5b"
        shinka_heavy = "qwen2.5:1.5b"
        openevolve_fast = "qwen2.5:0.5b"
        max_ensemble = 2
        pool = [
            "qwen2.5:1.5b",
            "gemma2:2b",
            "qwen2.5:0.5b",
        ]

    # Guarantee zero '-coder' suffixes across all returned model tags
    primary = sanitize_model_tag(primary)
    shinka_heavy = sanitize_model_tag(shinka_heavy)
    openevolve_fast = sanitize_model_tag(openevolve_fast)
    pool = [sanitize_model_tag(m) for m in pool]

    return tier, primary, shinka_heavy, openevolve_fast, max_ensemble, pool


# =============================================================================
# QuorumRAG AutoFusionRouter Parity: Resource-Calibrated GLM / Qwen / Gemma Selection
# =============================================================================

def can_run_gemma4_26b(free_ram_gb: float, free_vram_mb: float) -> bool:
    """Evaluates strictly runtime free memory (never total RAM) for 26B / GLM-4.7-Flash tier."""
    return free_ram_gb >= GEMMA4_26B_MIN_FREE_RAM_GB or free_vram_mb >= GEMMA4_26B_MIN_FREE_VRAM_MB


def size_glm_by_hardware(free_ram_gb: float, free_vram_mb: float) -> str:
    """
    Sizes the Z.ai GLM family strictly against live runtime available RAM & VRAM:
      - Free RAM >= 18 GB OR Free VRAM >= 16,384 MB -> glm-4.7-flash (30B MoE / 3B active, tools+thinking)
      - Free RAM >= 12 GB OR Free VRAM >= 4,500 MB  -> glm4:9b (9.4B, 128K context)
      - Free RAM >= 6 GB  OR Free VRAM >= 2,000 MB  -> qwen2.5:3b (memory-safe edge fallback)
      - Below 6 GB                                  -> qwen2.5:1.5b
    """
    if can_run_gemma4_26b(free_ram_gb, free_vram_mb):
        return "glm-4.7-flash"
    if free_ram_gb >= 12.0 or free_vram_mb >= 4_500.0:
        return "glm4:9b"
    if free_ram_gb >= 6.0 or free_vram_mb >= 2_000.0:
        return "qwen2.5:3b"
    return "qwen2.5:1.5b"


def size_gemma_by_hardware(free_ram_gb: float, free_vram_mb: float) -> str:
    """Matches QuorumRAG AutoFusionRouter::size_gemma_by_hardware strictly on runtime free memory."""
    if can_run_gemma4_26b(free_ram_gb, free_vram_mb):
        return "gemma4:26b"
    if free_ram_gb >= 12.0 or free_vram_mb >= 6_000.0:
        return "gemma2:9b"
    if free_ram_gb >= 6.0 or free_vram_mb >= 3_000.0:
        return "gemma3:4b"
    return "gemma2:2b"


def is_embedding_model(tag: str) -> bool:
    low = tag.strip().lower()
    return (
        "embed" in low
        or low.startswith("bge-")
        or "nomic-embed" in low
        or "minilm" in low
    )


def is_vision_model(tag: str) -> bool:
    low = tag.strip().lower()
    return (
        low.startswith("moondream")
        or low.startswith("llava")
        or low.startswith("minicpm-v")
        or low.startswith("qwen2.5-vl")
        or low.startswith("glm-ocr")
        or "-vl" in low
    )


def is_decision_only_model(tag: str) -> bool:
    """Identifies decision/logprob-only GGUF models (e.g. Bespoke-Nimble-9B) that do not support /api/generate."""
    low = tag.strip().lower()
    return "bespoke-nimble" in low or low.startswith("nimble")


def is_synthesis_chat_model(tag: str) -> bool:
    """True when an Ollama model tag supports general text/JSON generation (not embedding/vision/decision-only)."""
    return (
        not is_embedding_model(tag)
        and not is_vision_model(tag)
        and not is_decision_only_model(tag)
    )


def model_capability_score(tag: str) -> int:
    """Capability ranking matching QuorumRAG AutoFusionRouter::model_capability_score."""
    low = tag.strip().lower()
    if "70b" in low:
        return 700
    if "32b" in low:
        return 320
    if "26b" in low or "27b" in low or low.startswith("glm-4.7") or low.startswith("glm-5"):
        return 260
    if "14b" in low or "13b" in low or "12b" in low:
        return 140
    if "9b" in low or "8b" in low or "7b" in low or low.startswith("glm4"):
        return 80
    if "4b" in low or "3b" in low:
        return 35
    if "1.5b" in low or "2b" in low:
        return 20
    if "0.5b" in low:
        return 10
    return 15


def model_fits_free_memory(tag: str, free_ram_gb: float, free_vram_mb: float) -> bool:
    """Evaluates strictly runtime free RAM & free VRAM against model parameter size (QuorumRAG parity)."""
    low = tag.strip().lower()
    if "32b" in low or "70b" in low:
        return free_ram_gb >= 48.0 or free_vram_mb >= 24_000.0
    if "26b" in low or "27b" in low or low.startswith("glm-4.7") or low.startswith("glm-5"):
        return can_run_gemma4_26b(free_ram_gb, free_vram_mb)
    if "14b" in low or "12b" in low or "13b" in low:
        return free_ram_gb >= 24.0 or free_vram_mb >= 14_000.0
    if "nimble" in low:
        return free_ram_gb >= NIMBLE_MIN_FREE_RAM_GB or free_vram_mb >= NIMBLE_MIN_FREE_VRAM_MB
    if "9b" in low or "8b" in low or "7b" in low or low.startswith("glm4"):
        return free_ram_gb >= 12.0 or free_vram_mb >= 4_500.0
    if "4b" in low or "3b" in low:
        return free_ram_gb >= 6.0 or free_vram_mb >= 2_000.0
    return True


def estimated_model_disk_gb(tag: str) -> float:
    """Estimates minimum free disk headroom (GB) required before pulling a model tier (QuorumRAG parity)."""
    low = tag.strip().lower()
    if "70b" in low:
        return 42.0
    if "32b" in low or "26b" in low or "27b" in low or low.startswith("glm-4.7") or low.startswith("glm-5"):
        return 18.0
    if "14b" in low or "13b" in low or "12b" in low:
        return 9.5
    if "nimble" in low or "9b" in low or "8b" in low or "7b" in low or low.startswith("glm4"):
        return 6.0
    if "4b" in low or "3b" in low:
        return 2.5
    return 1.2


def find_best_installed_chat_model(
    preferred_tag: Optional[str],
    hw: SystemHardwareResources,
    installed_models: List[str],
) -> Optional[str]:
    """
    Matches QuorumRAG AutoFusionRouter::find_best_installed_chat_model:
    Strictly filters out embedding, vision, decision-only, and memory-ineligible models,
    matching `preferred_tag` first and otherwise selecting the highest-capability eligible model.
    """
    if not installed_models:
        return None

    eligible = [
        m
        for m in installed_models
        if is_synthesis_chat_model(m) and model_fits_free_memory(m, hw.free_ram_gb, hw.free_vram_mb)
    ]
    if not eligible:
        return None

    if preferred_tag:
        pref_low = sanitize_model_tag(preferred_tag).lower()
        pref_latest = f"{pref_low}:latest"
        for cand in eligible:
            cl = cand.strip().lower()
            if cl == pref_low or cl == pref_latest or cl.startswith(f"{pref_low}:"):
                return cand

    eligible.sort(key=model_capability_score, reverse=True)
    return eligible[0]


def calculate_provision_plan(
    requested_tag: str,
    hw: SystemHardwareResources,
    installed_models: List[str],
) -> Dict[str, Any]:
    """
    Pre-provisioning resource calculation matching QuorumRAG's `ModelProvisionPlan`:
    Evaluates live free RAM, free GPU VRAM, and free disk headroom BEFORE any `ollama pull`,
    automatically downsizing oversized requests to a memory- and disk-safe tier.
    """
    clean_req = sanitize_model_tag(requested_tag)
    _, rec_qwen, _, _, _, _ = size_models_by_available_memory(hw.free_ram_gb, hw.free_vram_mb)
    rec_glm = size_glm_by_hardware(hw.free_ram_gb, hw.free_vram_mb)
    rec_gemma = size_gemma_by_hardware(hw.free_ram_gb, hw.free_vram_mb)

    req_fits_mem = bool(clean_req) and model_fits_free_memory(clean_req, hw.free_ram_gb, hw.free_vram_mb)
    req_fits_disk = bool(clean_req) and hw.free_disk_gb >= estimated_model_disk_gb(clean_req)

    if clean_req and req_fits_mem and req_fits_disk:
        effective_tag = clean_req
        downsized = False
    elif clean_req.lower().startswith("glm") and hw.free_disk_gb >= estimated_model_disk_gb(rec_glm):
        effective_tag = rec_glm
        downsized = clean_req.lower() != rec_glm.lower()
    elif clean_req.lower().startswith("gemma") and hw.free_disk_gb >= estimated_model_disk_gb(rec_gemma):
        effective_tag = rec_gemma
        downsized = clean_req.lower() != rec_gemma.lower()
    elif hw.free_disk_gb >= estimated_model_disk_gb(rec_qwen):
        effective_tag = rec_qwen
        downsized = clean_req.lower() != rec_qwen.lower()
    else:
        effective_tag = "qwen2.5:1.5b"
        downsized = clean_req.lower() != "qwen2.5:1.5b"

    already_installed = is_model_installed(effective_tag, installed_models)
    disk_sufficient = hw.free_disk_gb >= estimated_model_disk_gb(effective_tag)

    return {
        "requested_tag": clean_req,
        "effective_tag": effective_tag,
        "downsized_for_resources": downsized,
        "already_installed": already_installed,
        "disk_sufficient": disk_sufficient,
        "free_ram_gb": hw.free_ram_gb,
        "free_vram_mb": hw.free_vram_mb,
        "free_disk_gb": hw.free_disk_gb,
        "recommended_glm_tier": rec_glm,
        "recommended_qwen_tier": rec_qwen,
        "recommended_gemma_tier": rec_gemma,
    }


def select_model_calibrated(
    requested_family: str,
    prompt: str,
    hw: SystemHardwareResources,
    installed_models: List[str],
    context_chars: int = 0,
) -> Tuple[str, str, str]:
    """
    QuorumRAG AutoFusionRouter calibrated model selection:
    Routes across `glm` (`glm-4.7-flash` / `glm4:9b`), `qwen` (`qwen2.5:*`), `gemma` (`gemma4:26b`/`gemma2:9b`/`gemma3:4b`/`gemma2:2b`),
    or `auto` (code-dominated -> Qwen 2.5; long-context paper extraction/reasoning -> GLM when free RAM >= 12 GB or VRAM >= 4.5 GB),
    always calibrating against actually installed memory-eligible chat/completion models!

    Returns `(selected_family, selected_model_tag, rationale)`.
    """
    fam = (requested_family or "auto").strip().lower()
    _, ideal_qwen, _, _, _, _ = size_models_by_available_memory(hw.free_ram_gb, hw.free_vram_mb)
    ideal_glm = size_glm_by_hardware(hw.free_ram_gb, hw.free_vram_mb)
    ideal_gemma = size_gemma_by_hardware(hw.free_ram_gb, hw.free_vram_mb)

    if fam in ("glm", "glm4", "glm-4", "glm4_agentic"):
        matched = find_best_installed_chat_model(ideal_glm, hw, installed_models)
        if matched and matched.lower().startswith("glm"):
            return "glm", matched, f"GLM ({matched}) explicitly selected and confirmed installed (free RAM {hw.free_ram_gb} GB)"
        if matched:
            return "glm", matched, f"GLM ({ideal_glm}) requested but not installed -> calibrated to installed {matched}"
        return "glm", ideal_glm, f"GLM resource-sized tier: {ideal_glm} (free RAM {hw.free_ram_gb} GB, free VRAM {hw.free_vram_mb} MB)"

    if fam in ("gemma", "gemma4", "gemma4_26b"):
        pref = "gemma4:26b" if (fam == "gemma4_26b" and can_run_gemma4_26b(hw.free_ram_gb, hw.free_vram_mb)) else ideal_gemma
        matched = find_best_installed_chat_model(pref, hw, installed_models)
        if matched:
            return "gemma", matched, f"Gemma calibrated to installed {matched}"
        return "gemma", pref, f"Gemma resource-sized tier: {pref}"

    if fam in ("qwen", "qwen25", "qwen2.5"):
        matched = find_best_installed_chat_model(ideal_qwen, hw, installed_models)
        if matched:
            return "qwen", matched, f"Qwen 2.5 calibrated to installed {matched}"
        return "qwen", ideal_qwen, f"Qwen 2.5 resource-sized tier: {ideal_qwen}"

    # AutoFusion: code vs heavy multi-doc / paper reasoning (QuorumRAG parity)
    lower = (prompt or "").lower()
    is_code_task = any(
        kw in lower
        for kw in ("fn ", "def ", "impl ", "struct ", "rust", "python", "typescript", "code", "evolve", "verify")
    )
    is_long_doc_or_reasoning = (
        context_chars > 4_000
        or any(kw in lower for kw in ("langextract", "arxiv", "paper", "extract", "document", "abstract"))
    )

    if is_long_doc_or_reasoning and not is_code_task and (hw.free_ram_gb >= 12.0 or hw.free_vram_mb >= 4_500.0):
        pref = ideal_glm
        pref_fam = "glm"
    elif is_code_task:
        # Prefer installed GLM or Qwen calibrated to VRAM/RAM
        pref = "qwen2.5:7b" if (4_500.0 <= hw.free_vram_mb < 14_000.0 and is_model_installed("qwen2.5:7b", installed_models)) else ideal_qwen
        pref_fam = "qwen"
    elif hw.free_ram_gb >= 12.0 or hw.free_vram_mb >= 4_500.0:
        pref = ideal_glm
        pref_fam = "glm"
    else:
        pref = ideal_qwen
        pref_fam = "qwen"

    matched = find_best_installed_chat_model(pref, hw, installed_models)
    if matched:
        ml = matched.lower()
        res_fam = "glm" if ml.startswith("glm") else ("gemma" if ml.startswith("gemma") else ("qwen" if ml.startswith("qwen") else pref_fam))
        return res_fam, matched, f"AutoFusion calibrated to installed memory-eligible model {matched}"

    return pref_fam, pref, f"AutoFusion resource-sized recommendation: {pref} (free RAM {hw.free_ram_gb} GB, free VRAM {hw.free_vram_mb} MB, free disk {hw.free_disk_gb} GB)"


# =============================================================================
# Multi-Objective Pareto Search & UCB1 Bandit Scoring (ModelFusion Parity)
# =============================================================================

@dataclass
class ParetoCandidate:
    model_id: str
    objectives: List[float]  # [Maximize Quality, Maximize Velocity+Following, Maximize Efficiency+UCB1, Minimize MemoryGB]


def calculate_ucb1_exploration_bonus(alpha: float, total_pulls_n: int, model_pulls_n_i: int) -> float:
    n_eff = max(1.0, float(max(0, model_pulls_n_i)))
    n_total = max(2.0, float(max(0, total_pulls_n)))
    return alpha * math.sqrt((2.0 * math.log(n_total)) / n_eff)


def calculate_memory_penalty(required_gb: float, free_ram_gb: float, free_vram_gb: float = 0.0) -> float:
    effective_free = max(free_ram_gb, free_vram_gb * 2.0)
    if effective_free <= 0.0 and required_gb > 0.0:
        return 1000.0
    if required_gb > effective_free and effective_free > 0.0:
        return ((required_gb - effective_free) / effective_free) * 10.0
    return 0.0


def dominates(a: List[float], b: List[float], directions: List[str]) -> bool:
    """
    Returns True if objective vector `a` Pareto-dominates `b`.
    directions: list of 'max' or 'min'.
    """
    better_or_equal = True
    strictly_better = False
    for val_a, val_b, direction in zip(a, b, directions):
        diff = val_a - val_b
        if abs(diff) <= EPS:
            continue
        if direction == "max":
            if diff > EPS:
                strictly_better = True
            else:
                better_or_equal = False
                break
        else:
            if diff < -EPS:
                strictly_better = True
            else:
                better_or_equal = False
                break
    return better_or_equal and strictly_better


def compute_pareto_front(candidates: List[ParetoCandidate], directions: Optional[List[str]] = None) -> List[ParetoCandidate]:
    if directions is None:
        directions = ["max", "max", "max", "min"]
    front: List[ParetoCandidate] = []
    for i, c in enumerate(candidates):
        is_dominated = any(
            i != j and dominates(other.objectives, c.objectives, directions)
            for j, other in enumerate(candidates)
        )
        if not is_dominated:
            front.append(c)
    return front


def select_knee_point(front: List[ParetoCandidate], directions: Optional[List[str]] = None) -> Optional[ParetoCandidate]:
    """Selects the geometric knee-point closest to the utopia point (Benefit=1.0, Cost=0.0)."""
    if not front:
        return None
    if len(front) == 1:
        return front[0]
    if directions is None:
        directions = ["max", "max", "max", "min"]

    n = len(front)
    n_obj = len(directions)
    min_val = [min(c.objectives[j] for c in front) for j in range(n_obj)]
    max_val = [max(c.objectives[j] for c in front) for j in range(n_obj)]

    norm = [[1.0] * n_obj for _ in range(n)]
    for i, c in enumerate(front):
        for j in range(n_obj):
            lo, hi = min_val[j], max_val[j]
            if abs(hi - lo) < 1e-12:
                norm[i][j] = 1.0
            else:
                z = (c.objectives[j] - lo) / (hi - lo)
                norm[i][j] = z if directions[j] == "max" else (1.0 - z)

    max_indices = [j for j, d in enumerate(directions) if d == "max"]
    min_indices = [j for j, d in enumerate(directions) if d == "min"]

    benefit = [
        sum(norm[i][j] for j in max_indices) / max(1, len(max_indices))
        for i in range(n)
    ]
    costrisk = [
        1.0 - (sum(norm[i][j] for j in min_indices) / max(1, len(min_indices)))
        for i in range(n)
    ]

    utopia_dists = [
        math.sqrt((1.0 - benefit[i]) ** 2 + (costrisk[i]) ** 2)
        for i in range(n)
    ]
    best_idx = min(range(n), key=lambda i: utopia_dists[i])
    return front[best_idx]


# =============================================================================
# Fast 400ms Ollama Probe & Resource-Based Auto-Installer
# =============================================================================

def probe_ollama_models(endpoint: str = "http://127.0.0.1:11434", timeout_sec: float = 0.4) -> Tuple[bool, List[str]]:
    """
    Probes Ollama `/api/tags` with a strict 400ms timeout (Fast Timeout & Zero-Crash Fallback Law)
    and explicit no-proxy opener so Windows proxy resolution can never stall loopback probes.
    Caches result for 15 seconds within the process.
    Returns (is_online, list_of_installed_model_tags).
    """
    now = time.monotonic()
    cached = _OLLAMA_PROBE_CACHE.get(endpoint)
    if cached is not None and (now - cached[0]) < 15.0:
        return cached[1], list(cached[2])

    url = f"{endpoint.rstrip('/')}/api/tags"
    # NOTE: Never send `Origin: null` to Ollama — Ollama's CORS filter rejects `Origin: null` with HTTP 403!
    req = urllib.request.Request(url, headers={"User-Agent": "ArxivToCode-DynaRot/1.0"})
    opener = build_no_proxy_opener()
    try:
        with opener.open(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                # Preserve exact model tag casing (required for HuggingFace GGUF tags like hf.co/...:Q4_K_M)
                models = [
                    sanitize_model_tag(str(m.get("name", "")).strip())
                    for m in payload.get("models", [])
                    if m.get("name")
                ]
                _OLLAMA_PROBE_CACHE[endpoint] = (now, True, models)
                return True, models
    except Exception:
        pass
    _OLLAMA_PROBE_CACHE[endpoint] = (now, False, [])
    return False, []


def is_model_installed(model_tag: str, installed_tags: List[str]) -> bool:
    target = sanitize_model_tag(model_tag).lower()
    target_base, _, target_ver = target.partition(":")
    for inst in installed_tags:
        inst_l = inst.lower()
        if inst_l == target:
            return True
        inst_base, _, inst_ver = inst_l.partition(":")
        if inst_base == target_base and (not target_ver or inst_ver.startswith(target_ver)):
            return True
    return False


class DynaRotModelRouter:
    """
    ModelFusion DynaRot Router & Resource-Based Model Provisioner.
    Combines Dynamic Available Memory sizing, Pareto front knee-point selection,
    UCB1 bandit routing, and automatic Ollama model provisioning.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:11434",
        probe_timeout_sec: float = 0.4,
    ) -> None:
        self.endpoint = os.environ.get("LOCAL_OLLAMA_ENDPOINT", endpoint)
        self.probe_timeout_sec = probe_timeout_sec
        self.bandit_pulls: Dict[str, int] = {m["model_id"]: 1 for m in CURATED_MODEL_CATALOG}
        self.bandit_rewards: Dict[str, float] = {m["model_id"]: 0.85 for m in CURATED_MODEL_CATALOG}

    def record_bandit_feedback(self, model_id: str, reward: float) -> None:
        clean = sanitize_model_tag(model_id)
        pulls = self.bandit_pulls.get(clean, 1)
        prev_r = self.bandit_rewards.get(clean, 0.8)
        self.bandit_pulls[clean] = pulls + 1
        self.bandit_rewards[clean] = ((prev_r * pulls) + max(0.0, min(1.0, reward))) / (pulls + 1)

    def route(
        self,
        prompt: str = "Synthesize production Python code from arXiv research paper and evolve with ShinkaEvolve + OpenEvolve",
        override_free_ram_gb: Optional[float] = None,
        override_free_vram_mb: Optional[float] = None,
        override_free_disk_gb: Optional[float] = None,
        model_family: str = "auto",
    ) -> DynaRotRoutingDecision:
        start_ts = time.perf_counter()
        hw = detect_live_hardware(
            override_free_ram_gb=override_free_ram_gb,
            override_free_vram_mb=override_free_vram_mb,
            override_free_disk_gb=override_free_disk_gb,
        )

        complexity = self._calculate_complexity(prompt)
        requires_reasoning = complexity >= 5 or any(
            kw in prompt.lower()
            for kw in ("arxiv", "algorithm", "evolve", "prove", "theorem", "optimize", "synthesize")
        )

        (
            hardware_tier,
            primary_model,
            shinka_heavy,
            openevolve_fast,
            max_hw_ensemble,
            tier_pool,
        ) = size_models_by_available_memory(
            free_ram_gb=hw.free_ram_gb,
            free_vram_mb=hw.free_vram_mb,
            requires_reasoning=requires_reasoning,
        )

        ollama_online, installed_models = probe_ollama_models(
            endpoint=self.endpoint,
            timeout_sec=self.probe_timeout_sec,
        )

        # Evaluate Multi-Objective + UCB1 scores and Pareto front across models fitting in memory
        total_pulls = sum(self.bandit_pulls.values())
        ucb1_scores: Dict[str, float] = {}
        pareto_candidates: List[ParetoCandidate] = []

        for entry in CURATED_MODEL_CATALOG:
            mid = sanitize_model_tag(entry["model_id"])
            req_gb = estimate_runtime_memory_gb(entry["params_b"], "ollama")
            # Filter strictly by Available Runtime Memory Law
            if mid not in tier_pool and req_gb > max(hw.free_ram_gb * 0.75, hw.gpu_vram_free_gb * 1.5):
                continue

            q_task = (entry["capability_score"] * 0.6 + entry["decision_score"] * 0.4) / 10.0
            v_trending = min(1.0, max(0.0, entry["trending_score"] / 1000.0))
            b_following = 1.0 if entry["author"].lower() in HIGH_SIGNAL_RESEARCH_ORGS else 0.0
            log_dl = math.log10(1.0 + max(0, entry["downloads"]))
            ucb1 = calculate_ucb1_exploration_bonus(0.20, total_pulls, self.bandit_pulls.get(mid, 1))
            mem_pen = calculate_memory_penalty(req_gb, hw.free_ram_gb, hw.gpu_vram_free_gb)

            raw_score = (
                0.25 * q_task
                + 0.15 * (log_dl / 8.0)
                + 0.25 * v_trending
                + 0.15 * b_following
                + ucb1
                - mem_pen
            )
            if is_model_installed(mid, installed_models):
                raw_score += 0.25

            ucb1_scores[mid] = round(raw_score, 4)
            pareto_candidates.append(
                ParetoCandidate(
                    model_id=mid,
                    objectives=[
                        q_task,
                        v_trending + b_following,
                        (entry["efficiency_score"] / 10.0) + ucb1,
                        req_gb,
                    ],
                )
            )

        front = compute_pareto_front(pareto_candidates)
        knee = select_knee_point(front)
        pareto_front_ids = [c.model_id for c in front]
        pareto_knee_id = knee.model_id if knee else primary_model

        # Upgrade tiny knee point if hardware explicitly supports primary_model tier
        if hw.free_ram_gb >= 12.0 or hw.free_vram_mb >= 4_500.0:
            if pareto_knee_id in ("qwen2.5:0.5b", "qwen2.5:1.5b"):
                pareto_knee_id = primary_model

        target_ensemble = min(max_hw_ensemble, max(2, min(len(tier_pool), (complexity + 1) // 2)))
        suggested = tier_pool[:target_ensemble]

        recommended_glm = size_glm_by_hardware(hw.free_ram_gb, hw.free_vram_mb)
        recommended_gemma = size_gemma_by_hardware(hw.free_ram_gb, hw.free_vram_mb)
        _, selected_tag, selection_rationale = select_model_calibrated(
            requested_family=model_family,
            prompt=prompt,
            hw=hw,
            installed_models=installed_models,
        )
        calibrated_installed = find_best_installed_chat_model(selected_tag, hw, installed_models)

        # If caller explicitly requested GLM or Gemma family, honor it under resource constraints
        fam_low = (model_family or "auto").strip().lower()
        if fam_low in ("glm", "glm4", "glm-4", "glm4_agentic"):
            primary_model = recommended_glm
            shinka_heavy = recommended_glm
        elif fam_low in ("gemma", "gemma4", "gemma4_26b"):
            primary_model = recommended_gemma
            shinka_heavy = recommended_gemma

        provision_plan = calculate_provision_plan(primary_model, hw, installed_models)

        required_core_models = list(dict.fromkeys([primary_model, shinka_heavy, openevolve_fast]))
        missing = [
            m for m in required_core_models if not is_model_installed(m, installed_models)
        ]
        install_cmds = [f"ollama pull {m}" for m in missing]

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000.0, 3)

        return DynaRotRoutingDecision(
            task_category="arxiv-paper-to-python-code-evolution",
            target_language="python",
            complexity_score=complexity,
            requires_reasoning=requires_reasoning,
            hardware_tier=hardware_tier,
            primary_model=primary_model,
            shinka_heavy_reasoning_model=shinka_heavy,
            openevolve_fast_mutation_model=openevolve_fast,
            ensemble_size=target_ensemble,
            suggested_models=suggested,
            pareto_knee_model=pareto_knee_id,
            pareto_front_models=pareto_front_ids,
            ucb1_scores=ucb1_scores,
            installed_ollama_models=installed_models,
            missing_models=missing,
            auto_install_commands=install_cmds,
            routed_by="modelfusion_dynarot_quorum_calibrated" if ollama_online else "modelfusion_dynarot_heuristic_fallback",
            routing_duration_ms=elapsed_ms,
            hardware=hw.to_dict(),
            recommended_glm_model=recommended_glm,
            recommended_gemma_model=recommended_gemma,
            calibrated_installed_model=calibrated_installed,
            selection_rationale=selection_rationale,
            provision_plan=provision_plan,
        )

    def provision_models_for_resources(
        self,
        override_free_ram_gb: Optional[float] = None,
        override_free_vram_mb: Optional[float] = None,
        override_free_disk_gb: Optional[float] = None,
        model_family: str = "auto",
        auto_pull: bool = True,
        pull_timeout_sec: int = 120,
    ) -> Dict[str, Any]:
        """
        Determines optimal models for current runtime free RAM, VRAM & free disk using
        ModelFusion DynaRot + QuorumRAG `calculate_provision_plan()` BEFORE any pull,
        automatically downsizing oversized requests and pulling via Ollama (`ollama pull <model>`).
        """
        hw = detect_live_hardware(
            override_free_ram_gb=override_free_ram_gb,
            override_free_vram_mb=override_free_vram_mb,
            override_free_disk_gb=override_free_disk_gb,
        )
        decision = self.route(
            override_free_ram_gb=override_free_ram_gb,
            override_free_vram_mb=override_free_vram_mb,
            override_free_disk_gb=override_free_disk_gb,
            model_family=model_family,
        )
        ollama_bin = shutil.which("ollama")
        ollama_online, installed_before = probe_ollama_models(
            endpoint=self.endpoint,
            timeout_sec=self.probe_timeout_sec,
        )

        raw_targets = list(
            dict.fromkeys(
                [
                    decision.primary_model,
                    decision.shinka_heavy_reasoning_model,
                    decision.openevolve_fast_mutation_model,
                ]
            )
        )
        # Apply QuorumRAG pre-provisioning RAM/VRAM/Disk downsizing gate to every target
        target_models: List[str] = []
        target_plans: Dict[str, Dict[str, Any]] = {}
        for mt in raw_targets:
            plan = calculate_provision_plan(mt, hw, installed_before)
            eff = plan["effective_tag"]
            if eff not in target_plans:
                target_plans[eff] = plan
                target_models.append(eff)

        pull_results: List[Dict[str, Any]] = []
        for model_tag in target_models:
            clean_tag = sanitize_model_tag(model_tag)
            plan = target_plans[clean_tag]
            if not plan["disk_sufficient"]:
                pull_results.append(
                    {
                        "model": clean_tag,
                        "status": "skipped_insufficient_disk",
                        "free_disk_gb": plan["free_disk_gb"],
                        "required_disk_gb": estimated_model_disk_gb(clean_tag),
                        "command": f"ollama pull {clean_tag}",
                    }
                )
                continue
            if is_model_installed(clean_tag, installed_before):
                pull_results.append(
                    {
                        "model": clean_tag,
                        "status": "already_installed",
                        "command": f"ollama pull {clean_tag}",
                    }
                )
                continue

            if not auto_pull:
                pull_results.append(
                    {
                        "model": clean_tag,
                        "status": "pending_install",
                        "command": f"ollama pull {clean_tag}",
                    }
                )
                continue

            if not ollama_bin and not ollama_online:
                pull_results.append(
                    {
                        "model": clean_tag,
                        "status": "deferred_ollama_offline",
                        "command": f"ollama pull {clean_tag}",
                        "guidance": f"Start Ollama (`ollama serve`) and run `ollama pull {clean_tag}`",
                    }
                )
                continue

            # Attempt background or bounded CLI pull
            try:
                creationflags = 0x08000000 if platform.system() == "Windows" else 0
                proc = subprocess.run(
                    [ollama_bin or "ollama", "pull", clean_tag],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=pull_timeout_sec,
                    creationflags=creationflags,
                )
                if proc.returncode == 0:
                    pull_results.append(
                        {
                            "model": clean_tag,
                            "status": "installed",
                            "command": f"ollama pull {clean_tag}",
                        }
                    )
                else:
                    pull_results.append(
                        {
                            "model": clean_tag,
                            "status": "pull_failed_fallback_ready",
                            "command": f"ollama pull {clean_tag}",
                            "stderr": (proc.stderr or "").strip()[:240],
                        }
                    )
            except Exception as exc:
                pull_results.append(
                    {
                        "model": clean_tag,
                        "status": "pull_deferred_or_timed_out",
                        "command": f"ollama pull {clean_tag}",
                        "detail": str(exc)[:200],
                    }
                )

        _, installed_after = probe_ollama_models(
            endpoint=self.endpoint,
            timeout_sec=self.probe_timeout_sec,
        )

        return {
            "dynarot_decision": decision.to_dict(),
            "ollama_online": ollama_online,
            "target_models": target_models,
            "installed_models": installed_after or installed_before,
            "provisioning_results": pull_results,
            "guidance_card": {
                "recommended_primary_model": decision.primary_model,
                "shinka_heavy_reasoning_model": decision.shinka_heavy_reasoning_model,
                "openevolve_fast_mutation_model": decision.openevolve_fast_mutation_model,
                "terminal_commands": [f"ollama pull {m}" for m in target_models],
                "installed_model_pills": installed_after or installed_before,
            },
        }

    @staticmethod
    def _calculate_complexity(prompt: str) -> int:
        lower = prompt.lower()
        score = 5
        if len(prompt) > 250:
            score += 1
        if len(prompt) > 700:
            score += 1
        for kw in (
            "arxiv",
            "langextract",
            "evolve",
            "shinkaevolve",
            "openevolve",
            "map-elites",
            "pareto",
            "conformal",
            "martingale",
            "algorithm",
            "proof",
            "theorem",
        ):
            if kw in lower:
                score += 1
        return max(1, min(10, score))
