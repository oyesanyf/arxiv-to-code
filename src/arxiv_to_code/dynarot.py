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
    "google",
    "meta-llama",
    "mistralai",
    "microsoft",
    "allenai",
    "nvidia",
)

# Curated open-weight catalog with parameter counts, capability scores, and popularity metrics
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


def _detect_gpu_vram() -> Tuple[Optional[str], float, float]:
    """Detects GPU name, total VRAM (GB), and free VRAM (GB) via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None, 0.0, 0.0
    try:
        creationflags = 0x08000000 if platform.system() == "Windows" else 0
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=1.5,
            creationflags=creationflags,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                total_gb = round(float(parts[1]) / 1024.0, 2)
                free_gb = round(float(parts[2]) / 1024.0, 2)
                return name, total_gb, free_gb
    except Exception:
        pass
    return None, 0.0, 0.0


def detect_live_hardware(
    override_free_ram_gb: Optional[float] = None,
    override_free_vram_mb: Optional[float] = None,
) -> SystemHardwareResources:
    """
    Probes live available system RAM and GPU VRAM at runtime.
    Never sizes against total RAM when free RAM is available.
    """
    total_ram_gb, free_ram_gb = _detect_windows_ram_gb()
    gpu_name, gpu_vram_total_gb, gpu_vram_free_gb = _detect_gpu_vram()
    cpu_cores = os.cpu_count() or 4

    if override_free_ram_gb is not None:
        free_ram_gb = float(override_free_ram_gb)
        total_ram_gb = max(total_ram_gb, free_ram_gb)
    if override_free_vram_mb is not None:
        gpu_vram_free_gb = round(float(override_free_vram_mb) / 1024.0, 2)
        gpu_vram_total_gb = max(gpu_vram_total_gb, gpu_vram_free_gb)
        if gpu_vram_free_gb > 0 and gpu_name is None:
            gpu_name = "Simulated/Override GPU"

    return SystemHardwareResources(
        total_ram_gb=total_ram_gb,
        free_ram_gb=free_ram_gb,
        cpu_cores=cpu_cores,
        gpu_name=gpu_name,
        gpu_vram_total_gb=gpu_vram_total_gb,
        gpu_vram_free_gb=gpu_vram_free_gb,
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
            "deepseek-r1:32b",
            "qwen2.5:14b",
            "deepseek-r1:14b",
            "qwen2.5:7b",
            "deepseek-r1:7b",
            "qwen2.5:3b",
        ]
    elif free_ram_gb >= 24.0 or free_vram_mb >= 14_000.0:
        tier = "Tier 2: High-Throughput 14B (Free RAM >= 24 GB or Free VRAM >= 14 GB)"
        primary = "qwen2.5:14b"
        shinka_heavy = "qwen2.5:14b"
        openevolve_fast = "qwen2.5:3b"
        max_ensemble = 5
        pool = [
            "qwen2.5:14b",
            "deepseek-r1:14b",
            "qwen2.5:7b",
            "deepseek-r1:7b",
            "qwen2.5:3b",
        ]
    elif free_ram_gb >= 12.0 or free_vram_mb >= 4_500.0:
        tier = "Tier 3: GPU-Resident 7B (Free RAM >= 12 GB or Free VRAM >= 4.5 GB)"
        primary = "qwen2.5:7b"
        shinka_heavy = "qwen2.5:7b"
        openevolve_fast = "qwen2.5:1.5b"
        max_ensemble = 4
        pool = [
            "qwen2.5:7b",
            "deepseek-r1:7b",
            "qwen2.5:3b",
            "deepseek-r1:1.5b",
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
            "qwen2.5:0.5b",
        ]

    # Guarantee zero '-coder' suffixes across all returned model tags
    primary = sanitize_model_tag(primary)
    shinka_heavy = sanitize_model_tag(shinka_heavy)
    openevolve_fast = sanitize_model_tag(openevolve_fast)
    pool = [sanitize_model_tag(m) for m in pool]

    return tier, primary, shinka_heavy, openevolve_fast, max_ensemble, pool


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
    Probes Ollama `/api/tags` with a strict 400ms timeout (Fast Timeout & Zero-Crash Fallback Law).
    Returns (is_online, list_of_installed_model_tags).
    """
    url = f"{endpoint.rstrip('/')}/api/tags"
    req = urllib.request.Request(url, headers={"User-Agent": "ArxivToCode-DynaRot/1.0", "Origin": "null"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                models = [
                    sanitize_model_tag(str(m.get("name", "")).lower())
                    for m in payload.get("models", [])
                    if m.get("name")
                ]
                return True, models
    except Exception:
        pass
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
    ) -> DynaRotRoutingDecision:
        start_ts = time.perf_counter()
        hw = detect_live_hardware(
            override_free_ram_gb=override_free_ram_gb,
            override_free_vram_mb=override_free_vram_mb,
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
            routed_by="modelfusion_dynarot_pareto_ucb1" if ollama_online else "modelfusion_dynarot_heuristic_fallback",
            routing_duration_ms=elapsed_ms,
            hardware=hw.to_dict(),
        )

    def provision_models_for_resources(
        self,
        override_free_ram_gb: Optional[float] = None,
        override_free_vram_mb: Optional[float] = None,
        auto_pull: bool = True,
        pull_timeout_sec: int = 120,
    ) -> Dict[str, Any]:
        """
        Determines optimal models for current runtime free RAM & VRAM using DynaRot
        and installs/pulls missing models via Ollama (`ollama pull <model>`).
        """
        decision = self.route(
            override_free_ram_gb=override_free_ram_gb,
            override_free_vram_mb=override_free_vram_mb,
        )
        ollama_bin = shutil.which("ollama")
        ollama_online, installed_before = probe_ollama_models(
            endpoint=self.endpoint,
            timeout_sec=self.probe_timeout_sec,
        )

        target_models = list(
            dict.fromkeys(
                [
                    decision.primary_model,
                    decision.shinka_heavy_reasoning_model,
                    decision.openevolve_fast_mutation_model,
                ]
            )
        )

        pull_results: List[Dict[str, Any]] = []
        for model_tag in target_models:
            clean_tag = sanitize_model_tag(model_tag)
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
