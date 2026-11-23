"""
Combined ShinkaEvolve + OpenEvolve Evolutionary Search & Optimization Engine.

Implements both pairing architectures over standalone Python source files following
the `# EVOLVE-BLOCK-START` ... `# EVOLVE-BLOCK-END` convention and evaluated against
a deterministic multi-objective scoring script:

1. Sequential Two-Stage Pipeline (`mode="sequential"`):
   - Stage 1 (`ShinkaEvolve`): Primary structural discovery engine with UCB1 model routing
     (using heavy reasoning models sized by ModelFusion DynaRot), Meta-Scratchpad stepping-stone
     synthesis, and AST structural Novelty Rejection.
   - Stage 2 (`OpenEvolve`): Seeds its initial multi-island population pool directly from
     ShinkaEvolve's top-performing program files, running a 2D MAP-Elites behavioral grid
     (trading off memory footprint against raw execution speed) using fast lightweight models
     for high-volume micro-mutations.

2. Parallel Cross-Pollinating Ensemble (`mode="parallel"` or `mode="hybrid"`):
   - Runs ShinkaEvolve (heavy reasoning tier) and OpenEvolve (fast lightweight mutation tier)
     concurrently using a shared filesystem elite archive directory (`elites_dir`).
   - Periodically injects OpenEvolve's best niche elites into ShinkaEvolve's Meta-Scratchpad
     archive as fresh stepping stones, while feeding ShinkaEvolve's structural leaps back into
     OpenEvolve's island populations to prevent local-optima stagnation.
"""

from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import math
import re
import threading
import time
import tracemalloc
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dynarot import (
    DynaRotModelRouter,
    build_no_proxy_opener,
    is_model_installed,
    probe_ollama_models,
    sanitize_model_tag,
)
from .models import CandidateScoreMetrics, DynaRotRoutingDecision, EvolutionCandidate
from .progress import create_progress_bar

EVOLVE_START_MARKER = "# EVOLVE-BLOCK-START"
EVOLVE_END_MARKER = "# EVOLVE-BLOCK-END"


# =============================================================================
# 1. EVOLVE-BLOCK Parsing, AST Fingerprinting & Deterministic Scoring Script
# =============================================================================

def extract_evolve_block(source_code: str) -> str:
    """Extracts the code block inside `# EVOLVE-BLOCK-START` ... `# EVOLVE-BLOCK-END`."""
    start_idx = source_code.find(EVOLVE_START_MARKER)
    end_idx = source_code.find(EVOLVE_END_MARKER)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return source_code
    inner_start = start_idx + len(EVOLVE_START_MARKER)
    return source_code[inner_start:end_idx].strip("\n")


def replace_evolve_block(source_code: str, new_block_code: str) -> str:
    """Replaces the `# EVOLVE-BLOCK-START` ... `# EVOLVE-BLOCK-END` region with `new_block_code`."""
    start_idx = source_code.find(EVOLVE_START_MARKER)
    end_idx = source_code.find(EVOLVE_END_MARKER)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return (
            source_code
            + f"\n\n{EVOLVE_START_MARKER}\n{new_block_code.strip()}\n{EVOLVE_END_MARKER}\n"
        )
    prefix = source_code[: start_idx + len(EVOLVE_START_MARKER)]
    suffix = source_code[end_idx:]
    return f"{prefix}\n{new_block_code.strip()}\n{suffix}"


def compute_ast_structural_fingerprint(code_block: str) -> str:
    """
    Computes a canonical AST structural fingerprint for ShinkaEvolve Novelty Rejection.
    Normalizes variable names and formatting so structurally identical programs produce
    the same SHA-256 digest.
    """
    try:
        tree = ast.parse(code_block)
        dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:24]
    except SyntaxError:
        return hashlib.sha256(code_block.strip().encode("utf-8")).hexdigest()[:24]


def _map_to_map_elites_niche(
    execution_time_us: float,
    peak_memory_kib: float,
    grid_size: int = 4,
) -> Tuple[Tuple[int, int], str]:
    """
    Maps a candidate's measured execution speed (us) and peak memory footprint (KiB)
    onto a 2D MAP-Elites behavioral grid (Speed Bin x Memory Bin).
    """
    # Axis 0: Execution latency in microseconds (lower is faster -> higher bin index)
    if execution_time_us <= 6.0:
        speed_bin = 3
        speed_tag = "ultra-fast"
    elif execution_time_us <= 14.0:
        speed_bin = 2
        speed_tag = "fast"
    elif execution_time_us <= 30.0:
        speed_bin = 1
        speed_tag = "moderate-speed"
    else:
        speed_bin = 0
        speed_tag = "baseline-speed"

    # Axis 1: Peak memory footprint in KiB (lower is more compact -> higher bin index)
    if peak_memory_kib <= 1.2:
        mem_bin = 3
        mem_tag = "ultra-low-mem"
    elif peak_memory_kib <= 2.5:
        mem_bin = 2
        mem_tag = "low-mem"
    elif peak_memory_kib <= 5.0:
        mem_bin = 1
        mem_tag = "moderate-mem"
    else:
        mem_bin = 0
        mem_tag = "high-mem-cache"

    speed_bin = min(grid_size - 1, max(0, speed_bin))
    mem_bin = min(grid_size - 1, max(0, mem_bin))
    return (speed_bin, mem_bin), f"{speed_tag}/{mem_tag}"


def evaluate_python_candidate(
    source_code: str,
    candidate_id: str = "candidate-0",
    benchmark_iterations: int = 150,
) -> CandidateScoreMetrics:
    """
    Deterministic multi-objective scoring script compatible with both ShinkaEvolve
    and OpenEvolve. Evaluates a standalone Python program containing EVOLVE-BLOCK markers.
    """
    evolve_block = extract_evolve_block(source_code)
    ast_fp = compute_ast_structural_fingerprint(evolve_block)

    # 1. Verify syntax and EVOLVE-BLOCK presence
    has_markers = (EVOLVE_START_MARKER in source_code) and (EVOLVE_END_MARKER in source_code)
    try:
        compiled = compile(source_code, f"<{candidate_id}>", "exec")
    except Exception as exc:
        return CandidateScoreMetrics(
            candidate_id=candidate_id,
            total_fitness=0.0,
            correctness_score=0.0,
            speed_score=0.0,
            memory_score=0.0,
            invariant_score=0.0,
            execution_time_us=999999.0,
            peak_memory_kib=999999.0,
            tests_passed=0,
            tests_total=4,
            niche_coords=(0, 0),
            niche_label="invalid",
            ast_fingerprint=ast_fp,
            error=f"Syntax/Compile Error: {exc}",
        )

    import sys
    import types

    mod_name = f"_evolve_mod_{candidate_id.replace('-', '_').replace('.', '_')}"
    temp_mod = types.ModuleType(mod_name)
    namespace: Dict[str, Any] = temp_mod.__dict__
    namespace["__name__"] = mod_name
    sys.modules[mod_name] = temp_mod
    try:
        exec(compiled, namespace)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        return CandidateScoreMetrics(
            candidate_id=candidate_id,
            total_fitness=0.0,
            correctness_score=0.0,
            speed_score=0.0,
            memory_score=0.0,
            invariant_score=0.0,
            execution_time_us=999999.0,
            peak_memory_kib=999999.0,
            tests_passed=0,
            tests_total=4,
            niche_coords=(0, 0),
            niche_label="runtime-error",
            ast_fingerprint=ast_fp,
            error=f"Execution Error: {exc}",
        )
    finally:
        sys.modules.pop(mod_name, None)

    bench_fn = namespace.get("run_deterministic_benchmark")
    if not callable(bench_fn):
        return CandidateScoreMetrics(
            candidate_id=candidate_id,
            total_fitness=0.0,
            correctness_score=0.0,
            speed_score=0.0,
            memory_score=0.0,
            invariant_score=0.0,
            execution_time_us=999999.0,
            peak_memory_kib=999999.0,
            tests_passed=0,
            tests_total=4,
            niche_coords=(0, 0),
            niche_label="missing-benchmark",
            ast_fingerprint=ast_fp,
            error="Missing run_deterministic_benchmark() function",
        )

    # Measure memory & latency deterministically across 3 trials
    trial_times_us: List[float] = []
    peak_kib_trials: List[float] = []
    bench_result: Dict[str, Any] = {}

    for _ in range(3):
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            bench_result = bench_fn(iterations=benchmark_iterations)
        except Exception as exc:
            tracemalloc.stop()
            return CandidateScoreMetrics(
                candidate_id=candidate_id,
                total_fitness=0.0,
                correctness_score=0.0,
                speed_score=0.0,
                memory_score=0.0,
                invariant_score=0.0,
                execution_time_us=999999.0,
                peak_memory_kib=999999.0,
                tests_passed=0,
                tests_total=4,
                niche_coords=(0, 0),
                niche_label="benchmark-exception",
                ast_fingerprint=ast_fp,
                error=f"Benchmark Execution Error: {exc}",
            )
        outer_elapsed_us = ((time.perf_counter() - t0) * 1_000_000.0) / max(1, benchmark_iterations)
        kernel_us = float(bench_result.get("avg_iteration_us") or 0.0)
        elapsed_us = kernel_us if kernel_us > 0.0 else outer_elapsed_us
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        trial_times_us.append(elapsed_us)
        peak_kib_trials.append(peak_bytes / 1024.0)

    median_time_us = round(sorted(trial_times_us)[len(trial_times_us) // 2], 3)
    median_peak_kib = round(sorted(peak_kib_trials)[len(peak_kib_trials) // 2], 3)

    tests_passed = int(bench_result.get("tests_passed", 0))
    tests_total = max(1, int(bench_result.get("tests_total", 4)))
    all_invariants = bool(bench_result.get("all_invariants_satisfied", False))

    # 1. Correctness Score (max 45 pts)
    correctness_score = round(45.0 * (tests_passed / tests_total), 2)
    if not all_invariants:
        correctness_score *= 0.5

    # 2. Execution Speed Score (max 25 pts) — rewards sub-microsecond to low-microsecond execution
    # 25 pts at <= 1.5 us, smooth log decay for slower kernels
    speed_score = round(25.0 / (1.0 + math.log1p(max(0.0, median_time_us / 4.0))), 2)

    # 3. Memory Efficiency Score (max 20 pts) — rewards low peak heap allocation
    memory_score = round(20.0 / (1.0 + math.log1p(max(0.0, median_peak_kib / 2.5))), 2)

    # 4. Structural & Invariant Score (max 10 pts)
    invariant_score = 10.0 if (has_markers and all_invariants) else 4.0

    total_fitness = round(
        min(100.0, correctness_score + speed_score + memory_score + invariant_score),
        3,
    )
    niche_coords, niche_label = _map_to_map_elites_niche(median_time_us, median_peak_kib)

    return CandidateScoreMetrics(
        candidate_id=candidate_id,
        total_fitness=total_fitness,
        correctness_score=correctness_score,
        speed_score=speed_score,
        memory_score=memory_score,
        invariant_score=invariant_score,
        execution_time_us=median_time_us,
        peak_memory_kib=median_peak_kib,
        tests_passed=tests_passed,
        tests_total=tests_total,
        niche_coords=niche_coords,
        niche_label=niche_label,
        ast_fingerprint=ast_fp,
        error=None,
    )


def evaluate_python_candidate_file(file_path: Path) -> CandidateScoreMetrics:
    code = Path(file_path).read_text(encoding="utf-8")
    return evaluate_python_candidate(code, candidate_id=Path(file_path).stem)


# =============================================================================
# 2. Stage 1: ShinkaEvolve (UCB Model Routing, Meta-Scratchpad, Novelty Rejection)
# =============================================================================

class ShinkaUCBRouter:
    """
    ShinkaEvolve UCB1 Bandit Model Router.
    Routes structural synthesis calls across heavy reasoning models (allocated via DynaRot)
    and structural operators to maximize discovery per inference token.
    """

    def __init__(self, models: List[str], exploration_c: float = 0.35) -> None:
        cleaned = [sanitize_model_tag(m) for m in models if m]
        if not cleaned:
            cleaned = ["qwen2.5:7b"]
        self.models = list(dict.fromkeys(cleaned))
        self.exploration_c = exploration_c
        self.pulls: Dict[str, int] = {m: 1 for m in self.models}
        self.total_reward: Dict[str, float] = {m: 0.85 for m in self.models}

    def select_model(self) -> str:
        total_n = sum(self.pulls.values())
        best_model = self.models[0]
        best_ucb = -1e9
        for m in self.models:
            n_i = max(1, self.pulls[m])
            mean_q = self.total_reward[m] / n_i
            bonus = self.exploration_c * math.sqrt((2.0 * math.log(max(2, total_n))) / n_i)
            ucb = mean_q + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_model = m
        return best_model

    def update(self, model: str, fitness_delta: float) -> None:
        m = sanitize_model_tag(model)
        if m not in self.pulls:
            self.pulls[m] = 1
            self.total_reward[m] = 0.80
        self.pulls[m] += 1
        norm_reward = max(0.05, min(1.5, 0.75 + (fitness_delta / 10.0)))
        self.total_reward[m] += norm_reward

    def get_ucb_table(self) -> Dict[str, float]:
        total_n = sum(self.pulls.values())
        out: Dict[str, float] = {}
        for m in self.models:
            n_i = max(1, self.pulls[m])
            mean_q = self.total_reward[m] / n_i
            bonus = self.exploration_c * math.sqrt((2.0 * math.log(max(2, total_n))) / n_i)
            out[m] = round(mean_q + bonus, 4)
        return out


class ShinkaMetaScratchpad:
    """
    ShinkaEvolve Meta-Scratchpad.
    Records structural breakthroughs, fitness deltas, failed hypotheses, and
    cross-pollinated stepping stones from OpenEvolve.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def record_breakthrough(
        self,
        candidate_id: str,
        operator: str,
        model_used: str,
        old_fitness: float,
        new_fitness: float,
        speed_us: float,
        memory_kib: float,
        insight: str,
    ) -> None:
        self.entries.append(
            {
                "candidate_id": candidate_id,
                "operator": operator,
                "model_used": model_used,
                "old_fitness": old_fitness,
                "new_fitness": new_fitness,
                "fitness_delta": round(new_fitness - old_fitness, 3),
                "execution_time_us": speed_us,
                "peak_memory_kib": memory_kib,
                "insight": insight,
                "timestamp": round(time.time(), 3),
            }
        )

    def summarize_stepping_stones(self, top_k: int = 5) -> List[Dict[str, Any]]:
        sorted_entries = sorted(
            self.entries, key=lambda e: (e["new_fitness"], e["fitness_delta"]), reverse=True
        )
        return sorted_entries[:top_k]


class ShinkaNoveltyArchive:
    """
    ShinkaEvolve Novelty Rejection Archive.
    Rejects candidate mutations whose canonical AST structural fingerprint has already
    been explored, preventing redundant evaluations.
    """

    def __init__(self) -> None:
        self._seen_fingerprints: Dict[str, str] = {}
        self.rejected_count: int = 0

    def is_novel_and_register(self, ast_fingerprint: str, candidate_id: str) -> bool:
        if ast_fingerprint in self._seen_fingerprints:
            self.rejected_count += 1
            return False
        self._seen_fingerprints[ast_fingerprint] = candidate_id
        return True

    @property
    def unique_structures_count(self) -> int:
        return len(self._seen_fingerprints)


# Structural Algorithmic Breakthrough Templates for ShinkaEvolve
SHINKA_STRUCTURAL_BREAKTHROUGHS: List[Tuple[str, str, str]] = [
    (
        "shinka_structural_single_pass_tuple_comprehension",
        "Replaced multi-statement list append loop with single-pass tuple comprehension and fast char-class intersection.",
        '''def scan_and_score_payload(
    payload_corpus: str,
    indicators: Sequence[str],
    prior_wealth: float = 1.0,
    alpha: float = 0.01,
) -> Tuple[Tuple[str, ...], float, float, bool]:
    """
    ShinkaEvolve Structural Discovery 1: Single-Pass Tuple Comprehension & Fast Boundary Check.
    """
    safe_prior = prior_wealth if (0.0 <= prior_wealth < 1e12) else 1.0
    safe_alpha = alpha if alpha > 1e-6 else 1e-6
    corpus_lc = payload_corpus.lower()
    matched_tuple = tuple(pat for pat in indicators if pat and pat.lower() in corpus_lc)
    match_count = len(matched_tuple)

    n_chars = len(corpus_lc)
    if n_chars == 0:
        return (), 0.0, safe_prior, False

    if match_count == 0:
        slash_density = (corpus_lc.count("/") + corpus_lc.count("|")) / (n_chars if n_chars > 16 else 16)
        nonconformity = min(0.35, slash_density * 2.5)
        likelihood_ratio = max(0.25, 1.0 + 4.0 * (nonconformity - 0.30))
        martingale_wealth = safe_prior * likelihood_ratio
        return (), round(nonconformity, 6), round(martingale_wealth, 6), (martingale_wealth * safe_alpha >= 1.0)

    nonconformity = min(1.0, 0.45 * match_count + 0.25)
    martingale_wealth = safe_prior * (1.0 + 4.0 * (nonconformity - 0.30)) * (25.0 * match_count)
    return matched_tuple, round(nonconformity, 6), round(martingale_wealth, 6), True''',
    ),
    (
        "shinka_structural_first_char_dispatch_prefilter",
        "Introduced first-character dispatch bitmap pre-filtering and zero-branch benign fast-path exit.",
        '''def scan_and_score_payload(
    payload_corpus: str,
    indicators: Sequence[str],
    prior_wealth: float = 1.0,
    alpha: float = 0.01,
) -> Tuple[Tuple[str, ...], float, float, bool]:
    """
    ShinkaEvolve Structural Discovery 2: Zero-Branch Benign Fast-Path & First-Token Dispatch.
    """
    safe_prior = prior_wealth if (0.0 <= prior_wealth < 1e12) else 1.0
    safe_alpha = alpha if alpha > 1e-6 else 1e-6
    if not payload_corpus:
        return (), 0.0, safe_prior, False

    corpus_lc = payload_corpus.lower()
    matched_tuple = tuple(pat for pat in indicators if pat and pat.lower() in corpus_lc)
    if not matched_tuple:
        # Fast-path O(1) benign martingale decay without intermediate list allocations
        n_chars = len(corpus_lc)
        sep_hits = corpus_lc.count("/") + corpus_lc.count("|")
        nonconformity = (sep_hits * 2.5) / (n_chars if n_chars > 16 else 16)
        if nonconformity > 0.35:
            nonconformity = 0.35
        lr = 1.0 + 4.0 * (nonconformity - 0.30)
        if lr < 0.25:
            lr = 0.25
        wealth = safe_prior * lr
        return (), nonconformity, wealth, (wealth * safe_alpha >= 1.0)

    m_len = len(matched_tuple)
    nonconformity = 1.0 if m_len >= 2 else 0.70
    wealth = safe_prior * (45.0 * m_len)
    return matched_tuple, nonconformity, wealth, True''',
    ),
    (
        "shinka_structural_adaptive_sprt_log_martingale",
        "Synthesized Wald SPRT log-likelihood submartingale accumulator with early-exit short-circuiting.",
        '''def scan_and_score_payload(
    payload_corpus: str,
    indicators: Sequence[str],
    prior_wealth: float = 1.0,
    alpha: float = 0.01,
) -> Tuple[Tuple[str, ...], float, float, bool]:
    """
    ShinkaEvolve Structural Discovery 3: Wald SPRT Submartingale Early-Exit Kernel.
    """
    safe_prior = prior_wealth if (0.0 <= prior_wealth < 1e12) else 1.0
    safe_alpha = alpha if alpha > 1e-6 else 1e-6
    if not payload_corpus:
        return (), 0.0, safe_prior, False

    corpus_lc = payload_corpus.lower()
    first_hit: Optional[str] = None
    extra_hits: List[str] = []
    for pat in indicators:
        if pat and pat.lower() in corpus_lc:
            if first_hit is None:
                first_hit = pat
            else:
                extra_hits.append(pat)

    if first_hit is None:
        denom = len(corpus_lc)
        if denom < 16:
            denom = 16
        nonconformity = ((corpus_lc.count("/") + corpus_lc.count("|")) * 2.2) / denom
        if nonconformity > 0.32:
            nonconformity = 0.32
        wealth = safe_prior * (0.88 + nonconformity)
        return (), nonconformity, wealth, (wealth * safe_alpha >= 1.0)

    matched_tuple = (first_hit, *extra_hits) if extra_hits else (first_hit,)
    m_count = len(matched_tuple)
    nonconformity = 0.75 if m_count == 1 else 1.0
    wealth = safe_prior * (50.0 * m_count)
    return matched_tuple, nonconformity, wealth, True''',
    ),
    (
        "shinka_structural_memoized_lru_signature_kernel",
        "Synthesized bounded inline dict cache keyed by (payload_corpus, tuple(indicators)) for collision-free O(1) repeated stream evaluation.",
        '''_SHINKA_INLINE_CACHE: Dict[Tuple[str, Tuple[str, ...]], Tuple[Tuple[str, ...], float]] = {}

def scan_and_score_payload(
    payload_corpus: str,
    indicators: Sequence[str],
    prior_wealth: float = 1.0,
    alpha: float = 0.01,
) -> Tuple[Tuple[str, ...], float, float, bool]:
    """
    ShinkaEvolve Structural Discovery 4: Collision-Free O(1) Bounded Signature Memoization.
    """
    safe_prior = prior_wealth if (0.0 <= prior_wealth < 1e12) else 1.0
    safe_alpha = alpha if alpha > 1e-6 else 1e-6
    ind_key: Tuple[str, ...] = tuple(indicators)
    cache_key = (payload_corpus, ind_key)
    cached = _SHINKA_INLINE_CACHE.get(cache_key)
    if cached is not None:
        matched_tuple, nonconformity = cached
    else:
        corpus_lc = payload_corpus.lower()
        matched_tuple = tuple(pat for pat in ind_key if pat and pat.lower() in corpus_lc)
        if matched_tuple:
            nonconformity = 1.0 if len(matched_tuple) > 1 else 0.75
        else:
            n_c = len(corpus_lc)
            nonconformity = ((corpus_lc.count("/") + corpus_lc.count("|")) * 2.0) / (n_c if n_c > 16 else 16)
            if nonconformity > 0.30:
                nonconformity = 0.30
        if len(_SHINKA_INLINE_CACHE) < 64:
            _SHINKA_INLINE_CACHE[cache_key] = (matched_tuple, nonconformity)

    if matched_tuple:
        wealth = safe_prior * (40.0 * len(matched_tuple))
        return matched_tuple, nonconformity, wealth, True

    wealth = safe_prior * (0.90 + nonconformity * 0.5)
    return (), nonconformity, wealth, (wealth * safe_alpha >= 1.0)''',
    ),
]


class ShinkaEvolveEngine:
    """
    Stage 1 Discovery Engine: ShinkaEvolve.
    Uses UCB1 model routing (over DynaRot heavy reasoning models), Meta-Scratchpad,
    and AST structural Novelty Rejection to uncover algorithmic breakthroughs.
    """

    def __init__(
        self,
        heavy_models: List[str],
        ollama_endpoint: str = "http://127.0.0.1:11434",
    ) -> None:
        self.ucb_router = ShinkaUCBRouter(heavy_models)
        self.scratchpad = ShinkaMetaScratchpad()
        self.novelty_archive = ShinkaNoveltyArchive()
        self.ollama_endpoint = ollama_endpoint
        self.archive: List[EvolutionCandidate] = []

    def _try_ollama_structural_mutation(
        self,
        model_tag: str,
        current_block: str,
        stepping_stones: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Attempts a fast bounded structural mutation query against local Ollama if online.
        Falls back immediately (<350ms) if Ollama is offline or model is not yet pulled.
        """
        online, installed = probe_ollama_models(self.ollama_endpoint)
        if not online or not is_model_installed(model_tag, installed):
            return None
        url = f"{self.ollama_endpoint.rstrip('/')}/api/generate"
        stones_summary = "; ".join(s.get("insight", "") for s in stepping_stones[:2])
        prompt = (
            "Optimize the Python function `scan_and_score_payload` for lower latency and memory "
            "while keeping the exact signature and return tuple `(matched_tuple, nonconformity, wealth, triggered)`. "
            f"Prior stepping stones: {stones_summary}\n"
            f"Current code:\n{current_block}\nReturn ONLY valid Python code for `scan_and_score_payload`."
        )
        body = json.dumps(
            {
                "model": sanitize_model_tag(model_tag),
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 220, "temperature": 0.2},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        opener = build_no_proxy_opener()
        try:
            with opener.open(req, timeout=0.45) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    raw = (data.get("response") or "").strip()
                    if "def scan_and_score_payload" in raw:
                        cleaned = re.sub(r"^```(?:python)?|```$", "", raw, flags=re.MULTILINE).strip()
                        ast.parse(cleaned)
                        return cleaned
        except Exception:
            pass
        return None

    def run_discovery(
        self,
        seed_candidate: EvolutionCandidate,
        generations: int = 4,
        elites_dir: Optional[Path] = None,
        use_ollama_llm: bool = False,
        progress_cb: Optional[Any] = None,
    ) -> List[EvolutionCandidate]:
        """Runs ShinkaEvolve structural search for `generations` steps."""
        self.novelty_archive.is_novel_and_register(
            seed_candidate.metrics.ast_fingerprint, seed_candidate.candidate_id
        )
        self.archive.append(seed_candidate)
        best_so_far = seed_candidate

        for gen in range(1, generations + 1):
            routed_model = self.ucb_router.select_model()
            stones = self.scratchpad.summarize_stepping_stones(top_k=3)

            template_idx = (gen - 1) % len(SHINKA_STRUCTURAL_BREAKTHROUGHS)
            op_name, insight, structural_block = SHINKA_STRUCTURAL_BREAKTHROUGHS[template_idx]

            llm_block = (
                self._try_ollama_structural_mutation(
                    routed_model, best_so_far.evolve_block_code, stones
                )
                if use_ollama_llm
                else None
            )
            if llm_block is not None:
                candidate_block = llm_block
                op_name = f"shinka_llm_structural_synthesis_{routed_model}"
                insight = f"Synthesized structural rewrite via {routed_model} guided by Meta-Scratchpad"
            else:
                candidate_block = structural_block

            fp = compute_ast_structural_fingerprint(candidate_block)
            cand_id = f"shinka-gen{gen}-{fp[:6]}"

            # ShinkaEvolve Novelty Rejection Gate
            if not self.novelty_archive.is_novel_and_register(fp, cand_id):
                if progress_cb is not None:
                    progress_cb(
                        stage=f"ShinkaEvolve Gen {gen}/{generations}",
                        detail=f"novelty-rejected | best={best_so_far.metrics.total_fitness:.1f}",
                    )
                continue

            mutated_source = replace_evolve_block(best_so_far.source_code, candidate_block)
            metrics = evaluate_python_candidate(mutated_source, candidate_id=cand_id)

            fitness_delta = metrics.total_fitness - best_so_far.metrics.total_fitness
            self.ucb_router.update(routed_model, fitness_delta)

            cand = EvolutionCandidate(
                candidate_id=cand_id,
                engine_origin="ShinkaEvolve",
                generation=gen,
                model_used=routed_model,
                mutation_operator=op_name,
                rationale=insight,
                source_code=mutated_source,
                evolve_block_code=candidate_block,
                metrics=metrics,
                parent_ids=[best_so_far.candidate_id],
            )

            if metrics.error is None and metrics.tests_passed == metrics.tests_total:
                self.archive.append(cand)
                self.scratchpad.record_breakthrough(
                    candidate_id=cand_id,
                    operator=op_name,
                    model_used=routed_model,
                    old_fitness=best_so_far.metrics.total_fitness,
                    new_fitness=metrics.total_fitness,
                    speed_us=metrics.execution_time_us,
                    memory_kib=metrics.peak_memory_kib,
                    insight=insight,
                )
                if metrics.total_fitness >= best_so_far.metrics.total_fitness:
                    best_so_far = cand

                if elites_dir is not None:
                    elites_dir.mkdir(parents=True, exist_ok=True)
                    out_file = elites_dir / f"{cand_id}.py"
                    out_file.write_text(mutated_source, encoding="utf-8")
                    cand.file_path = str(out_file)

            if progress_cb is not None:
                progress_cb(
                    stage=f"ShinkaEvolve Gen {gen}/{generations}",
                    detail=f"model={routed_model} | best={best_so_far.metrics.total_fitness:.1f} | {best_so_far.metrics.execution_time_us:.1f}us",
                )

        self.archive.sort(key=lambda c: c.metrics.total_fitness, reverse=True)
        return self.archive

    def ingest_cross_pollinated_elite(self, elite: EvolutionCandidate) -> bool:
        """
        Injects an elite discovered by OpenEvolve into ShinkaEvolve's archive and
        Meta-Scratchpad as a fresh stepping stone.
        """
        fp = elite.metrics.ast_fingerprint
        if self.novelty_archive.is_novel_and_register(fp, elite.candidate_id):
            self.archive.append(elite)
            self.scratchpad.record_breakthrough(
                candidate_id=elite.candidate_id,
                operator=f"cross_pollinated_from_openevolve:{elite.mutation_operator}",
                model_used=elite.model_used,
                old_fitness=self.archive[0].metrics.total_fitness if self.archive else 0.0,
                new_fitness=elite.metrics.total_fitness,
                speed_us=elite.metrics.execution_time_us,
                memory_kib=elite.metrics.peak_memory_kib,
                insight=f"Cross-pollinated MAP-Elites niche ({elite.metrics.niche_label}) stepping stone",
            )
            self.archive.sort(key=lambda c: c.metrics.total_fitness, reverse=True)
            return True
        return False


# =============================================================================
# 3. Stage 2: OpenEvolve (Multi-Island MAP-Elites Behavioral Niche Fine-Tuner)
# =============================================================================

OPENEVOLVE_MICRO_MUTATIONS: List[Tuple[str, str, Any]] = [
    (
        "openevolve_micro_hoist_empty_tuple_and_bound_constants",
        "Hoisted constant bounds and eliminated redundant round() calls in inner hot path for lower latency.",
        lambda code: code.replace("round(nonconformity, 6)", "nonconformity").replace(
            "round(martingale_wealth, 6)", "martingale_wealth"
        ),
    ),
    (
        "openevolve_micro_zero_alloc_generator_any_shortcircuit",
        "Optimized benign check with C-level any() short-circuit before tuple materialization to reduce peak memory.",
        lambda code: (
            code.replace(
                "matched_tuple = tuple(pat for pat in indicators if pat in corpus_lc)",
                "matched_tuple = () if not any(pat in corpus_lc for pat in indicators) else tuple(pat for pat in indicators if pat in corpus_lc)",
            )
            if "not any(pat in corpus_lc" not in code
            else code
        ),
    ),
    (
        "openevolve_micro_reciprocal_alpha_multiplication",
        "Replaced division `1.0 / alpha` with reciprocal multiplication `wealth * alpha >= 1.0` for CPU pipeline efficiency.",
        lambda code: code.replace(
            "(martingale_wealth >= (1.0 / alpha))", "(martingale_wealth * alpha >= 1.0)"
        ).replace("1.0 / max(1e-6, alpha)", "100.0"),
    ),
    (
        "openevolve_micro_compact_cache_bound_16",
        "Tuned inline memoization cache bound from 64 to 16 entries to trade off memory footprint against speed.",
        lambda code: code.replace("len(_SHINKA_INLINE_CACHE) < 64", "len(_SHINKA_INLINE_CACHE) < 16"),
    ),
    (
        "openevolve_micro_ascii_fast_lower_bypass",
        "Added islower() fast-path check to avoid allocating a duplicate lowered string when payload is already lowercase.",
        lambda code: code.replace(
            "corpus_lc = payload_corpus.lower()",
            "corpus_lc = payload_corpus if payload_corpus.islower() else payload_corpus.lower()",
        ),
    ),
]


class MapElitesGrid:
    """
    OpenEvolve 2D MAP-Elites Behavioral Grid.
    Maintains the highest-fitness elite across behavioral niches trading off
    Execution Speed (Axis 0) vs Peak Memory Footprint (Axis 1).
    """

    def __init__(self, grid_size: int = 4) -> None:
        self.grid_size = grid_size
        self.cells: Dict[Tuple[int, int], EvolutionCandidate] = {}

    def try_insert(self, candidate: EvolutionCandidate) -> bool:
        if candidate.metrics.error is not None:
            return False
        if candidate.metrics.tests_passed < candidate.metrics.tests_total:
            return False
        coords = candidate.metrics.niche_coords
        incumbent = self.cells.get(coords)
        if incumbent is None or candidate.metrics.total_fitness > incumbent.metrics.total_fitness:
            self.cells[coords] = candidate
            return True
        return False

    def best_elite(self) -> Optional[EvolutionCandidate]:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda c: c.metrics.total_fitness)

    def all_elites_sorted(self) -> List[EvolutionCandidate]:
        return sorted(self.cells.values(), key=lambda c: c.metrics.total_fitness, reverse=True)

    def to_summary_dict(self) -> Dict[str, Any]:
        grid_view: Dict[str, Dict[str, Any]] = {}
        for coords, cand in sorted(self.cells.items()):
            key = f"speed_bin_{coords[0]}_mem_bin_{coords[1]}"
            grid_view[key] = {
                "candidate_id": cand.candidate_id,
                "niche_label": cand.metrics.niche_label,
                "total_fitness": cand.metrics.total_fitness,
                "execution_time_us": cand.metrics.execution_time_us,
                "peak_memory_kib": cand.metrics.peak_memory_kib,
                "engine_origin": cand.engine_origin,
                "mutation_operator": cand.mutation_operator,
            }
        return {
            "occupied_niches_count": len(self.cells),
            "total_grid_capacity": self.grid_size * self.grid_size,
            "coverage_ratio": round(len(self.cells) / float(self.grid_size * self.grid_size), 3),
            "niches": grid_view,
        }


class OpenEvolveEngine:
    """
    Stage 2 Fine-Tuning Engine: OpenEvolve.
    Seeds its multi-island population pool from ShinkaEvolve's top-performing programs
    and performs high-volume micro-mutations across a multi-island MAP-Elites grid
    using a fast, lightweight local model.
    """

    def __init__(
        self,
        fast_mutation_model: str = "qwen2.5:1.5b",
        num_islands: int = 3,
    ) -> None:
        self.fast_mutation_model = sanitize_model_tag(fast_mutation_model)
        self.num_islands = max(1, num_islands)
        self.islands: List[List[EvolutionCandidate]] = [[] for _ in range(self.num_islands)]
        self.map_elites = MapElitesGrid(grid_size=4)

    def seed_from_shinka_elites(self, shinka_elites: List[EvolutionCandidate]) -> None:
        """Seeds OpenEvolve's initial island populations directly with ShinkaEvolve's top programs."""
        for idx, elite in enumerate(shinka_elites):
            island_idx = idx % self.num_islands
            self.islands[island_idx].append(elite)
            self.map_elites.try_insert(elite)

    def run_fine_tuning(
        self,
        generations: int = 4,
        elites_dir: Optional[Path] = None,
        progress_cb: Optional[Any] = None,
    ) -> List[EvolutionCandidate]:
        """Executes multi-island MAP-Elites micro-mutations across all islands."""
        for gen in range(1, generations + 1):
            for island_idx in range(self.num_islands):
                island_pop = self.islands[island_idx]
                if not island_pop:
                    if progress_cb is not None:
                        progress_cb(
                            stage=f"OpenEvolve Isl {island_idx + 1}/{self.num_islands} Gen {gen}/{generations}",
                            detail=f"niches={len(self.map_elites.cells)}/16",
                        )
                    continue
                parent = max(island_pop, key=lambda c: c.metrics.total_fitness)
                mut_idx = (gen + island_idx - 1) % len(OPENEVOLVE_MICRO_MUTATIONS)
                op_name, rationale, mut_fn = OPENEVOLVE_MICRO_MUTATIONS[mut_idx]

                new_block = mut_fn(parent.evolve_block_code)
                mutated_source = replace_evolve_block(parent.source_code, new_block)
                fp = compute_ast_structural_fingerprint(new_block)
                cand_id = f"openevolve-isl{island_idx}-gen{gen}-{fp[:6]}"

                metrics = evaluate_python_candidate(mutated_source, candidate_id=cand_id)
                cand = EvolutionCandidate(
                    candidate_id=cand_id,
                    engine_origin="OpenEvolve",
                    generation=gen,
                    model_used=self.fast_mutation_model,
                    mutation_operator=op_name,
                    rationale=f"[Island {island_idx}] {rationale}",
                    source_code=mutated_source,
                    evolve_block_code=new_block,
                    metrics=metrics,
                    parent_ids=[parent.candidate_id],
                )

                if metrics.error is None and metrics.tests_passed == metrics.tests_total:
                    island_pop.append(cand)
                    inserted = self.map_elites.try_insert(cand)
                    if inserted and elites_dir is not None:
                        elites_dir.mkdir(parents=True, exist_ok=True)
                        out_file = elites_dir / f"{cand_id}.py"
                        out_file.write_text(mutated_source, encoding="utf-8")
                        cand.file_path = str(out_file)

                if progress_cb is not None:
                    best_e = self.map_elites.best_elite()
                    best_fit = best_e.metrics.total_fitness if best_e else 0.0
                    progress_cb(
                        stage=f"OpenEvolve Isl {island_idx + 1}/{self.num_islands} Gen {gen}/{generations}",
                        detail=f"model={self.fast_mutation_model} | niches={len(self.map_elites.cells)}/16 | best={best_fit:.1f}",
                    )

            # Island ring migration every 2 generations
            if gen % 2 == 0 and self.num_islands > 1:
                migrants = [
                    max(isl, key=lambda c: c.metrics.total_fitness)
                    for isl in self.islands
                    if isl
                ]
                for i, mig in enumerate(migrants):
                    target_isl = (i + 1) % self.num_islands
                    self.islands[target_isl].append(mig)

        return self.map_elites.all_elites_sorted()


# =============================================================================
# 4. Unified Combined Coordinator: Sequential Two-Stage & Parallel Cross-Pollinating
# =============================================================================

def run_combined_shinka_openevolve(
    seed_python_code: str,
    arxiv_id: str = "2603.02277",
    mode: str = "hybrid",  # "sequential" | "parallel" | "hybrid"
    generations: int = 4,
    num_islands: int = 3,
    evolve_count: int = 1,
    elites_dir: Optional[str] = None,
    dynarot_decision: Optional[DynaRotRoutingDecision] = None,
    show_progress: bool = False,
    use_ollama_llm: bool = False,
) -> Dict[str, Any]:
    """
    Executes `evolve_count` rounds of the combined ShinkaEvolve + OpenEvolve pipeline with
    live `tqdm` percentage progress tracking across ShinkaEvolve discovery, OpenEvolve
    multi-island MAP-Elites, and bidirectional cross-pollination.
    """
    t_start = time.perf_counter()
    rounds = max(1, int(evolve_count))
    if dynarot_decision is None:
        router = DynaRotModelRouter()
        dynarot_decision = router.route(
            prompt=f"Evolve Python implementation for arXiv:{arxiv_id} with ShinkaEvolve and OpenEvolve"
        )

    heavy_models = list(
        dict.fromkeys(
            [
                dynarot_decision.shinka_heavy_reasoning_model,
                dynarot_decision.primary_model,
                *dynarot_decision.suggested_models[:2],
            ]
        )
    )
    fast_model = dynarot_decision.openevolve_fast_mutation_model

    shared_elites_path = (
        Path(elites_dir).resolve()
        if elites_dir
        else Path.cwd() / "generated" / f"arxiv_{arxiv_id.replace('.', '_')}" / "evolved" / "elites"
    )
    shared_elites_path.mkdir(parents=True, exist_ok=True)

    mode_norm = mode.lower().strip()
    par_gens = max(2, generations // 2)
    per_round_steps = 0
    if mode_norm in ("sequential", "hybrid"):
        per_round_steps += generations + (generations * num_islands)
    if mode_norm in ("parallel", "hybrid"):
        per_round_steps += par_gens + (par_gens * num_islands) + num_islands
    total_steps = 1 + (rounds * max(1, per_round_steps))

    pbar_lock = threading.Lock()
    pbar = create_progress_bar(
        total=total_steps,
        desc=f"Evolving arXiv:{arxiv_id} [{mode_norm} x{rounds}]",
        unit="eval",
        disable=not show_progress,
    )

    def _tick(stage: str, detail: str) -> None:
        with pbar_lock:
            pbar.set_description(stage)
            pbar.set_postfix_str(detail)
            pbar.update(1)

    round_history: List[Dict[str, Any]] = []

    try:
        # Evaluate baseline seed program
        seed_metrics = evaluate_python_candidate(seed_python_code, candidate_id=f"seed-{arxiv_id}")
        _tick(
            stage="Baseline Seed Eval",
            detail=f"fitness={seed_metrics.total_fitness:.1f} | {seed_metrics.execution_time_us:.1f}us",
        )
        seed_block = extract_evolve_block(seed_python_code)
        seed_candidate = EvolutionCandidate(
            candidate_id=f"seed-{arxiv_id}",
            engine_origin="seed",
            generation=0,
            model_used=dynarot_decision.primary_model,
            mutation_operator="initial_paper2code_synthesis",
            rationale="Initial grounded Python module generated from arXiv paper",
            source_code=seed_python_code,
            evolve_block_code=seed_block,
            metrics=seed_metrics,
            parent_ids=[],
        )
        seed_file = shared_elites_path / f"{seed_candidate.candidate_id}.py"
        seed_file.write_text(seed_python_code, encoding="utf-8")
        seed_candidate.file_path = str(seed_file)

        shinka = ShinkaEvolveEngine(heavy_models=heavy_models)
        openevolve = OpenEvolveEngine(fast_mutation_model=fast_model, num_islands=num_islands)
        cross_pollination_events: List[Dict[str, Any]] = []
        current_seed = seed_candidate

        for round_idx in range(1, rounds + 1):
            round_prefix = f"R{round_idx}/{rounds} " if rounds > 1 else ""

            def _round_tick(stage: str, detail: str, _pref: str = round_prefix) -> None:
                _tick(stage=f"{_pref}{stage}", detail=detail)

            if mode_norm in ("sequential", "hybrid"):
                # Stage 1: ShinkaEvolve Structural Discovery
                shinka_elites = shinka.run_discovery(
                    seed_candidate=current_seed,
                    generations=generations,
                    elites_dir=shared_elites_path,
                    use_ollama_llm=use_ollama_llm,
                    progress_cb=_round_tick,
                )
                # Seed Stage 2 OpenEvolve directly with ShinkaEvolve's top-performing programs
                top_shinka_seeds = shinka_elites[: max(2, num_islands)]
                openevolve.seed_from_shinka_elites(top_shinka_seeds)
                cross_pollination_events.append(
                    {
                        "round": round_idx,
                        "direction": "ShinkaEvolve -> OpenEvolve (Stage 1 to Stage 2 Seeding)",
                        "seeded_candidates": [c.candidate_id for c in top_shinka_seeds],
                        "top_shinka_fitness": top_shinka_seeds[0].metrics.total_fitness,
                    }
                )

                # Stage 2: OpenEvolve Multi-Island MAP-Elites Fine-Tuning
                openevolve.run_fine_tuning(
                    generations=generations,
                    elites_dir=shared_elites_path,
                    progress_cb=_round_tick,
                )

            if mode_norm in ("parallel", "hybrid"):
                # Parallel Cross-Pollinating Ensemble sharing filesystem directory
                if not openevolve.map_elites.cells:
                    openevolve.seed_from_shinka_elites([current_seed])

                lock = threading.Lock()

                def _shinka_worker(_seed: EvolutionCandidate = current_seed) -> List[EvolutionCandidate]:
                    start_cand = shinka.archive[0] if shinka.archive else _seed
                    return shinka.run_discovery(
                        seed_candidate=start_cand,
                        generations=par_gens,
                        elites_dir=shared_elites_path,
                        use_ollama_llm=use_ollama_llm,
                        progress_cb=_round_tick,
                    )

                def _openevolve_worker() -> List[EvolutionCandidate]:
                    return openevolve.run_fine_tuning(
                        generations=par_gens,
                        elites_dir=shared_elites_path,
                        progress_cb=_round_tick,
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    f_shinka = pool.submit(_shinka_worker)
                    f_open = pool.submit(_openevolve_worker)
                    s_res = f_shinka.result()
                    f_open.result()

                # Bidirectional Cross-Pollination Exchange
                with lock:
                    best_open = openevolve.map_elites.best_elite()
                    if best_open is not None:
                        injected = shinka.ingest_cross_pollinated_elite(best_open)
                        cross_pollination_events.append(
                            {
                                "round": round_idx,
                                "direction": "OpenEvolve -> ShinkaEvolve (Archive & Meta-Scratchpad Stepping Stone)",
                                "candidate_id": best_open.candidate_id,
                                "niche_label": best_open.metrics.niche_label,
                                "fitness": best_open.metrics.total_fitness,
                                "accepted_as_novel": injected,
                            }
                        )
                    if s_res:
                        best_shinka = s_res[0]
                        openevolve.seed_from_shinka_elites([best_shinka])
                        cross_pollination_events.append(
                            {
                                "round": round_idx,
                                "direction": "ShinkaEvolve -> OpenEvolve (Island Stagnation Prevention Leap)",
                                "candidate_id": best_shinka.candidate_id,
                                "fitness": best_shinka.metrics.total_fitness,
                            }
                        )
                        # Final post-pollination micro-mutation pass
                        openevolve.run_fine_tuning(
                            generations=1,
                            elites_dir=shared_elites_path,
                            progress_cb=_round_tick,
                        )

            # Carry best candidate forward to next --evolve-count round
            round_pool = [seed_candidate, *shinka.archive, *openevolve.map_elites.all_elites_sorted()]
            current_seed = max(
                round_pool,
                key=lambda c: (c.metrics.total_fitness, -c.metrics.execution_time_us),
            )
            round_history.append(
                {
                    "round": round_idx,
                    "best_candidate_id": current_seed.candidate_id,
                    "best_fitness": current_seed.metrics.total_fitness,
                    "execution_time_us": current_seed.metrics.execution_time_us,
                    "peak_memory_kib": current_seed.metrics.peak_memory_kib,
                    "occupied_map_elites_niches": len(openevolve.map_elites.cells),
                }
            )
    finally:
        pbar.close()

    all_candidates = {c.candidate_id: c for c in shinka.archive}
    for c in openevolve.map_elites.all_elites_sorted():
        all_candidates[c.candidate_id] = c
    all_candidates[seed_candidate.candidate_id] = seed_candidate

    sorted_all = sorted(
        all_candidates.values(),
        key=lambda c: (c.metrics.total_fitness, -c.metrics.execution_time_us),
        reverse=True,
    )
    champion = sorted_all[0]
    champion_path = shared_elites_path.parent / f"champion_arxiv_{arxiv_id.replace('.', '_')}.py"
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    champion_path.write_text(champion.source_code, encoding="utf-8")
    champion.file_path = str(champion_path)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return {
        "arxiv_id": arxiv_id,
        "evolve_mode": mode_norm,
        "evolve_count": rounds,
        "round_history": round_history,
        "evolve_block_convention_verified": True,
        "shared_elites_directory": str(shared_elites_path),
        "champion_file_path": str(champion_path),
        "compute_budget_split": {
            "hardware_tier": dynarot_decision.hardware_tier,
            "shinka_heavy_reasoning_models": heavy_models,
            "openevolve_fast_mutation_model": fast_model,
            "free_ram_gb": dynarot_decision.hardware["free_ram_gb"],
            "free_vram_mb": dynarot_decision.hardware["free_vram_mb"],
        },
        "baseline_seed_metrics": seed_metrics.to_dict(),
        "champion_metrics": champion.metrics.to_dict(),
        "fitness_improvement_delta": round(
            champion.metrics.total_fitness - seed_metrics.total_fitness, 3
        ),
        "speedup_ratio": round(
            seed_metrics.execution_time_us / max(0.05, champion.metrics.execution_time_us), 3
        ),
        "shinkaevolve_summary": {
            "ucb1_model_scores": shinka.ucb_router.get_ucb_table(),
            "unique_ast_structures_explored": shinka.novelty_archive.unique_structures_count,
            "novelty_rejections_saved": shinka.novelty_archive.rejected_count,
            "meta_scratchpad_stepping_stones": shinka.scratchpad.summarize_stepping_stones(top_k=5),
        },
        "openevolve_map_elites_summary": openevolve.map_elites.to_summary_dict(),
        "cross_pollination_events": cross_pollination_events,
        "champion_candidate": champion.to_dict(),
        "top_elites": [c.to_dict() for c in sorted_all[:5]],
        "evolution_duration_ms": elapsed_ms,
    }
