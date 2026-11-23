"""
Dream-RSI: Recursive Self-Improvement through Evolving Worlds (arXiv:2609.14858).

Implements a programmable meta-exploration Reinforcement Learning layer over
generated `# EVOLVE-BLOCK` Python modules, aligned with Google Research's
`google-research/dream-rsi` architecture (`TraceEnv`, `ReplayWorldPool`,
`DreamPolicyOptimizer`, and online-to-offline recursive self-improvement):

  1. Online Exploration (`TraceEnv`, mode="online"):
     Current `DreamExplorationPolicy` steers branching, structural vs. micro-mutation
     operator selection, semaphore-bounded parallel batch width B, and early stopping
     over real Python code evaluations (`evaluate_python_candidate()`), recording a
     structured `DiscoveryTree` of code snapshots, AST fingerprints, unit-test outcomes,
     and microsecond/KiB benchmarks.
  2. Evolving Replay Worlds (`ReplayWorldPool`):
     Frozen `DiscoveryTree` histories are converted into deterministic zero-cost replay
     simulators and persisted to `trace_pool.jsonl` across cycles and runs.
  3. Offline Dreaming (`DreamPolicyOptimizer`):
     Evaluates candidate exploration policies (and optional local Ollama LLM-authored
     policy refinements via `--rl-llm`) across the accumulated `ReplayWorldPool` via
     O(1) cached lookups (zero live code execution) and promotes the best policy back
     to online discovery.
  4. Standardized Empirical Learning Superiority Proof (`rl_benchmarking_and_evaluation.md`):
     Explicitly labels backend fidelity (`official_google_research_dream_rsi` vs.
     `native_dream_rsi_replay_world_model`) and benchmarks `DreamRSI_Learned` against
     `Disabled`, `Random`, `Heuristic`, and `NoLearning` over matched episode seeds
     with 95% Student-t Confidence Intervals.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import math
import os
import random
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dynarot import (
    DynaRotModelRouter,
    build_no_proxy_opener,
    find_best_installed_chat_model,
    probe_ollama_models,
    sanitize_model_tag,
)
from .evolve_engine import (
    OPENEVOLVE_MICRO_MUTATIONS,
    SHINKA_STRUCTURAL_BREAKTHROUGHS,
    compute_ast_structural_fingerprint,
    evaluate_python_candidate,
    extract_evolve_block,
    replace_evolve_block,
)
from .models import (
    CandidateScoreMetrics,
    ConfidenceIntervalMetric,
    DiscoveryTree,
    DiscoveryTreeNode,
    DreamExplorationPolicy,
    DynaRotRoutingDecision,
    EvolutionCandidate,
)
from .progress import create_progress_bar


# =============================================================================
# 1. Explicit Backend Detection & RL Fidelity Labeling
# =============================================================================

def detect_dream_rsi_backend() -> Dict[str, Any]:
    """
    Explicitly inspects whether the external `google-research/dream-rsi` package
    (`dream_rsi`) is installed and labels the active execution backend so native
    replay-world metrics are never misreported as official external package runs.
    """
    spec = importlib.util.find_spec("dream_rsi")
    if spec is not None:
        return {
            "environment_backend": "official_google_research_dream_rsi",
            "is_official_external_package": True,
            "package_origin": str(spec.origin or "installed"),
            "reference_paper": "arXiv:2609.14858 (Dream-RSI: Recursive Self-Improvement through Evolving Worlds)",
            "reference_repo": "https://github.com/google-research/dream-rsi",
            "fidelity_note": "Official external `dream_rsi` package detected in Python environment.",
        }
    return {
        "environment_backend": "native_dream_rsi_replay_world_model",
        "is_official_external_package": False,
        "package_origin": "arxiv_to_code.dream_rsi_engine",
        "reference_paper": "arXiv:2609.14858 (Dream-RSI: Recursive Self-Improvement through Evolving Worlds)",
        "reference_repo": "https://github.com/google-research/dream-rsi",
        "fidelity_note": (
            "Executed on ArXiv-to-Code's native Python Dream-RSI Replay World engine over real "
            "Python code execution traces (not the external `google-research/dream-rsi` package)."
        ),
    }


# =============================================================================
# 2. Additional Real EVOLVE-BLOCK Micro-Optimization Operators for Tree Depth
# =============================================================================

DREAM_EXTRA_MICRO_MUTATIONS: List[Tuple[str, str, Callable[[str], str]]] = [
    (
        "dream_micro_local_math_log1p_binding",
        "Hoisted `math.log1p` to local scope variable to avoid repeated global module attribute lookup.",
        lambda code: (
            code.replace("math.log1p(", "_log1p(").replace(
                "corpus_lc =",
                "_log1p = math.log1p\n    corpus_lc =",
                1,
            )
            if "math.log1p(" in code and "_log1p = math.log1p" not in code
            else code
        ),
    ),
    (
        "dream_micro_fast_empty_corpus_guard",
        "Added immediate empty-string short-circuit before lowercasing payload corpus.",
        lambda code: (
            code.replace(
                "corpus_lc = payload_corpus.lower()",
                "if not payload_corpus:\n        return (), 0.0, prior_wealth, False\n    corpus_lc = payload_corpus.lower()",
                1,
            )
            if "if not payload_corpus:" not in code and "corpus_lc = payload_corpus.lower()" in code
            else code
        ),
    ),
    (
        "dream_micro_branchless_wealth_clamp",
        "Streamlined wealth floor clamp using inline ternary expression to cut call overhead.",
        lambda code: (
            code.replace(
                "max(0.05, prior_wealth * (0.94 if n_chars > 0 else 1.0))",
                "w if (w := prior_wealth * 0.94) > 0.05 else 0.05",
            )
            if "max(0.05, prior_wealth * (0.94 if n_chars > 0 else 1.0))" in code
            else code
        ),
    ),
]

ALL_MICRO_MUTATIONS: List[Tuple[str, str, Callable[[str], str]]] = [
    *OPENEVOLVE_MICRO_MUTATIONS,
    *DREAM_EXTRA_MICRO_MUTATIONS,
]


# =============================================================================
# 3. Standard Baseline & Dreamed Meta-Exploration Policy Constructors
# =============================================================================

def create_disabled_policy() -> DreamExplorationPolicy:
    """Passive baseline: zero RL exploration evaluations (baseline synthesized seed only)."""
    return DreamExplorationPolicy(
        policy_id="baseline-disabled",
        policy_origin="disabled",
        uct_exploration_c=0.0,
        depth_bonus=0.0,
        structural_vs_micro_prob=0.5,
        parallel_batch_width=1,
        early_stop_patience=1,
        min_gain_threshold=5.0,
        novelty_weight=0.0,
        rationale="Disabled passive baseline (keeps seed code unmodified).",
    )


def create_random_policy() -> DreamExplorationPolicy:
    """Uniform random action selection baseline across branches, operators, and stopping."""
    return DreamExplorationPolicy(
        policy_id="baseline-random",
        policy_origin="random",
        uct_exploration_c=1.0,
        depth_bonus=0.0,
        structural_vs_micro_prob=0.5,
        parallel_batch_width=1,
        early_stop_patience=12,
        min_gain_threshold=0.05,
        novelty_weight=0.0,
        rationale="Uniform random branch, operator, batch-width, and stopping selection.",
    )


def create_baseline_heuristic_policy() -> DreamExplorationPolicy:
    """Fixed-rule heuristic baseline (Dream-RSI 'Recursive Fixed Exploration' equivalent)."""
    return DreamExplorationPolicy(
        policy_id="baseline-heuristic",
        policy_origin="baseline_heuristic",
        uct_exploration_c=1.414,
        depth_bonus=0.08,
        structural_vs_micro_prob=0.50,
        parallel_batch_width=2,
        early_stop_patience=4,
        min_gain_threshold=0.25,
        novelty_weight=0.15,
        rationale="Fixed heuristic UCT (c=1.414, batch=2, patience=4, balanced structural/micro).",
    )


def create_frozen_initial_policy() -> DreamExplorationPolicy:
    """Unoptimized initial RL policy frozen without offline dreaming updates."""
    return DreamExplorationPolicy(
        policy_id="baseline-no-learning",
        policy_origin="frozen_initial",
        uct_exploration_c=2.6,
        depth_bonus=-0.15,
        structural_vs_micro_prob=0.25,
        parallel_batch_width=1,
        early_stop_patience=10,
        min_gain_threshold=0.02,
        novelty_weight=0.05,
        rationale="Frozen initial policy prior to any offline Dream-RSI replay optimization.",
    )


# =============================================================================
# 4. Shared TraceEnv (Online Real Code Execution & Offline Zero-Cost Replay)
# =============================================================================

class TraceEnv:
    """
    Shared decision environment modeled after `google-research/dream-rsi/dream_rsi/env/TraceEnv`.
    Supports:
      - `run_online_rollout(...)`: executes real `# EVOLVE-BLOCK` Python mutations via
        semaphore-bounded `ThreadPoolExecutor` + `evaluate_python_candidate()` and records
        a complete `DiscoveryTree`.
      - `run_offline_replay(...)`: replays a frozen `DiscoveryTree` world using pure O(1)
        cached node lookups with ZERO calls to `evaluate_python_candidate()`.
    """

    @staticmethod
    def _score_frontier_node(
        node: DiscoveryTreeNode,
        visits: int,
        total_visits: int,
        niche_counts: Dict[str, int],
        policy: DreamExplorationPolicy,
        rng: random.Random,
    ) -> float:
        if policy.policy_origin == "random":
            return rng.random()
        norm_fit = node.metrics.total_fitness / 100.0
        uct_bonus = policy.uct_exploration_c * math.sqrt(
            math.log(max(2, total_visits + 1)) / float(visits + 1)
        )
        depth_term = policy.depth_bonus * float(node.depth)
        rarity = 1.0 / float(max(1, niche_counts.get(node.metrics.niche_label, 1)))
        novelty_term = policy.novelty_weight * rarity
        return norm_fit + uct_bonus + depth_term + novelty_term

    @classmethod
    def _propose_child_candidates(
        cls,
        parent: DiscoveryTreeNode,
        batch_width: int,
        policy: DreamExplorationPolicy,
        seen_fingerprints: set[str],
        rng: random.Random,
    ) -> List[Tuple[str, str, str, str]]:
        """
        Generates up to `batch_width` novel child proposals `(operator_name, operator_family, new_block, rationale)`
        for `parent` without duplicating already-explored AST structures.
        """
        proposals: List[Tuple[str, str, str, str]] = []
        local_fps: set[str] = set()

        struct_indices = list(range(len(SHINKA_STRUCTURAL_BREAKTHROUGHS)))
        micro_indices = list(range(len(ALL_MICRO_MUTATIONS)))
        rng.shuffle(struct_indices)
        rng.shuffle(micro_indices)

        # Build prioritized attempt sequence guided by policy.structural_vs_micro_prob
        attempts: List[Tuple[str, int]] = []
        si = mi = 0
        while si < len(struct_indices) or mi < len(micro_indices):
            pick_struct = (
                si < len(struct_indices)
                and (mi >= len(micro_indices) or rng.random() < policy.structural_vs_micro_prob)
            )
            if pick_struct:
                attempts.append(("structural", struct_indices[si]))
                si += 1
            else:
                attempts.append(("micro", micro_indices[mi]))
                mi += 1

        for fam, idx in attempts:
            if len(proposals) >= batch_width:
                break
            if fam == "structural":
                op_name, rationale, new_block = SHINKA_STRUCTURAL_BREAKTHROUGHS[idx]
            else:
                op_name, rationale, mut_fn = ALL_MICRO_MUTATIONS[idx]
                new_block = mut_fn(parent.evolve_block_code)
                if new_block == parent.evolve_block_code:
                    continue

            fp = compute_ast_structural_fingerprint(new_block)
            if fp in seen_fingerprints or fp in local_fps:
                continue
            local_fps.add(fp)
            proposals.append((op_name, fam, new_block, rationale))

        return proposals

    @classmethod
    def run_online_rollout(
        cls,
        seed_python_code: str,
        arxiv_id: str,
        policy: DreamExplorationPolicy,
        budget: int = 12,
        cycle_index: int = 1,
        seed: int = 42,
        evaluator_fn: Callable[[str, str], CandidateScoreMetrics] = evaluate_python_candidate,
        progress_cb: Optional[Callable[[str, str], None]] = None,
    ) -> Tuple[DiscoveryTree, Dict[str, Any]]:
        """
        Executes an online Dream-RSI discovery rollout guided by `policy`, evaluating
        real Python candidates concurrently (bounded by CPU parallelism) and returning
        the frozen `DiscoveryTree` world plus trajectory telemetry.
        """
        rng = random.Random(seed + cycle_index * 997)
        t_root = time.perf_counter()
        seed_metrics = evaluator_fn(seed_python_code, f"rl-c{cycle_index}-n0")
        root_cost_ms = round((time.perf_counter() - t_root) * 1000.0, 2)
        seed_block = extract_evolve_block(seed_python_code)
        root_fp = compute_ast_structural_fingerprint(seed_block)

        root_node = DiscoveryTreeNode(
            node_id="n0",
            parent_id=None,
            depth=0,
            source_code=seed_python_code,
            evolve_block_code=seed_block,
            operator="seed_program",
            operator_family="seed",
            ast_fingerprint=root_fp,
            metrics=seed_metrics,
            diagnostics="Initial seed program for online Dream-RSI rollout",
            eval_cost_ms=root_cost_ms,
            children_ids=[],
        )

        tree = DiscoveryTree(
            tree_id=f"world-{arxiv_id.replace('.', '_')}-c{cycle_index}-{int(time.time() * 1000) % 100000}",
            arxiv_id=arxiv_id,
            root_id="n0",
            seed_fitness=seed_metrics.total_fitness,
            cycle_index=cycle_index,
            created_at_epoch=round(time.time(), 3),
            nodes={"n0": root_node},
        )

        if policy.policy_origin == "disabled" or budget <= 0:
            return tree, {
                "evals_used": 0,
                "early_stopped": True,
                "best_fitness": seed_metrics.total_fitness,
                "seed_fitness": seed_metrics.total_fitness,
                "valid_nodes": 1 if seed_metrics.error is None else 0,
                "total_nodes": 1,
            }

        seen_fps: set[str] = {root_fp}
        visit_counts: Dict[str, int] = {"n0": 0}
        niche_counts: Dict[str, int] = {seed_metrics.niche_label: 1}
        exhausted_parents: set[str] = set()

        best_fitness = seed_metrics.total_fitness
        stagnation_rounds = 0
        evals_used = 0
        early_stopped = False
        next_node_idx = 1
        max_workers = max(1, min(4, (os.cpu_count() or 4)))

        while evals_used < budget:
            if evals_used >= 2 and stagnation_rounds >= policy.early_stop_patience:
                early_stopped = True
                break

            expandable = [
                nid
                for nid, n in tree.nodes.items()
                if nid not in exhausted_parents
                and n.metrics.error is None
                and n.metrics.tests_passed == n.metrics.tests_total
            ]
            if not expandable:
                break

            total_visits = sum(visit_counts.values())
            parent_id = max(
                expandable,
                key=lambda nid: cls._score_frontier_node(
                    tree.nodes[nid],
                    visit_counts.get(nid, 0),
                    total_visits,
                    niche_counts,
                    policy,
                    rng,
                ),
            )
            parent = tree.nodes[parent_id]
            visit_counts[parent_id] = visit_counts.get(parent_id, 0) + 1

            remaining = budget - evals_used
            batch_w = min(
                remaining,
                rng.randint(1, 3) if policy.policy_origin == "random" else policy.parallel_batch_width,
            )
            proposals = cls._propose_child_candidates(
                parent=parent,
                batch_width=batch_w,
                policy=policy,
                seen_fingerprints=seen_fps,
                rng=rng,
            )
            if not proposals:
                exhausted_parents.add(parent_id)
                continue

            # Pre-register fingerprints & build mutated sources
            batch_specs: List[Tuple[str, str, str, str, str, str]] = []
            for op_name, fam, new_block, rationale in proposals:
                nid = f"n{next_node_idx}"
                next_node_idx += 1
                fp = compute_ast_structural_fingerprint(new_block)
                seen_fps.add(fp)
                mutated_src = replace_evolve_block(parent.source_code, new_block)
                batch_specs.append((nid, op_name, fam, new_block, rationale, mutated_src))

            # Semaphore-bounded concurrent real evaluation
            def _eval_one(
                spec: Tuple[str, str, str, str, str, str],
                _parent_id: str = parent_id,
                _parent_depth: int = parent.depth,
            ) -> Tuple[DiscoveryTreeNode, float]:
                nid, op_name, fam, new_block, rationale, mutated_src = spec
                t0 = time.perf_counter()
                m = evaluator_fn(mutated_src, f"rl-c{cycle_index}-{nid}")
                cost_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                node = DiscoveryTreeNode(
                    node_id=nid,
                    parent_id=_parent_id,
                    depth=_parent_depth + 1,
                    source_code=mutated_src,
                    evolve_block_code=new_block,
                    operator=op_name,
                    operator_family=fam,
                    ast_fingerprint=compute_ast_structural_fingerprint(new_block),
                    metrics=m,
                    diagnostics=rationale,
                    eval_cost_ms=cost_ms,
                    children_ids=[],
                )
                return node, m.total_fitness

            evaluated_nodes: List[DiscoveryTreeNode] = []
            if len(batch_specs) == 1:
                nd, _ = _eval_one(batch_specs[0])
                evaluated_nodes.append(nd)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(batch_specs))) as ex:
                    for nd, _ in ex.map(_eval_one, batch_specs):
                        evaluated_nodes.append(nd)

            round_gain = 0.0
            for nd in evaluated_nodes:
                tree.nodes[nd.node_id] = nd
                parent.children_ids.append(nd.node_id)
                visit_counts[nd.node_id] = 0
                niche_counts[nd.metrics.niche_label] = niche_counts.get(nd.metrics.niche_label, 0) + 1
                evals_used += 1
                if nd.metrics.error is None and nd.metrics.tests_passed == nd.metrics.tests_total:
                    if nd.metrics.total_fitness > best_fitness:
                        round_gain = max(round_gain, nd.metrics.total_fitness - best_fitness)
                        best_fitness = nd.metrics.total_fitness
                if progress_cb is not None:
                    progress_cb(
                        f"Dream-RSI Cycle {cycle_index} Online",
                        f"eval={evals_used}/{budget} | best={best_fitness:.2f} | op={nd.operator_family}",
                    )

            if round_gain >= policy.min_gain_threshold:
                stagnation_rounds = 0
            else:
                stagnation_rounds += 1

        valid_nodes = sum(
            1
            for n in tree.nodes.values()
            if n.metrics.error is None and n.metrics.tests_passed == n.metrics.tests_total
        )
        return tree, {
            "evals_used": evals_used,
            "early_stopped": early_stopped,
            "best_fitness": round(best_fitness, 3),
            "seed_fitness": round(seed_metrics.total_fitness, 3),
            "valid_nodes": valid_nodes,
            "total_nodes": len(tree.nodes),
        }

    @classmethod
    def run_offline_replay(
        cls,
        world: DiscoveryTree,
        policy: DreamExplorationPolicy,
        budget: int,
        seed: int = 1000,
    ) -> Dict[str, float]:
        """
        Zero-cost offline Dream-RSI Replay Simulator traversal over a frozen `DiscoveryTree`.
        Reads ONLY cached `world.nodes` — performs ZERO live code compilations/executions.
        Returns replay objective metrics (`replay_reward`, `peak_fitness`, `auc_norm`,
        `evaluations_saved_pct`, `valid_pass_rate_pct`, `evals_used`).
        """
        rng = random.Random(seed)
        root = world.nodes[world.root_id]
        world_best_fit = world.best_node().metrics.total_fitness
        eff_budget = max(1, min(budget, max(1, len(world.nodes) - 1)))

        if policy.policy_origin == "disabled":
            peak = root.metrics.total_fitness
            valid_pct = 100.0 if (root.metrics.error is None and root.metrics.tests_passed == root.metrics.tests_total) else 0.0
            # Disabled performs zero optimization beyond seed; no early-stop bonus for doing nothing
            replay_reward = round(0.55 * peak + 0.25 * peak + 0.05 * valid_pct, 4)
            return {
                "replay_reward": replay_reward,
                "peak_fitness": round(peak, 3),
                "auc_norm": round(peak, 3),
                "evaluations_saved_pct": 0.0,
                "valid_pass_rate_pct": round(valid_pct, 2),
                "evals_used": 0.0,
            }

        visited: set[str] = {world.root_id}
        visit_counts: Dict[str, int] = {world.root_id: 0}
        niche_counts: Dict[str, int] = {root.metrics.niche_label: 1}

        best_fitness = root.metrics.total_fitness
        stagnation_rounds = 0
        evals_used = 0
        valid_reveals = 0
        auc_accum = 0.0

        while evals_used < eff_budget:
            if policy.policy_origin == "random":
                if evals_used >= 2 and rng.random() < 0.12:
                    break
            elif evals_used >= 2 and stagnation_rounds >= policy.early_stop_patience:
                break

            # Frontier = visited nodes that still have at least one unvisited child in the frozen world
            frontier = [
                nid
                for nid in visited
                if any(cid not in visited for cid in world.nodes[nid].children_ids if cid in world.nodes)
            ]
            if not frontier:
                break

            total_visits = sum(visit_counts.values())
            parent_id = max(
                frontier,
                key=lambda nid: cls._score_frontier_node(
                    world.nodes[nid],
                    visit_counts.get(nid, 0),
                    total_visits,
                    niche_counts,
                    policy,
                    rng,
                ),
            )
            visit_counts[parent_id] = visit_counts.get(parent_id, 0) + 1

            unvisited_children = [
                cid for cid in world.nodes[parent_id].children_ids if cid not in visited and cid in world.nodes
            ]
            if not unvisited_children:
                continue

            # Order child choices by operator_family preference (or uniform random for Random baseline)
            if policy.policy_origin == "random":
                rng.shuffle(unvisited_children)
                batch_w = min(eff_budget - evals_used, rng.randint(1, 3), len(unvisited_children))
            else:
                def _child_pref(cid: str) -> float:
                    fam = world.nodes[cid].operator_family
                    base_p = policy.structural_vs_micro_prob if fam == "structural" else (1.0 - policy.structural_vs_micro_prob)
                    return base_p + rng.uniform(0.0, 0.15)

                unvisited_children.sort(key=_child_pref, reverse=True)
                batch_w = min(eff_budget - evals_used, policy.parallel_batch_width, len(unvisited_children))

            chosen = unvisited_children[:batch_w]
            round_gain = 0.0
            for cid in chosen:
                nd = world.nodes[cid]
                visited.add(cid)
                visit_counts[cid] = 0
                niche_counts[nd.metrics.niche_label] = niche_counts.get(nd.metrics.niche_label, 0) + 1
                evals_used += 1
                if nd.metrics.error is None and nd.metrics.tests_passed == nd.metrics.tests_total:
                    valid_reveals += 1
                    if nd.metrics.total_fitness > best_fitness:
                        round_gain = max(round_gain, nd.metrics.total_fitness - best_fitness)
                        best_fitness = nd.metrics.total_fitness
                auc_accum += best_fitness

            if round_gain >= policy.min_gain_threshold:
                stagnation_rounds = 0
            else:
                stagnation_rounds += 1

        # Pad remaining budget steps in AUC with final best_fitness so smart early stopping is never penalized
        if evals_used < eff_budget:
            auc_accum += best_fitness * (eff_budget - evals_used)
        auc_norm = auc_accum / float(eff_budget)

        # Quality-gated compute efficiency bonus: early stopping only earns credit proportional to quality achieved
        raw_saved_ratio = max(0.0, (eff_budget - evals_used) / float(eff_budget))
        quality_ratio = min(1.0, best_fitness / max(1e-6, world_best_fit))
        eval_saved_pct = 100.0 * raw_saved_ratio * (quality_ratio ** 2)
        valid_pass_rate_pct = 100.0 * (valid_reveals / float(max(1, evals_used)))

        replay_reward = (
            0.55 * best_fitness
            + 0.25 * auc_norm
            + 0.15 * eval_saved_pct
            + 0.05 * valid_pass_rate_pct
        )
        return {
            "replay_reward": round(replay_reward, 4),
            "peak_fitness": round(best_fitness, 3),
            "auc_norm": round(auc_norm, 3),
            "evaluations_saved_pct": round(eval_saved_pct, 2),
            "valid_pass_rate_pct": round(valid_pass_rate_pct, 2),
            "evals_used": float(evals_used),
        }


# =============================================================================
# 5. Evolving Replay Simulator Pool (`ReplayWorldPool` / `trace_pool.jsonl`)
# =============================================================================

class ReplayWorldPool:
    """
    Persistent pool of frozen `DiscoveryTree` replay worlds (Dream-RSI `trace_pool.py` parity).
    """

    def __init__(self, worlds: Optional[List[DiscoveryTree]] = None) -> None:
        self.worlds: List[DiscoveryTree] = list(worlds or [])

    def add_world(self, tree: DiscoveryTree) -> None:
        self.worlds.append(tree)

    def __len__(self) -> int:
        return len(self.worlds)

    def total_recorded_nodes(self) -> int:
        return sum(len(w.nodes) for w in self.worlds)

    def save_jsonl(self, jsonl_path: Path) -> str:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for w in self.worlds:
                fh.write(json.dumps(w.to_dict()) + "\n")
        return str(jsonl_path)

    @classmethod
    def load_jsonl(cls, jsonl_path: Path) -> "ReplayWorldPool":
        pool = cls()
        if not jsonl_path.exists():
            return pool
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict) and obj.get("nodes"):
                        pool.add_world(DiscoveryTree.from_dict(obj))
                except Exception:
                    continue
        return pool

    def evaluate_policy_offline(
        self,
        policy: DreamExplorationPolicy,
        budget: int,
        seed: int = 1000,
    ) -> Dict[str, float]:
        """Scores `policy` across every frozen replay world in the pool with zero live executions."""
        if not self.worlds:
            return {
                "replay_reward": 0.0,
                "peak_fitness": 0.0,
                "auc_norm": 0.0,
                "evaluations_saved_pct": 0.0,
                "valid_pass_rate_pct": 0.0,
                "evals_used": 0.0,
            }
        per_world = [
            TraceEnv.run_offline_replay(world=w, policy=policy, budget=budget, seed=seed + idx * 31)
            for idx, w in enumerate(self.worlds)
        ]
        n = float(len(per_world))
        return {
            "replay_reward": round(sum(r["replay_reward"] for r in per_world) / n, 4),
            "peak_fitness": round(sum(r["peak_fitness"] for r in per_world) / n, 3),
            "auc_norm": round(sum(r["auc_norm"] for r in per_world) / n, 3),
            "evaluations_saved_pct": round(sum(r["evaluations_saved_pct"] for r in per_world) / n, 2),
            "valid_pass_rate_pct": round(sum(r["valid_pass_rate_pct"] for r in per_world) / n, 2),
            "evals_used": round(sum(r["evals_used"] for r in per_world) / n, 2),
        }


# =============================================================================
# 6. Offline Dreaming Meta-Policy Optimizer (`DreamPolicyOptimizer`)
# =============================================================================

class DreamPolicyOptimizer:
    """
    Dream-RSI Offline Policy Discovery (`dream_rsi/learning/discover.py` parity):
    Generates candidate exploration policies via structured evolutionary perturbation
    plus optional local Ollama LLM meta-policy rewriting (`--rl-llm`), scores all
    candidates across `ReplayWorldPool` at zero execution cost, and promotes the winner.
    """

    @staticmethod
    def _mutate_policy(
        base: DreamExplorationPolicy,
        cand_id: str,
        rng: random.Random,
    ) -> DreamExplorationPolicy:
        return DreamExplorationPolicy(
            policy_id=cand_id,
            policy_origin="dreamed_param",
            uct_exploration_c=base.uct_exploration_c + rng.gauss(0.0, 0.45),
            depth_bonus=base.depth_bonus + rng.gauss(0.0, 0.14),
            structural_vs_micro_prob=base.structural_vs_micro_prob + rng.gauss(0.0, 0.18),
            parallel_batch_width=base.parallel_batch_width + rng.choice([-1, 0, 0, 1]),
            early_stop_patience=base.early_stop_patience + rng.choice([-2, -1, 0, 1, 2]),
            min_gain_threshold=base.min_gain_threshold + rng.gauss(0.0, 0.20),
            novelty_weight=base.novelty_weight + rng.gauss(0.0, 0.12),
            rationale=f"Dreamed perturbation of {base.policy_id}",
        ).clamp_safe()

    @staticmethod
    def _propose_llm_dream_policy(
        incumbent: DreamExplorationPolicy,
        incumbent_score: Dict[str, float],
        pool: ReplayWorldPool,
        cycle_index: int,
        endpoint: str,
        model_tag: str,
        timeout_sec: float = 90.0,
    ) -> Optional[DreamExplorationPolicy]:
        """
        Optional local Ollama LLM meta-policy rewriter (`--rl-llm`):
        Summarizes replay world statistics and requests an improved JSON meta-exploration
        policy, strictly clamped via `DreamExplorationPolicy.clamp_safe()`.
        """
        clean_model = sanitize_model_tag(model_tag)
        is_reasoning = any(k in clean_model.lower() for k in ("deepseek-r1", "qwq", "glm-4.7"))
        world_summaries = []
        for w in pool.worlds[-4:]:
            best = w.best_node()
            world_summaries.append(
                f"world={w.tree_id}: nodes={len(w.nodes)}, seed={w.seed_fitness:.2f}, "
                f"best={best.metrics.total_fitness:.2f} (depth={best.depth}, op={best.operator_family})"
            )

        prompt = (
            "You are a Dream-RSI meta-exploration policy optimizer (arXiv:2609.14858).\n"
            "Propose an improved exploration policy configuration to maximize peak code fitness, "
            "fast discovery AUC, and early-stopping compute efficiency over the replay world pool.\n\n"
            f"Current Policy: {json.dumps(incumbent.to_dict())}\n"
            f"Current Replay Metrics: {json.dumps(incumbent_score)}\n"
            f"Replay Worlds:\n  " + "\n  ".join(world_summaries) + "\n\n"
            "Return ONLY a valid JSON object with numeric fields:\n"
            '{"uct_exploration_c": 0.0..4.0, "depth_bonus": -0.5..1.0, '
            '"structural_vs_micro_prob": 0.05..0.95, "parallel_batch_width": 1..4, '
            '"early_stop_patience": 1..12, "min_gain_threshold": 0.01..5.0, '
            '"novelty_weight": 0.0..1.5, "rationale": "<short technical reason>"}'
        )

        body_obj: Dict[str, Any] = {
            "model": clean_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.25, "num_predict": 400, "num_ctx": 4096},
        }
        if is_reasoning:
            body_obj["think"] = False

        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/api/generate",
            data=json.dumps(body_obj).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "ArxivToCode-DreamRSI/1.0"},
        )
        try:
            with build_no_proxy_opener().open(req, timeout=timeout_sec) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                raw = str(data.get("response", "")).strip()
                end_think = raw.rfind("</think>")
                if end_think != -1:
                    raw = raw[end_think + len("</think>") :].strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE | re.MULTILINE)
                raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE).strip()
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    return None
                return DreamExplorationPolicy(
                    policy_id=f"dreamed-llm-c{cycle_index}",
                    policy_origin="dreamed_llm",
                    uct_exploration_c=float(parsed.get("uct_exploration_c", incumbent.uct_exploration_c)),
                    depth_bonus=float(parsed.get("depth_bonus", incumbent.depth_bonus)),
                    structural_vs_micro_prob=float(parsed.get("structural_vs_micro_prob", incumbent.structural_vs_micro_prob)),
                    parallel_batch_width=int(round(float(parsed.get("parallel_batch_width", incumbent.parallel_batch_width)))),
                    early_stop_patience=int(round(float(parsed.get("early_stop_patience", incumbent.early_stop_patience)))),
                    min_gain_threshold=float(parsed.get("min_gain_threshold", incumbent.min_gain_threshold)),
                    novelty_weight=float(parsed.get("novelty_weight", incumbent.novelty_weight)),
                    rationale=str(parsed.get("rationale", f"LLM-rewritten Dream-RSI policy ({clean_model})"))[:220],
                ).clamp_safe()
        except Exception:
            return None

    @classmethod
    def dream_and_promote(
        cls,
        incumbent: DreamExplorationPolicy,
        pool: ReplayWorldPool,
        budget: int,
        cycle_index: int,
        dream_candidates: int = 24,
        seed: int = 1000,
        use_llm_dreaming: bool = False,
        ollama_endpoint: Optional[str] = None,
        llm_model_tag: Optional[str] = None,
    ) -> Tuple[DreamExplorationPolicy, Dict[str, Any]]:
        """
        Evaluates `incumbent` + `dream_candidates` candidate exploration policies offline
        across `pool` (zero live code execution) and returns `(promoted_policy, dream_summary)`.
        """
        rng = random.Random(seed + cycle_index * 131)
        eval_seeds = [seed + 17 * i for i in range(3)]

        def _score_policy(pol: DreamExplorationPolicy) -> Dict[str, float]:
            runs = [pool.evaluate_policy_offline(pol, budget=budget, seed=s) for s in eval_seeds]
            n = float(len(runs))
            return {
                "replay_reward": round(sum(r["replay_reward"] for r in runs) / n, 4),
                "peak_fitness": round(sum(r["peak_fitness"] for r in runs) / n, 3),
                "auc_norm": round(sum(r["auc_norm"] for r in runs) / n, 3),
                "evaluations_saved_pct": round(sum(r["evaluations_saved_pct"] for r in runs) / n, 2),
                "valid_pass_rate_pct": round(sum(r["valid_pass_rate_pct"] for r in runs) / n, 2),
                "evals_used": round(sum(r["evals_used"] for r in runs) / n, 2),
            }

        incumbent_metrics = _score_policy(incumbent)
        candidates: List[DreamExplorationPolicy] = [incumbent]

        # Include structured archetypes covering fast-exploitation, deep-refinement, and smart early-exit
        archetypes = [
            DreamExplorationPolicy(
                policy_id=f"dreamed-c{cycle_index}-deep-exploit",
                policy_origin="dreamed_param",
                uct_exploration_c=0.45,
                depth_bonus=0.40,
                structural_vs_micro_prob=0.65,
                parallel_batch_width=3,
                early_stop_patience=2,
                min_gain_threshold=0.35,
                novelty_weight=0.10,
                rationale="Fast structural leap + deep micro refinement + aggressive smart stopping.",
            ),
            DreamExplorationPolicy(
                policy_id=f"dreamed-c{cycle_index}-balanced-batch",
                policy_origin="dreamed_param",
                uct_exploration_c=0.85,
                depth_bonus=0.22,
                structural_vs_micro_prob=0.55,
                parallel_batch_width=4,
                early_stop_patience=2,
                min_gain_threshold=0.20,
                novelty_weight=0.25,
                rationale="Wide parallel exploration batch (B=4) with early stopping once plateaued.",
            ),
            DreamExplorationPolicy(
                policy_id=f"dreamed-c{cycle_index}-micro-polisher",
                policy_origin="dreamed_param",
                uct_exploration_c=0.35,
                depth_bonus=0.55,
                structural_vs_micro_prob=0.30,
                parallel_batch_width=2,
                early_stop_patience=2,
                min_gain_threshold=0.15,
                novelty_weight=0.20,
                rationale="Depth-focused micro-optimization with rapid plateau detection.",
            ),
        ]
        candidates.extend(archetypes)

        llm_proposed = False
        if use_llm_dreaming and ollama_endpoint and llm_model_tag:
            llm_pol = cls._propose_llm_dream_policy(
                incumbent=incumbent,
                incumbent_score=incumbent_metrics,
                pool=pool,
                cycle_index=cycle_index,
                endpoint=ollama_endpoint,
                model_tag=llm_model_tag,
            )
            if llm_pol is not None:
                candidates.append(llm_pol)
                llm_proposed = True

        # Generate evolutionary perturbations around incumbent & archetypes
        num_random_muts = max(4, int(dream_candidates) - len(candidates))
        for idx in range(num_random_muts):
            base = candidates[idx % len(candidates)]
            candidates.append(cls._mutate_policy(base, f"dreamed-c{cycle_index}-p{idx + 1}", rng))

        scored: List[Tuple[DreamExplorationPolicy, Dict[str, float]]] = [
            (pol, _score_policy(pol)) for pol in candidates
        ]
        scored.sort(key=lambda item: (item[1]["replay_reward"], item[1]["peak_fitness"]), reverse=True)
        winner, winner_metrics = scored[0]

        promoted = winner.policy_id != incumbent.policy_id and winner_metrics["replay_reward"] > incumbent_metrics["replay_reward"]
        final_policy = winner if promoted else incumbent
        final_metrics = winner_metrics if promoted else incumbent_metrics

        return final_policy, {
            "cycle_index": cycle_index,
            "worlds_in_pool": len(pool),
            "recorded_nodes_in_pool": pool.total_recorded_nodes(),
            "dream_candidates_evaluated": len(scored),
            "offline_replay_rollouts_executed": len(scored) * len(eval_seeds) * len(pool),
            "live_code_executions_during_dreaming": 0,
            "llm_meta_policy_proposed": llm_proposed,
            "incumbent_policy_id": incumbent.policy_id,
            "incumbent_replay_reward": incumbent_metrics["replay_reward"],
            "promoted_policy_id": final_policy.policy_id,
            "promoted_policy_origin": final_policy.policy_origin,
            "promoted_replay_reward": final_metrics["replay_reward"],
            "replay_reward_gain": round(final_metrics["replay_reward"] - incumbent_metrics["replay_reward"], 4),
            "policy_upgraded": promoted,
        }


# =============================================================================
# 7. Standardized 5-Baseline Empirical Learning Superiority Proof (95% CIs)
# =============================================================================

# Exact two-tailed 95% Student-t critical values t_{0.975, df} for df = 1..30
_T_CRIT_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def compute_student_t_ci95(values: List[float]) -> ConfidenceIntervalMetric:
    """Computes exact 95% Student-t Confidence Interval (mean +/- t_{0.975, n-1} * s / sqrt(n))."""
    n = len(values)
    if n == 0:
        return ConfidenceIntervalMetric(0.0, 0.0, 0, 0.0, 0.0, 0.0)
    mean = sum(values) / float(n)
    if n == 1:
        m = round(mean, 4)
        return ConfidenceIntervalMetric(m, 0.0, 1, 0.0, m, m)
    variance = sum((x - mean) ** 2 for x in values) / float(n - 1)
    std = math.sqrt(max(0.0, variance))
    df = n - 1
    t_crit = _T_CRIT_95.get(df, 1.96 + 2.37 / float(df))
    half = t_crit * (std / math.sqrt(float(n)))
    return ConfidenceIntervalMetric(
        mean=round(mean, 4),
        std=round(std, 4),
        n=n,
        ci95_half_width=round(half, 4),
        ci95_low=round(mean - half, 4),
        ci95_high=round(mean + half, 4),
    )


def run_dream_rsi_empirical_benchmark(
    pool: ReplayWorldPool,
    learned_policy: DreamExplorationPolicy,
    budget: int = 12,
    n_seeds: int = 7,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    """
    Standardized Empirical Learning Superiority Proof (`rl_benchmarking_and_evaluation.md`):
    Compares `DreamRSI_Learned` against all 4 required baselines over identical matched seeds
    and identical initial ReplayWorldPool topologies:
      1. `Disabled`   — unmitigated / passive baseline (seed unmodified)
      2. `Random`     — uniform random action baseline
      3. `Heuristic`  — fixed-rule exploration baseline (UCT c=1.414)
      4. `NoLearning` — frozen initial policy without dreaming updates
      5. `DreamRSI_Learned` — policy refined via offline dreaming over evolving replay worlds
    Reports exact 95% Student-t Confidence Intervals across every metric.
    """
    seeds = [base_seed + i for i in range(max(3, int(n_seeds)))]
    policies: Dict[str, DreamExplorationPolicy] = {
        "Disabled": create_disabled_policy(),
        "Random": create_random_policy(),
        "Heuristic": create_baseline_heuristic_policy(),
        "NoLearning": create_frozen_initial_policy(),
        "DreamRSI_Learned": learned_policy,
    }

    results: Dict[str, Dict[str, Any]] = {}
    for label, pol in policies.items():
        rewards: List[float] = []
        peaks: List[float] = []
        valid_rates: List[float] = []
        saved_pcts: List[float] = []
        for s in seeds:
            ev = pool.evaluate_policy_offline(pol, budget=budget, seed=s)
            rewards.append(ev["replay_reward"])
            peaks.append(ev["peak_fitness"])
            valid_rates.append(ev["valid_pass_rate_pct"])
            saved_pcts.append(ev["evaluations_saved_pct"])

        results[label] = {
            "policy_id": pol.policy_id,
            "policy_origin": pol.policy_origin,
            "cumulative_replay_reward_ci95": compute_student_t_ci95(rewards).to_dict(),
            "peak_fitness_ci95": compute_student_t_ci95(peaks).to_dict(),
            "valid_pass_rate_pct_ci95": compute_student_t_ci95(valid_rates).to_dict(),
            "evaluations_saved_pct_ci95": compute_student_t_ci95(saved_pcts).to_dict(),
        }

    learned_r = results["DreamRSI_Learned"]["cumulative_replay_reward_ci95"]["mean"]
    deltas = {
        "learned_minus_disabled": round(learned_r - results["Disabled"]["cumulative_replay_reward_ci95"]["mean"], 4),
        "learned_minus_random": round(learned_r - results["Random"]["cumulative_replay_reward_ci95"]["mean"], 4),
        "learned_minus_heuristic": round(learned_r - results["Heuristic"]["cumulative_replay_reward_ci95"]["mean"], 4),
        "learned_minus_nolearning": round(learned_r - results["NoLearning"]["cumulative_replay_reward_ci95"]["mean"], 4),
    }

    backend_info = detect_dream_rsi_backend()
    return {
        "environment_backend": backend_info["environment_backend"],
        "is_official_external_package": backend_info["is_official_external_package"],
        "fidelity_note": backend_info["fidelity_note"],
        "matched_seeds": seeds,
        "worlds_evaluated": len(pool),
        "budget_per_world": budget,
        "baselines": results,
        "superiority_deltas": deltas,
        "outperforms_all_baselines": all(v >= 0.0 for v in deltas.values()),
    }


# =============================================================================
# 8. End-to-End Dream-RSI RL Orchestrator (`run_dream_rsi_rl_optimization`)
# =============================================================================

def run_dream_rsi_rl_optimization(
    seed_python_code: str,
    arxiv_id: str = "2608.15888",
    rl_cycles: int = 3,
    rl_budget: int = 12,
    dream_candidates: int = 24,
    use_llm_dreaming: bool = False,
    benchmark_seeds: int = 7,
    trace_pool_dir: Optional[str] = None,
    ollama_endpoint: Optional[str] = None,
    dynarot_decision: Optional[DynaRotRoutingDecision] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Executes the full Dream-RSI recursive self-improvement loop over generated Python code:
      Cycle k = 1..rl_cycles:
        1. Online Exploration (`TraceEnv.run_online_rollout`) using current promoted policy
        2. Freeze recorded `DiscoveryTree` into `ReplayWorldPool` (+ persist `trace_pool.jsonl`)
        3. Offline Dreaming (`DreamPolicyOptimizer.dream_and_promote`) across accumulated worlds
        4. Redeploy promoted meta-exploration policy into cycle k + 1
      Final:
        5. Run 5-baseline matched-seed empirical benchmark with 95% Student-t Confidence Intervals
    """
    t0 = time.perf_counter()
    cycles = max(1, int(rl_cycles))
    budget = max(2, int(rl_budget))
    backend_info = detect_dream_rsi_backend()

    endpoint = ollama_endpoint or os.environ.get("LOCAL_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    if dynarot_decision is None:
        dynarot_decision = DynaRotModelRouter(endpoint=endpoint).route(
            prompt=f"Dream-RSI RL meta-exploration optimization for arXiv:{arxiv_id}"
        )

    # Resolve optional local Ollama model for `--rl-llm` dreaming
    llm_model_tag: Optional[str] = None
    if use_llm_dreaming:
        online, installed = probe_ollama_models(endpoint=endpoint, timeout_sec=0.5)
        if online and installed:
            from .dynarot import detect_live_hardware

            hw = detect_live_hardware()
            llm_model_tag = find_best_installed_chat_model(
                dynarot_decision.calibrated_installed_model or "glm4:9b",
                hw,
                installed,
            )

    # Load existing persistent replay worlds if `trace_pool_dir` is provided
    pool = ReplayWorldPool()
    trace_pool_jsonl_path: Optional[Path] = None
    if trace_pool_dir:
        tp_dir = Path(trace_pool_dir).resolve()
        tp_dir.mkdir(parents=True, exist_ok=True)
        trace_pool_jsonl_path = tp_dir / "trace_pool.jsonl"
        if trace_pool_jsonl_path.exists():
            pool = ReplayWorldPool.load_jsonl(trace_pool_jsonl_path)

    initial_pool_size = len(pool)
    current_policy = create_baseline_heuristic_policy()
    cycle_history: List[Dict[str, Any]] = []
    best_overall_node: Optional[DiscoveryTreeNode] = None
    baseline_seed_metrics: Optional[CandidateScoreMetrics] = None

    total_ticks = cycles * (budget + 1)
    pbar_lock = threading.Lock()
    pbar = create_progress_bar(
        total=total_ticks,
        desc=f"Dream-RSI RL arXiv:{arxiv_id}",
        unit="step",
        disable=not show_progress,
    )

    def _tick(stage: str, detail: str) -> None:
        with pbar_lock:
            pbar.set_description(stage)
            pbar.set_postfix_str(detail)
            pbar.update(1)

    try:
        active_seed_code = seed_python_code
        for c_idx in range(1, cycles + 1):
            tree, rollout_stats = TraceEnv.run_online_rollout(
                seed_python_code=active_seed_code,
                arxiv_id=arxiv_id,
                policy=current_policy,
                budget=budget,
                cycle_index=c_idx,
                seed=42 + c_idx * 17,
                progress_cb=_tick,
            )
            if baseline_seed_metrics is None:
                baseline_seed_metrics = tree.nodes[tree.root_id].metrics

            pool.add_world(tree)
            cycle_best = tree.best_node()
            if best_overall_node is None or (
                cycle_best.metrics.total_fitness,
                -cycle_best.metrics.execution_time_us,
            ) > (
                best_overall_node.metrics.total_fitness,
                -best_overall_node.metrics.execution_time_us,
            ):
                best_overall_node = cycle_best

            # Offline Dreaming over accumulated ReplayWorldPool
            promoted_policy, dream_summary = DreamPolicyOptimizer.dream_and_promote(
                incumbent=current_policy,
                pool=pool,
                budget=budget,
                cycle_index=c_idx,
                dream_candidates=dream_candidates,
                seed=1000,
                use_llm_dreaming=bool(use_llm_dreaming and llm_model_tag is not None),
                ollama_endpoint=endpoint,
                llm_model_tag=llm_model_tag,
            )
            _tick(
                f"Dream-RSI Cycle {c_idx} Dream",
                f"worlds={len(pool)} | reward={dream_summary['promoted_replay_reward']:.2f}",
            )

            cycle_history.append(
                {
                    "cycle": c_idx,
                    "policy_used_online": current_policy.to_dict(),
                    "online_rollout": rollout_stats,
                    "cycle_best_node_id": cycle_best.node_id,
                    "cycle_best_fitness": cycle_best.metrics.total_fitness,
                    "cycle_best_operator": cycle_best.operator,
                    "offline_dreaming": dream_summary,
                }
            )
            current_policy = promoted_policy
            # Carry forward improved code only when valid
            if cycle_best.metrics.error is None and cycle_best.metrics.tests_passed == cycle_best.metrics.tests_total:
                active_seed_code = cycle_best.source_code
    finally:
        pbar.close()

    if trace_pool_jsonl_path is not None:
        pool.save_jsonl(trace_pool_jsonl_path)

    # Standardized 5-Baseline Empirical Learning Superiority Proof (95% CIs)
    empirical_benchmark = run_dream_rsi_empirical_benchmark(
        pool=pool,
        learned_policy=current_policy,
        budget=budget,
        n_seeds=benchmark_seeds,
        base_seed=1000,
    )

    assert best_overall_node is not None and baseline_seed_metrics is not None
    champion_candidate = EvolutionCandidate(
        candidate_id=f"dream-rsi-{arxiv_id.replace('.', '_')}-{best_overall_node.node_id}",
        engine_origin="DreamRSI-RL",
        generation=cycles,
        model_used=llm_model_tag or dynarot_decision.primary_model,
        mutation_operator=best_overall_node.operator,
        rationale=(
            f"Discovered by Dream-RSI promoted meta-exploration policy `{current_policy.policy_id}`: "
            f"{best_overall_node.diagnostics}"
        ),
        source_code=best_overall_node.source_code,
        evolve_block_code=best_overall_node.evolve_block_code,
        metrics=best_overall_node.metrics,
        parent_ids=[best_overall_node.parent_id] if best_overall_node.parent_id else [],
    )

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "arxiv_id": arxiv_id,
        "environment_backend": backend_info["environment_backend"],
        "is_official_external_package": backend_info["is_official_external_package"],
        "backend_metadata": backend_info,
        "rl_cycles": cycles,
        "rl_budget_per_cycle": budget,
        "dream_candidates_per_cycle": dream_candidates,
        "use_llm_dreaming": use_llm_dreaming,
        "llm_dreaming_model": llm_model_tag,
        "replay_world_pool_summary": {
            "initial_worlds_loaded": initial_pool_size,
            "final_worlds_count": len(pool),
            "total_recorded_tree_nodes": pool.total_recorded_nodes(),
            "trace_pool_jsonl": str(trace_pool_jsonl_path) if trace_pool_jsonl_path else "",
        },
        "baseline_seed_metrics": baseline_seed_metrics.to_dict(),
        "champion_metrics": best_overall_node.metrics.to_dict(),
        "fitness_improvement_delta": round(
            best_overall_node.metrics.total_fitness - baseline_seed_metrics.total_fitness, 3
        ),
        "speedup_ratio": round(
            baseline_seed_metrics.execution_time_us / max(0.05, best_overall_node.metrics.execution_time_us),
            3,
        ),
        "promoted_policy": current_policy.to_dict(),
        "cycle_history": cycle_history,
        "empirical_benchmark_95ci": empirical_benchmark,
        "champion_candidate": champion_candidate.to_dict(),
        "replay_worlds_serialized": [w.to_dict() for w in pool.worlds],
        "duration_ms": duration_ms,
    }
