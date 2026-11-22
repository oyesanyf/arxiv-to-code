"""
Core data structures and typed schemas for ArXiv-to-Code.

Covers:
1. arXiv Paper Metadata & 3-Class Google LangExtract Grounded Spans
2. ModelFusion DynaRot Hardware Sizing, Pareto Routing & Compute Split Plans
3. Paper-to-Python Code Bundles (with EVOLVE-BLOCK convention & deterministic evaluator)
4. Combined ShinkaEvolve + OpenEvolve Evolutionary Search Trajectories & MAP-Elites Grids
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SystemHardwareResources:
    """Live runtime hardware telemetry strictly following the Dynamic Available Memory Law (QuorumRAG + DynaRot parity)."""

    total_ram_gb: float
    free_ram_gb: float
    cpu_cores: int
    gpu_name: Optional[str]
    gpu_vram_total_gb: float
    gpu_vram_free_gb: float
    free_disk_gb: float = 100.0
    gpu_count: int = 0

    @property
    def free_vram_mb(self) -> float:
        return round(self.gpu_vram_free_gb * 1024.0, 1)

    @property
    def has_usable_gpu(self) -> bool:
        return self.gpu_name is not None and self.gpu_vram_free_gb > 0.5

    @property
    def ram_budget_gb(self) -> float:
        return round(max(0.0, (self.free_ram_gb - 3.0) * 0.70), 2)

    @property
    def gpu_budget_gb(self) -> float:
        return round(max(0.0, self.gpu_vram_free_gb * 0.85), 2)

    @property
    def model_budget_gb(self) -> float:
        return max(self.ram_budget_gb, self.gpu_budget_gb)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["free_vram_mb"] = self.free_vram_mb
        data["has_usable_gpu"] = self.has_usable_gpu
        data["ram_budget_gb"] = self.ram_budget_gb
        data["gpu_budget_gb"] = self.gpu_budget_gb
        data["model_budget_gb"] = self.model_budget_gb
        return data


@dataclass
class DynaRotRoutingDecision:
    """
    ModelFusion DynaRot routing and hardware-aware model allocation decision.
    Splits compute budget between:
    - primary_model: Sized strictly by live available free RAM & free VRAM (Anti-OOM Law)
    - shinka_heavy_reasoning_model: Heavy reasoning model for ShinkaEvolve structural discovery
    - openevolve_fast_mutation_model: Fast lightweight model for OpenEvolve high-volume MAP-Elites micro-mutations
    """

    task_category: str
    target_language: str
    complexity_score: int
    requires_reasoning: bool
    hardware_tier: str
    primary_model: str
    shinka_heavy_reasoning_model: str
    openevolve_fast_mutation_model: str
    ensemble_size: int
    suggested_models: List[str]
    pareto_knee_model: str
    pareto_front_models: List[str]
    ucb1_scores: Dict[str, float]
    installed_ollama_models: List[str]
    missing_models: List[str]
    auto_install_commands: List[str]
    routed_by: str
    routing_duration_ms: float
    hardware: Dict[str, Any]
    recommended_glm_model: str = "glm4:9b"
    recommended_gemma_model: str = "gemma2:2b"
    calibrated_installed_model: Optional[str] = None
    selection_rationale: str = ""
    provision_plan: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateScoreMetrics:
    """Deterministic multi-objective evaluation score for an EVOLVE-BLOCK Python program."""

    candidate_id: str
    total_fitness: float
    correctness_score: float
    speed_score: float
    memory_score: float
    invariant_score: float
    execution_time_us: float
    peak_memory_kib: float
    tests_passed: int
    tests_total: int
    niche_coords: Tuple[int, int] = (0, 0)
    niche_label: str = "balanced"
    ast_fingerprint: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["niche_coords"] = list(self.niche_coords)
        return data


@dataclass
class EvolutionCandidate:
    """A single evolved Python source file adhering to the EVOLVE-BLOCK convention."""

    candidate_id: str
    engine_origin: str  # "seed" | "ShinkaEvolve" | "OpenEvolve" | "CrossPollinated"
    generation: int
    model_used: str
    mutation_operator: str
    rationale: str
    source_code: str
    evolve_block_code: str
    metrics: CandidateScoreMetrics
    parent_ids: List[str] = field(default_factory=list)
    file_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


# =============================================================================
# Dream-RSI: Recursive Self-Improvement through Evolving Worlds (arXiv:2609.14858)
# =============================================================================

@dataclass
class DiscoveryTreeNode:
    """A single recorded node in a Dream-RSI DiscoveryTree (code snapshot + realized execution metrics)."""

    node_id: str
    parent_id: Optional[str]
    depth: int
    source_code: str
    evolve_block_code: str
    operator: str
    operator_family: str  # "seed" | "structural" | "micro" | "hybrid_llm"
    ast_fingerprint: str
    metrics: CandidateScoreMetrics
    diagnostics: str = ""
    eval_cost_ms: float = 0.0
    children_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryTreeNode":
        m = data["metrics"]
        coords_raw = m.get("niche_coords", [0, 0])
        metrics = CandidateScoreMetrics(
            candidate_id=str(m.get("candidate_id", data["node_id"])),
            total_fitness=float(m.get("total_fitness", 0.0)),
            correctness_score=float(m.get("correctness_score", 0.0)),
            speed_score=float(m.get("speed_score", 0.0)),
            memory_score=float(m.get("memory_score", 0.0)),
            invariant_score=float(m.get("invariant_score", 0.0)),
            execution_time_us=float(m.get("execution_time_us", 999999.0)),
            peak_memory_kib=float(m.get("peak_memory_kib", 999999.0)),
            tests_passed=int(m.get("tests_passed", 0)),
            tests_total=int(m.get("tests_total", 4)),
            niche_coords=(int(coords_raw[0]), int(coords_raw[1])),
            niche_label=str(m.get("niche_label", "balanced")),
            ast_fingerprint=str(m.get("ast_fingerprint", data.get("ast_fingerprint", ""))),
            error=m.get("error"),
        )
        return cls(
            node_id=str(data["node_id"]),
            parent_id=data.get("parent_id"),
            depth=int(data.get("depth", 0)),
            source_code=str(data.get("source_code", "")),
            evolve_block_code=str(data.get("evolve_block_code", "")),
            operator=str(data.get("operator", "")),
            operator_family=str(data.get("operator_family", "micro")),
            ast_fingerprint=str(data.get("ast_fingerprint", "")),
            metrics=metrics,
            diagnostics=str(data.get("diagnostics", "")),
            eval_cost_ms=float(data.get("eval_cost_ms", 0.0)),
            children_ids=list(data.get("children_ids") or []),
        )


@dataclass
class DiscoveryTree:
    """Frozen structured discovery history serving as a zero-cost Dream-RSI Replay World."""

    tree_id: str
    arxiv_id: str
    root_id: str
    seed_fitness: float
    cycle_index: int
    created_at_epoch: float
    nodes: Dict[str, DiscoveryTreeNode] = field(default_factory=dict)

    def best_node(self) -> DiscoveryTreeNode:
        valid = [
            n
            for n in self.nodes.values()
            if n.metrics.error is None and n.metrics.tests_passed == n.metrics.tests_total
        ]
        pool = valid or list(self.nodes.values())
        return max(pool, key=lambda n: (n.metrics.total_fitness, -n.metrics.execution_time_us))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "arxiv_id": self.arxiv_id,
            "root_id": self.root_id,
            "seed_fitness": self.seed_fitness,
            "cycle_index": self.cycle_index,
            "created_at_epoch": self.created_at_epoch,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryTree":
        nodes_raw = data.get("nodes") or {}
        nodes = {nid: DiscoveryTreeNode.from_dict(nd) for nid, nd in nodes_raw.items()}
        return cls(
            tree_id=str(data["tree_id"]),
            arxiv_id=str(data.get("arxiv_id", "custom")),
            root_id=str(data.get("root_id", "n0")),
            seed_fitness=float(data.get("seed_fitness", 0.0)),
            cycle_index=int(data.get("cycle_index", 0)),
            created_at_epoch=float(data.get("created_at_epoch", 0.0)),
            nodes=nodes,
        )


@dataclass
class DreamExplorationPolicy:
    """
    Programmable Dream-RSI meta-exploration policy governing branching,
    operator selection (structural vs micro), parallel batch width, and early stopping.
    """

    policy_id: str
    policy_origin: str  # "baseline_heuristic" | "random" | "frozen_initial" | "dreamed_param" | "dreamed_llm"
    uct_exploration_c: float = 1.414
    depth_bonus: float = 0.08
    structural_vs_micro_prob: float = 0.50
    parallel_batch_width: int = 2
    early_stop_patience: int = 4
    min_gain_threshold: float = 0.25
    novelty_weight: float = 0.15
    rationale: str = ""

    def clamp_safe(self) -> "DreamExplorationPolicy":
        """Enforces safe bounded parameter ranges for every meta-exploration hyperparameter."""
        return DreamExplorationPolicy(
            policy_id=self.policy_id,
            policy_origin=self.policy_origin,
            uct_exploration_c=round(max(0.0, min(4.0, float(self.uct_exploration_c))), 4),
            depth_bonus=round(max(-0.5, min(1.0, float(self.depth_bonus))), 4),
            structural_vs_micro_prob=round(max(0.05, min(0.95, float(self.structural_vs_micro_prob))), 4),
            parallel_batch_width=max(1, min(4, int(round(self.parallel_batch_width)))),
            early_stop_patience=max(1, min(12, int(round(self.early_stop_patience)))),
            min_gain_threshold=round(max(0.01, min(5.0, float(self.min_gain_threshold))), 4),
            novelty_weight=round(max(0.0, min(1.5, float(self.novelty_weight))), 4),
            rationale=self.rationale,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DreamExplorationPolicy":
        return cls(
            policy_id=str(data.get("policy_id", "policy-loaded")),
            policy_origin=str(data.get("policy_origin", "dreamed_param")),
            uct_exploration_c=float(data.get("uct_exploration_c", 1.414)),
            depth_bonus=float(data.get("depth_bonus", 0.08)),
            structural_vs_micro_prob=float(data.get("structural_vs_micro_prob", 0.50)),
            parallel_batch_width=int(data.get("parallel_batch_width", 2)),
            early_stop_patience=int(data.get("early_stop_patience", 4)),
            min_gain_threshold=float(data.get("min_gain_threshold", 0.25)),
            novelty_weight=float(data.get("novelty_weight", 0.15)),
            rationale=str(data.get("rationale", "")),
        ).clamp_safe()


@dataclass
class ConfidenceIntervalMetric:
    """95% Student-t Confidence Interval statistics for standardized RL evaluation."""

    mean: float
    std: float
    n: int
    ci95_half_width: float
    ci95_low: float
    ci95_high: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
