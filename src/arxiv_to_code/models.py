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
    """Live runtime hardware telemetry strictly following the Dynamic Available Memory Law."""

    total_ram_gb: float
    free_ram_gb: float
    cpu_cores: int
    gpu_name: Optional[str]
    gpu_vram_total_gb: float
    gpu_vram_free_gb: float

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
