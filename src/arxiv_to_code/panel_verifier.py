"""
ModelFusion Panel-of-Models Code Verification Engine (`panel_verifier.py`).

Ports ModelFusion's multi-model panel, memory-aware batching, and minority-preserving
Judge Analysis (`D:\\harfile\\ModelFusion\\crates\\core\\src\\fusion_engine\\{mod.rs, fusion.rs, judge.rs}`)
combined with QuorumRAG's resource-calibrated GLM/Qwen/Gemma selection:

  1. Dynamic Panel Sizing (`derive_fusion_model_count`):
     - Free RAM >= 64 GB OR Free VRAM >= 24 GB -> 7 panelists
     - Free RAM >= 32 GB OR Free VRAM >= 12 GB -> 5 panelists
     - Free RAM >= 16 GB OR Free VRAM >= 6 GB  -> 4 panelists
     - Free RAM >= 8 GB  OR Free VRAM >= 3 GB  -> 3 panelists
     - Below 8 GB                              -> 2 panelists
  2. Memory-Aware Local Batching (`calculate_panel_batch_size`):
     - Free VRAM >= 20 GB -> up to 4 concurrent models per batch
     - Free VRAM >= 11 GB -> up to 2 concurrent models per batch
     - Free VRAM <  11 GB -> batch size 1 (sequential) to prevent VRAM OOM/thrashing
  3. Resource-Gated Multi-Family Panel Assembly:
     - Injects Z.ai GLM specialist (`glm-4.7-flash` / `glm4:9b`) when Free RAM >= 8 GB or Free VRAM >= 6 GB,
       paired with Qwen 2.5, DeepSeek-R1 (`<think>` reasoning), and Gemma specialists strictly fitting
       available runtime memory.
     - Automatically calibrates against installed chat/completion models in Ollama (excluding
       embedding/vision/decision-only models like Bespoke-Nimble-9B) and supplements with diverse
       temperature + reviewer-persona samples when fewer distinct models are installed locally.
  4. 3-Stage Deliberation & Fail-Closed Reconciliation:
     - Stage 0: Layer-1 Deterministic AST, security invariant, anti-mock, and runtime unit-test gate.
     - Stage 1: Batched multi-model panel audit (stripping `<think>...</think>` blocks cleanly).
     - Stage 2: Judge Analysis (`consensus`, `disagreements`, `unique_insights`, `blind_spots`,
       `risk_flags`, `recommended_final_position`) enforcing ModelFusion's Critical Minority-Quality Rule.
     - Stage 3: Fail-closed verdict reconciliation (deterministic failures & verified CRITICAL minority
       defects can NEVER be overruled by majority optimism).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .dynarot import (
    DynaRotModelRouter,
    detect_live_hardware,
    estimated_model_disk_gb,
    is_synthesis_chat_model,
    model_capability_score,
    model_fits_free_memory,
    probe_ollama_models,
    sanitize_model_tag,
    size_gemma_by_hardware,
    size_glm_by_hardware,
)
from .langextract_engine import normalize_arxiv_id
from .llm_verifier import (
    CodeVerificationReport,
    _build_llm_verification_prompt,
    _invoke_ollama_llm_judge,
    parse_llm_verification_json,
    run_deterministic_code_verification,
)
from .models import DynaRotRoutingDecision, SystemHardwareResources


PANEL_REVIEWER_PERSONAS: List[Tuple[str, float, str]] = [
    (
        "Formal-Methods & Paper-Fidelity Auditor",
        0.1,
        "Focus strictly on whether every mathematical formulation (conformal bounds, martingales, "
        "meet-semilattice narrowing) and Google LangExtract grounded span is faithfully implemented.",
    ),
    (
        "Adversarial Security & Anti-Mock Red-Teamer",
        0.25,
        "Hunt for bypasses, timing leaks, naive single-token mocks, unsafe dynamic execution, "
        "or fail-open boundary logic.",
    ),
    (
        "Algorithmic Correctness & Edge-Case Critic",
        0.45,
        "Inspect empty inputs, numerical edge cases (alpha -> 0, overflow, case folding), "
        "stateful session tracking, and test completeness.",
    ),
    (
        "Systems Performance & Memory Efficiency Reviewer",
        0.65,
        "Audit EVOLVE-BLOCK hot-path allocations, latency, memory footprint, and concurrency safety.",
    ),
]


@dataclass
class PanelMemberConfig:
    """Configuration for a single model slot inside the ModelFusion verification panel."""

    slot_index: int
    display_name: str
    model_tag: str
    family: str
    persona: str
    persona_directive: str
    temperature: float
    installed: bool = False


@dataclass
class PanelistVerdict:
    """Structured audit response returned by one panel member."""

    slot_index: int
    display_name: str
    model_tag: str
    persona: str
    temperature: float
    succeeded: bool
    status: str
    verdict: str = "N/A"
    paper_fidelity_score: float = 0.0
    correctness_score: float = 0.0
    security_score: float = 0.0
    summary: str = ""
    grounded_span_coverage: List[str] = field(default_factory=list)
    findings: List[Dict[str, str]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeAnalysis:
    """Exact ModelFusion `judge.rs` structured multi-model deliberation output."""

    consensus: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)
    unique_insights: List[str] = field(default_factory=list)
    blind_spots: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    recommended_final_position: str = ""
    judge_verdict: str = "WARN"
    minority_override_triggered: bool = False
    judge_model: str = ""
    judged_by_llm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 1. Dynamic Panel Sizing & Memory-Aware Batching (ModelFusion Parity)
# =============================================================================

def derive_fusion_model_count(hw: Optional[SystemHardwareResources] = None) -> int:
    """
    Derives dynamic panel model count strictly from live available runtime memory
    (exact ModelFusion `crates/model_selection/src/memory.rs::derive_fusion_model_count`).
    """
    if hw is None:
        hw = detect_live_hardware()
    free_ram = hw.free_ram_gb
    free_vram = hw.gpu_vram_free_gb

    if free_ram >= 64.0 or free_vram >= 24.0:
        return 7
    if free_ram >= 32.0 or free_vram >= 12.0:
        return 5
    if free_ram >= 16.0 or free_vram >= 6.0:
        return 4
    if free_ram >= 8.0 or free_vram >= 3.0:
        return 3
    return 2


def calculate_panel_batch_size(
    panel_len: int,
    hw: Optional[SystemHardwareResources] = None,
) -> int:
    """
    Calculates how many local models can execute concurrently without VRAM thrashing/OOM
    (exact ModelFusion `crates/core/src/fusion_engine/fusion.rs::calculate_batch_size`):
      - MODELFUSION_BATCH_SIZE env override respected if set
      - Free VRAM >= 20 GB -> up to 4 concurrent models
      - Free VRAM >= 11 GB -> up to 2 concurrent models
      - Free VRAM <  11 GB -> 1 (sequential for safety)
    """
    if panel_len <= 1:
        return 1
    env_override = os.environ.get("MODELFUSION_BATCH_SIZE", "").strip()
    if env_override.isdigit():
        return max(1, min(panel_len, int(env_override)))

    if hw is None:
        hw = detect_live_hardware()
    if hw.has_usable_gpu:
        if hw.gpu_vram_free_gb >= 20.0:
            return min(4, panel_len)
        if hw.gpu_vram_free_gb >= 11.0:
            return min(2, panel_len)
    return 1


def assess_glm_for_panel(hw: SystemHardwareResources) -> Dict[str, Any]:
    """
    Resource-gated Z.ai GLM activation matching ModelFusion `memory.rs::assess_glm_for_mode`:
    Strictly activates GLM in panel mode when `free_ram_gb >= 8.0 or gpu_vram_free_gb >= 6.0`.
    """
    resources_available = hw.free_ram_gb >= 8.0 or hw.gpu_vram_free_gb >= 6.0
    if not resources_available:
        return {
            "activated": False,
            "ollama_tag": "glm4:9b",
            "reason": (
                f"GLM excluded from panel: Free RAM ({hw.free_ram_gb:.1f} GB) < 8.0 GB "
                f"and Free VRAM ({hw.gpu_vram_free_gb:.1f} GB) < 6.0 GB."
            ),
        }
    tag = "glm-4.7-flash" if (hw.free_ram_gb >= 18.0 or hw.gpu_vram_free_gb >= 16.0) else "glm4:9b"
    return {
        "activated": True,
        "ollama_tag": tag,
        "reason": f"Z.ai GLM specialist ({tag}) activated in panel (Free RAM {hw.free_ram_gb:.1f} GB, Free VRAM {hw.gpu_vram_free_gb:.1f} GB).",
    }


def _infer_family(tag: str) -> str:
    low = tag.strip().lower()
    if low.startswith("glm"):
        return "glm"
    if low.startswith("qwen"):
        return "qwen"
    if low.startswith("deepseek"):
        return "deepseek"
    if low.startswith("gemma"):
        return "gemma"
    if low.startswith("llama"):
        return "llama"
    return "open-weight"


def assemble_verification_panel(
    hw: SystemHardwareResources,
    installed_models: List[str],
    dynarot_decision: DynaRotRoutingDecision,
    requested_panel_size: Optional[int] = None,
    explicit_models: Optional[List[str]] = None,
    panel_mode: str = "multi-model",
) -> Tuple[List[PanelMemberConfig], List[str], str, Dict[str, Any]]:
    """
    Constructs a ModelFusion verification panel calibrated to runtime RAM/VRAM and installed Ollama models.

    Returns `(panel_members, fallback_pool, judge_model_tag, assembly_meta)`:
      - Filters installed models to chat/completion-eligible & memory-fitting tags (excluding
        embedding, vision, and decision-only models such as Bespoke-Nimble-9B).
      - In `multi-model` mode, assembles distinct families (Z.ai GLM + Qwen 2.5 + DeepSeek-R1 + Gemma).
        When Ollama is online with >=1 eligible installed model(s), prioritizes installed models and
        supplements remaining panel slots with temperature + persona variations (matching ModelFusion
        `mod.rs` lines 411-430) so live panel deliberation succeeds even with 1-2 installed models!
      - In `multi-sample` mode, creates N diverse temperature & persona configurations of the top model.
    """
    derived_count = derive_fusion_model_count(hw)
    # Default cap to 3 panelists for snappy local code verification unless explicitly overridden
    target_size = max(2, min(7, int(requested_panel_size))) if requested_panel_size else min(3, derived_count)
    mode_norm = (panel_mode or "multi-model").strip().lower()
    glm_assessment = assess_glm_for_panel(hw)

    eligible_installed = [
        m
        for m in installed_models
        if is_synthesis_chat_model(m) and model_fits_free_memory(m, hw.free_ram_gb, hw.free_vram_mb)
    ]
    gpu_residency_gb = max(hw.gpu_vram_free_gb, hw.gpu_vram_total_gb * 0.85)
    eligible_installed.sort(
        key=lambda m: (
            1 if (gpu_residency_gb >= 4.0 and estimated_model_disk_gb(m) <= (gpu_residency_gb + 0.5)) else 0,
            model_capability_score(m),
        ),
        reverse=True,
    )

    # Ideal multi-family catalog fitting current runtime resources (used when offline or as recommendations)
    ideal_candidates: List[str] = []
    if explicit_models:
        for em in explicit_models:
            clean = sanitize_model_tag(em)
            if clean and clean not in ideal_candidates:
                ideal_candidates.append(clean)
    else:
        if glm_assessment["activated"]:
            ideal_candidates.append(glm_assessment["ollama_tag"])
        for cand in [
            dynarot_decision.shinka_heavy_reasoning_model,
            dynarot_decision.primary_model,
            "deepseek-r1:14b" if model_fits_free_memory("deepseek-r1:14b", hw.free_ram_gb, hw.free_vram_mb) else "deepseek-r1:7b",
            size_gemma_by_hardware(hw.free_ram_gb, hw.free_vram_mb),
            dynarot_decision.openevolve_fast_mutation_model,
            size_glm_by_hardware(hw.free_ram_gb, hw.free_vram_mb),
        ]:
            clean = sanitize_model_tag(cand)
            if clean and clean not in ideal_candidates and model_fits_free_memory(clean, hw.free_ram_gb, hw.free_vram_mb):
                ideal_candidates.append(clean)

    # Decide the active base pool: prefer actually installed eligible models if present (QuorumRAG calibration)
    if explicit_models:
        base_tags = list(ideal_candidates)
        installed_set = {m.lower() for m in eligible_installed}
    elif eligible_installed:
        # Order installed models: put any installed GLM/DeepSeek/Qwen/Gemma matching ideal order first
        ordered: List[str] = []
        for ideal in ideal_candidates:
            il = ideal.lower()
            for inst in eligible_installed:
                if inst not in ordered and (inst.lower() == il or inst.lower().startswith(f"{il}:")):
                    ordered.append(inst)
        for inst in eligible_installed:
            if inst not in ordered:
                ordered.append(inst)
        base_tags = ordered
        installed_set = {m.lower() for m in eligible_installed}
    else:
        base_tags = ideal_candidates or [dynarot_decision.primary_model]
        installed_set = set()

    members: List[PanelMemberConfig] = []
    fallback_pool: List[str] = []

    if mode_norm == "multi-sample":
        primary_tag = base_tags[0]
        for idx in range(target_size):
            persona_name, persona_temp, persona_dir = PANEL_REVIEWER_PERSONAS[idx % len(PANEL_REVIEWER_PERSONAS)]
            temp = round(0.15 + idx * (0.65 / max(1, target_size - 1)), 2)
            members.append(
                PanelMemberConfig(
                    slot_index=idx + 1,
                    display_name=f"{primary_tag} [{persona_name}, T={temp}]",
                    model_tag=primary_tag,
                    family=_infer_family(primary_tag),
                    persona=persona_name,
                    persona_directive=persona_dir,
                    temperature=temp,
                    installed=(primary_tag.lower() in installed_set),
                )
            )
        fallback_pool = base_tags[1:]
    else:
        # Multi-model mode: populate distinct models first, then supplement with diverse temperatures/personas
        distinct_used = base_tags[:target_size]
        fallback_pool = base_tags[target_size:]
        for idx, tag in enumerate(distinct_used):
            persona_name, persona_temp, persona_dir = PANEL_REVIEWER_PERSONAS[idx % len(PANEL_REVIEWER_PERSONAS)]
            members.append(
                PanelMemberConfig(
                    slot_index=idx + 1,
                    display_name=f"{tag} ({persona_name})",
                    model_tag=tag,
                    family=_infer_family(tag),
                    persona=persona_name,
                    persona_directive=persona_dir,
                    temperature=persona_temp,
                    installed=(tag.lower() in installed_set),
                )
            )

        # ModelFusion mod.rs Step 6: fill remaining panel slots with temperature + persona variations
        var_idx = 0
        while len(members) < target_size and distinct_used:
            base_tag = distinct_used[var_idx % len(distinct_used)]
            slot_num = len(members) + 1
            persona_name, persona_temp, persona_dir = PANEL_REVIEWER_PERSONAS[(slot_num - 1) % len(PANEL_REVIEWER_PERSONAS)]
            temp = round(min(0.85, persona_temp + 0.1 * (var_idx // max(1, len(distinct_used)))), 2)
            members.append(
                PanelMemberConfig(
                    slot_index=slot_num,
                    display_name=f"{base_tag} (var #{slot_num}, {persona_name}, T={temp})",
                    model_tag=base_tag,
                    family=_infer_family(base_tag),
                    persona=persona_name,
                    persona_directive=persona_dir,
                    temperature=temp,
                    installed=(base_tag.lower() in installed_set),
                )
            )
            var_idx += 1

    # Select Judge model (prefer DeepSeek-R1 / GLM / Qwen among installed or base_tags, matching ModelFusion mod.rs)
    judge_model = base_tags[0]
    for pref_kw in ("deepseek", "glm", "qwen", "gemma"):
        hit = next((t for t in base_tags if pref_kw in t.lower()), None)
        if hit is not None:
            judge_model = hit
            break

    batch_size = calculate_panel_batch_size(len(members), hw)
    meta = {
        "panel_mode": mode_norm,
        "derived_max_panel_count": derived_count,
        "effective_panel_size": len(members),
        "batch_size": batch_size,
        "glm_panel_assessment": glm_assessment,
        "eligible_installed_chat_models": eligible_installed,
        "recommended_panel_pulls": [f"ollama pull {t}" for t in ideal_candidates[:target_size] if t.lower() not in installed_set],
    }
    return members, fallback_pool, judge_model, meta


# =============================================================================
# 2. `<think>`-Safe Parsing & ModelFusion Judge Prompt Builder (`judge.rs`)
# =============================================================================

def strip_think_tags(raw_text: str) -> str:
    """Strips `<think>...</think>` reasoning blocks emitted by DeepSeek-R1 / GLM-4.7-Flash."""
    if not raw_text:
        return ""
    end_idx = raw_text.rfind("</think>")
    if end_idx != -1:
        return raw_text[end_idx + len("</think>") :].strip()
    return re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()


def parse_judge_analysis_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parses ModelFusion `JudgeAnalysis` JSON strictly after stripping `<think>` & markdown fences."""
    cleaned = strip_think_tags(raw_text)
    if not cleaned:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    obj = None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            obj = parsed
    except json.JSONDecodeError:
        s_idx = cleaned.find("{")
        e_idx = cleaned.rfind("}")
        if s_idx != -1 and e_idx > s_idx:
            try:
                parsed = json.loads(cleaned[s_idx : e_idx + 1])
                if isinstance(parsed, dict):
                    obj = parsed
            except json.JSONDecodeError:
                obj = None

    if obj is None:
        return None

    def _str_list(val: Any) -> List[str]:
        if not isinstance(val, list):
            return []
        return [str(x).strip() for x in val if str(x).strip()][:12]

    verdict = str(obj.get("judge_verdict") or obj.get("verdict") or "WARN").strip().upper()
    if verdict not in ("PASS", "WARN", "FAIL"):
        verdict = "WARN"

    return {
        "consensus": _str_list(obj.get("consensus")),
        "disagreements": _str_list(obj.get("disagreements")),
        "unique_insights": _str_list(obj.get("unique_insights")),
        "blind_spots": _str_list(obj.get("blind_spots")),
        "risk_flags": _str_list(obj.get("risk_flags")),
        "recommended_final_position": str(obj.get("recommended_final_position") or "").strip(),
        "judge_verdict": verdict,
        "minority_override_triggered": bool(obj.get("minority_override_triggered", False)),
    }


def _build_panel_judge_prompt(
    paper: Dict[str, Any],
    det_checks: Dict[str, Any],
    panelist_results: List[PanelistVerdict],
) -> str:
    """
    Constructs the ModelFusion `judge.rs` panel deliberation prompt incorporating the
    Critical Minority-Quality Rule (never blindly follow majority consensus when a minority
    reviewer cites concrete security or code-alignment evidence).
    """
    responses_block: List[str] = []
    for p in panelist_results:
        if not p.succeeded:
            continue
        responses_block.append(
            f"--- PANELIST #{p.slot_index}: {p.display_name} ---\n"
            f"Verdict: {p.verdict} | Fidelity: {p.paper_fidelity_score}/100 | "
            f"Correctness: {p.correctness_score}/100 | Security: {p.security_score}/100\n"
            f"Summary: {p.summary}\n"
            f"Findings: {json.dumps(p.findings)}\n"
        )

    return (
        "You are the Expert Judge Model in a ModelFusion multi-model code verification panel.\n"
        "Compare the independent code verification audits below for generated Python code implementing "
        f"`arXiv:{paper.get('arxiv_id', 'custom')}` ({paper.get('title', '')}).\n\n"
        f"### LAYER-1 DETERMINISTIC GATE FACTS:\n"
        f"AST Syntax Valid: {det_checks['ast_syntax_valid']} | Tests Passed: "
        f"{det_checks['runtime_tests_passed']}/{det_checks['runtime_tests_total']} | "
        f"HMAC Constant-Time: {det_checks['constant_time_hmac_used']} | "
        f"Zero Dangerous Calls: {det_checks['zero_dangerous_calls']}\n\n"
        "### PANELIST VERIFICATION AUDITS TO EVALUATE:\n"
        + "\n".join(responses_block)
        + "\n### INSTRUCTIONS FOR THE JUDGE:\n"
        "Identify consensus, disagreements, unique insights, blind spots, and risk flags.\n"
        "CRITICAL QUALITY RULE: Do NOT automatically select the majority consensus. If a minority "
        "panelist surfaces concrete evidence of a security flaw, missing paper invariant, or bug, "
        "set `minority_override_triggered: true` and elevate that finding in `risk_flags` and `judge_verdict`.\n"
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "judge_verdict": "PASS" | "WARN" | "FAIL",\n'
        '  "minority_override_triggered": true | false,\n'
        '  "consensus": ["points of agreement across panelists"],\n'
        '  "disagreements": ["points where panelists disagreed"],\n'
        '  "unique_insights": ["distinct technical insights from individual models"],\n'
        '  "blind_spots": ["potential omissions across the panel"],\n'
        '  "risk_flags": ["security, anti-mock, or correctness risk flags"],\n'
        '  "recommended_final_position": "synthesized technical verification judgment"\n'
        "}\n"
    )


def _deterministic_judge_synthesis(
    panelists: List[PanelistVerdict],
    judge_model: str,
) -> JudgeAnalysis:
    """
    Deterministic ModelFusion Judge aggregation used when the dedicated Judge LLM call
    is unavailable or parses invalid JSON, strictly preserving minority CRITICAL/HIGH risk flags.
    """
    ok = [p for p in panelists if p.succeeded]
    if not ok:
        return JudgeAnalysis(
            consensus=[],
            disagreements=[],
            unique_insights=[],
            blind_spots=["No live panel LLM responses available."],
            risk_flags=[],
            recommended_final_position="Verified solely via Layer-1 deterministic AST and runtime test gate.",
            judge_verdict="WARN",
            minority_override_triggered=False,
            judge_model=judge_model,
            judged_by_llm=False,
        )

    verdicts = [p.verdict for p in ok]
    counts = {v: verdicts.count(v) for v in ("PASS", "WARN", "FAIL")}
    majority_v = max(("PASS", "WARN", "FAIL"), key=lambda v: (counts[v], v == "FAIL"))

    consensus: List[str] = []
    disagreements: List[str] = []
    unique_insights: List[str] = []
    risk_flags: List[str] = []

    if len(set(verdicts)) == 1:
        consensus.append(f"Unanimous panel verdict: {verdicts[0]} across all {len(ok)} panelists.")
    else:
        disagreements.append(
            f"Panel verdict split: PASS={counts['PASS']}, WARN={counts['WARN']}, FAIL={counts['FAIL']}."
        )

    # Inspect findings across panelists; check for minority CRITICAL/HIGH override
    has_critical_finding = False
    seen_msgs = set()
    for p in ok:
        if p.summary and p.summary not in seen_msgs:
            seen_msgs.add(p.summary)
            unique_insights.append(f"[{p.model_tag} / {p.persona}] {p.summary}")
        for f in p.findings:
            sev = str(f.get("severity", "INFO")).upper()
            msg = f"[{sev}] ({p.model_tag}) {f.get('message', '')}"
            if sev in ("CRITICAL", "HIGH"):
                risk_flags.append(msg)
            if sev == "CRITICAL":
                has_critical_finding = True

    minority_override = False
    final_v = majority_v
    if majority_v != "FAIL" and (counts["FAIL"] >= 1 and has_critical_finding):
        final_v = "FAIL"
        minority_override = True
        risk_flags.insert(
            0,
            "Minority-Quality Override: Minority panelist reported a CRITICAL defect with FAIL verdict.",
        )
    elif majority_v == "PASS" and counts["FAIL"] >= 1:
        final_v = "WARN"
        minority_override = True
        risk_flags.insert(
            0,
            "Minority-Quality Caution: At least one panelist voted FAIL despite majority PASS.",
        )

    avg_fid = round(sum(p.paper_fidelity_score for p in ok) / len(ok), 2)
    avg_cor = round(sum(p.correctness_score for p in ok) / len(ok), 2)
    avg_sec = round(sum(p.security_score for p in ok) / len(ok), 2)
    consensus.append(
        f"Panel mean scores across {len(ok)} models: Fidelity={avg_fid}/100, Correctness={avg_cor}/100, Security={avg_sec}/100."
    )

    return JudgeAnalysis(
        consensus=consensus,
        disagreements=disagreements,
        unique_insights=unique_insights[:8],
        blind_spots=[],
        risk_flags=risk_flags[:10],
        recommended_final_position=(
            f"Panel consensus verdict '{final_v}' across {len(ok)} reviewers "
            f"(Fidelity {avg_fid}, Correctness {avg_cor}, Security {avg_sec})."
        ),
        judge_verdict=final_v,
        minority_override_triggered=minority_override,
        judge_model=judge_model,
        judged_by_llm=False,
    )


# =============================================================================
# 3. Main Entrypoint: ModelFusion Panel-of-Models Verification Orchestrator
# =============================================================================

def _invoke_panelist(
    member: PanelMemberConfig,
    base_prompt: str,
    endpoint: str,
    timeout_sec: float,
) -> PanelistVerdict:
    t0 = time.perf_counter()
    persona_prompt = (
        f"YOUR ASSIGNED PANEL ROLE: {member.persona}\n"
        f"FOCUS DIRECTIVE: {member.persona_directive}\n\n"
        + base_prompt
    )
    parsed, status, raw_excerpt = _invoke_ollama_llm_judge(
        endpoint=endpoint,
        model_tag=member.model_tag,
        prompt=persona_prompt,
        timeout_sec=timeout_sec,
    )
    if parsed is None and raw_excerpt:
        # Retry parse after stripping `<think>...</think>` blocks for reasoning models (DeepSeek-R1 / GLM-4.7-Flash)
        parsed = parse_llm_verification_json(strip_think_tags(raw_excerpt))
        if parsed is not None:
            status = "llm_verified"

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if parsed is None:
        return PanelistVerdict(
            slot_index=member.slot_index,
            display_name=member.display_name,
            model_tag=member.model_tag,
            persona=member.persona,
            temperature=member.temperature,
            succeeded=False,
            status=status,
            latency_ms=elapsed_ms,
            error=f"Panelist returned fallback status: {status}",
        )

    return PanelistVerdict(
        slot_index=member.slot_index,
        display_name=member.display_name,
        model_tag=member.model_tag,
        persona=member.persona,
        temperature=member.temperature,
        succeeded=True,
        status="llm_verified",
        verdict=parsed["verdict"],
        paper_fidelity_score=parsed["paper_fidelity_score"],
        correctness_score=parsed["correctness_score"],
        security_score=parsed["security_score"],
        summary=parsed["summary"],
        grounded_span_coverage=parsed["grounded_span_coverage"],
        findings=parsed["findings"],
        latency_ms=elapsed_ms,
    )


def run_model_panel_verification(
    source_code: str,
    paper: Dict[str, Any],
    extractions: List[Dict[str, Any]],
    code_stage: str = "synthesized",
    panel_size: Optional[int] = None,
    panel_models: Optional[List[str]] = None,
    panel_mode: str = "multi-model",
    ollama_endpoint: Optional[str] = None,
    dynarot_decision: Optional[DynaRotRoutingDecision] = None,
    timeout_sec: float = 120.0,
    enable_llm: bool = True,
) -> CodeVerificationReport:
    """
    Executes ModelFusion's full Panel-of-Models verification pipeline:
      Stage 0: Layer-1 Deterministic AST, security invariant, anti-mock & unit-test gate
      Stage 1: Memory-batched execution of the assembled multi-model panel (GLM + Qwen + DeepSeek + Gemma)
               with dynamic fallback pool replacement on any failed panelist
      Stage 2: ModelFusion Judge Analysis preserving minority technical/security findings
      Stage 3: Fail-closed final verdict reconciliation
    """
    t0 = time.perf_counter()
    clean_id = normalize_arxiv_id(str(paper.get("arxiv_id", "custom"))) or "custom"
    title = str(paper.get("title", f"arXiv:{clean_id}"))
    endpoint = ollama_endpoint or os.environ.get("LOCAL_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")

    # Stage 0: Layer-1 Deterministic Verification
    det_checks = run_deterministic_code_verification(source_code=source_code, arxiv_id=clean_id)
    hw = detect_live_hardware()
    if dynarot_decision is None:
        dynarot_decision = DynaRotModelRouter(endpoint=endpoint).route(
            prompt=f"Panel verify generated Python code for arXiv:{clean_id}"
        )

    ollama_online, installed_models = probe_ollama_models(endpoint=endpoint, timeout_sec=0.4)
    members, fallback_pool, judge_model, assembly_meta = assemble_verification_panel(
        hw=hw,
        installed_models=installed_models,
        dynarot_decision=dynarot_decision,
        requested_panel_size=panel_size,
        explicit_models=panel_models,
        panel_mode=panel_mode,
    )

    panelist_results: List[PanelistVerdict] = []
    batch_size = int(assembly_meta["batch_size"])
    any_installed_member = any(m.installed for m in members)

    # Stage 1: Batched Panel Execution (only when enable_llm and Ollama has at least one eligible installed model)
    if enable_llm and ollama_online and any_installed_member:
        base_prompt = _build_llm_verification_prompt(paper, extractions, source_code, det_checks)

        for start in range(0, len(members), batch_size):
            batch = members[start : start + batch_size]
            if len(batch) == 1:
                panelist_results.append(_invoke_panelist(batch[0], base_prompt, endpoint, timeout_sec))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    futures = [
                        pool.submit(_invoke_panelist, m, base_prompt, endpoint, timeout_sec)
                        for m in batch
                    ]
                    for fut in futures:
                        panelist_results.append(fut.result())

        # Dynamic fallback replacement (ModelFusion mod.rs lines 499-542)
        fb_iter = iter(fallback_pool)
        for idx, res in enumerate(panelist_results):
            if not res.succeeded:
                replacement_tag = next(fb_iter, None)
                if replacement_tag:
                    rep_member = PanelMemberConfig(
                        slot_index=res.slot_index,
                        display_name=f"{replacement_tag} (fallback for {res.model_tag})",
                        model_tag=replacement_tag,
                        family=_infer_family(replacement_tag),
                        persona=res.persona,
                        persona_directive=PANEL_REVIEWER_PERSONAS[(res.slot_index - 1) % len(PANEL_REVIEWER_PERSONAS)][2],
                        temperature=res.temperature,
                        installed=True,
                    )
                    panelist_results[idx] = _invoke_panelist(rep_member, base_prompt, endpoint, timeout_sec)

    succeeded_panelists = [p for p in panelist_results if p.succeeded]

    # Stage 2: ModelFusion Judge Analysis
    judge_analysis: JudgeAnalysis
    if succeeded_panelists:
        judge_prompt = _build_panel_judge_prompt(paper, det_checks, succeeded_panelists)
        # Use dedicated Judge LLM call via Ollama
        from .dynarot import build_no_proxy_opener
        import urllib.request

        judge_parsed: Optional[Dict[str, Any]] = None
        judge_url = f"{endpoint.rstrip('/')}/api/generate"
        clean_judge = sanitize_model_tag(judge_model)
        judge_body_obj: Dict[str, Any] = {
            "model": clean_judge,
            "prompt": judge_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 900, "num_ctx": 4096},
        }
        if any(k in clean_judge.lower() for k in ("deepseek-r1", "qwq", "glm-4.7")):
            judge_body_obj["think"] = False
        judge_body = json.dumps(judge_body_obj).encode("utf-8")
        try:
            req = urllib.request.Request(
                judge_url,
                data=judge_body,
                method="POST",
                headers={"Content-Type": "application/json", "User-Agent": "ArxivToCode-PanelJudge/1.0"},
            )
            with build_no_proxy_opener().open(req, timeout=timeout_sec) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    judge_parsed = parse_judge_analysis_json(str(data.get("response", "")))
        except Exception:
            judge_parsed = None

        if judge_parsed is not None:
            judge_analysis = JudgeAnalysis(
                consensus=judge_parsed["consensus"],
                disagreements=judge_parsed["disagreements"],
                unique_insights=judge_parsed["unique_insights"],
                blind_spots=judge_parsed["blind_spots"],
                risk_flags=judge_parsed["risk_flags"],
                recommended_final_position=judge_parsed["recommended_final_position"],
                judge_verdict=judge_parsed["judge_verdict"],
                minority_override_triggered=judge_parsed["minority_override_triggered"],
                judge_model=judge_model,
                judged_by_llm=True,
            )
        else:
            judge_analysis = _deterministic_judge_synthesis(succeeded_panelists, judge_model)
    else:
        judge_analysis = _deterministic_judge_synthesis([], judge_model)

    # Aggregate panel scores & findings
    combined_findings: List[Dict[str, str]] = list(det_checks["static_findings"])
    for p in succeeded_panelists:
        for f in p.findings:
            tagged = dict(f)
            tagged["reviewer"] = p.display_name
            combined_findings.append(tagged)

    recommendations = list(
        dict.fromkeys(f["recommendation"] for f in combined_findings if f.get("recommendation"))
    )

    det_hard_fail = (
        (not det_checks["ast_syntax_valid"])
        or (not det_checks["zero_dangerous_calls"])
        or (not det_checks.get("mypy_passed", True))
        or (not det_checks.get("fuzz_passed", True))
        or (not det_checks.get("pytest_passed", True))
        or (det_checks["runtime_error"] is not None)
        or (det_checks["runtime_tests_passed"] < det_checks["runtime_tests_total"])
    )
    det_score = float(det_checks["deterministic_score"])

    llm_verified = len(succeeded_panelists) > 0
    if llm_verified:
        avg_fid = round(sum(p.paper_fidelity_score for p in succeeded_panelists) / len(succeeded_panelists), 2)
        avg_cor = round(sum(p.correctness_score for p in succeeded_panelists) / len(succeeded_panelists), 2)
        avg_sec = round(sum(p.security_score for p in succeeded_panelists) / len(succeeded_panelists), 2)
        llm_avg = (avg_fid + avg_cor + avg_sec) / 3.0
        overall_score = round((0.45 * det_score) + (0.55 * llm_avg), 2)

        # Enforce minority-quality CRITICAL override even if Judge LLM was optimistic
        has_critical_minority_fail = any(
            p.verdict == "FAIL" and any(str(f.get("severity", "")).upper() == "CRITICAL" for f in p.findings)
            for p in succeeded_panelists
        )
        if has_critical_minority_fail and judge_analysis.judge_verdict != "FAIL":
            judge_analysis.judge_verdict = "FAIL"
            judge_analysis.minority_override_triggered = True
            judge_analysis.risk_flags.insert(
                0,
                "Fail-Closed Minority Veto: Panelist reported a CRITICAL defect with FAIL verdict.",
            )

        if det_hard_fail:
            final_verdict = "FAIL"
            overall_score = min(overall_score, 45.0)
        elif judge_analysis.judge_verdict == "FAIL" or overall_score < 55.0:
            final_verdict = "FAIL"
        elif judge_analysis.judge_verdict == "WARN" or overall_score < 75.0 or not det_checks["zero_mock_single_token_indicators"]:
            final_verdict = "WARN"
        else:
            final_verdict = "PASS"

        verification_mode = "panel_llm_and_deterministic_verified"
        verifier_provider = "ollama_panel"
        panel_status = f"panel_verified_{len(succeeded_panelists)}_of_{len(members)}"
        summary_text = judge_analysis.recommended_final_position or (
            f"ModelFusion panel ({len(succeeded_panelists)}/{len(members)} models) judged '{final_verdict}'."
        )
    else:
        avg_fid = avg_cor = avg_sec = 0.0
        overall_score = det_score
        if det_hard_fail:
            final_verdict = "FAIL"
        elif overall_score >= 85.0 and det_checks["zero_mock_single_token_indicators"]:
            final_verdict = "PASS"
        elif overall_score >= 60.0:
            final_verdict = "WARN"
        else:
            final_verdict = "FAIL"

        verification_mode = "deterministic_static_and_runtime_fallback"
        verifier_provider = "deterministic_fallback"
        if not enable_llm:
            panel_status = "panel_llm_disabled_by_caller"
            summary_text = "Panel LLM execution disabled; verified via Layer-1 deterministic gate."
        elif not ollama_online:
            panel_status = "ollama_offline_fallback"
            summary_text = (
                f"Local Ollama ({endpoint}) is offline; verified via Layer-1 deterministic gate. "
                f"Pull panel models: {', '.join(assembly_meta['recommended_panel_pulls'][:3])}"
            )
        elif any_installed_member and panelist_results:
            err_details = ", ".join(
                f"{p.model_tag}: {p.status} ({round(p.latency_ms / 1000.0, 1)}s)"
                for p in panelist_results
            )
            panel_status = "panel_llm_timeout_or_error_fallback"
            summary_text = (
                f"Ollama is online and panel models are installed, but live panel calls fell back to "
                f"Layer-1 deterministic verification ({err_details})."
            )
        else:
            panel_status = "panel_no_chat_models_installed_fallback"
            summary_text = (
                f"Ollama is online (installed: {installed_models}) but no memory-eligible chat/completion "
                f"panel models are pulled yet. Run: {' && '.join(assembly_meta['recommended_panel_pulls'][:2])}"
            )

    llm_audit: Dict[str, Any] = {
        "status": panel_status,
        "verdict": judge_analysis.judge_verdict if llm_verified else "N/A",
        "paper_fidelity_score": avg_fid,
        "correctness_score": avg_cor,
        "security_score": avg_sec,
        "summary": summary_text,
        "grounded_span_coverage": list(
            dict.fromkeys(span for p in succeeded_panelists for span in p.grounded_span_coverage)
        ),
        "findings": [f for p in succeeded_panelists for f in p.findings],
        "ollama_guidance": {
            "ollama_endpoint": endpoint,
            "ollama_online": ollama_online,
            "installed_models": installed_models,
            "eligible_chat_models": assembly_meta["eligible_installed_chat_models"],
            "recommended_panel_pulls": assembly_meta["recommended_panel_pulls"],
            "pull_command": (
                assembly_meta["recommended_panel_pulls"][0]
                if assembly_meta["recommended_panel_pulls"]
                else f"ollama pull {members[0].model_tag}"
            ),
        },
        "panel_audit": {
            "panel_mode": assembly_meta["panel_mode"],
            "derived_max_panel_count": assembly_meta["derived_max_panel_count"],
            "effective_panel_size": len(members),
            "responded_panelists": len(succeeded_panelists),
            "batch_size": batch_size,
            "glm_panel_assessment": assembly_meta["glm_panel_assessment"],
            "panel_roster": [asdict(m) for m in members],
            "panelist_responses": [p.to_dict() for p in panelist_results],
            "judge_model": judge_model,
            "judge_analysis": judge_analysis.to_dict(),
        },
    }

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    panel_label = " + ".join(dict.fromkeys(m.model_tag for m in members))

    return CodeVerificationReport(
        arxiv_id=clean_id,
        title=title,
        code_stage=code_stage,
        verdict=final_verdict,
        overall_score=overall_score,
        verification_mode=verification_mode,
        llm_verified=llm_verified,
        verifier_model=f"ModelFusion Panel ({panel_label}) -> Judge:{judge_model}",
        verifier_provider=verifier_provider,
        deterministic_checks=det_checks,
        llm_audit=llm_audit,
        findings=combined_findings,
        recommendations=recommendations,
        duration_ms=duration_ms,
    )
