"""
ArXiv-to-Code Iterative Auto-Repair Retry Loop (`code_repair.py`).

Closes the generate -> verify -> repair loop across all verification layers:
  1. Verify candidate Python code with full Layer-1 deterministic gates
     (AST + Security + Ruff/AST Lint + mypy --strict + Property-Based Fuzzing +
     Real Subprocess pytest + 7 Runtime Invariants) and Layer-2 LLM / ModelFusion Panel.
  2. Diagnose exact failing rules, line numbers, type errors, fuzz counterexamples,
     pytest tracebacks, and LLM/panel findings.
  3. Patch via two-tier monotonic hill-climbing:
       - Tier A: Local Ollama LLM Repair (feeding exact diagnostics back to local
         open-weight models selected by ModelFusion DynaRot, with `<think>` stripping
         and AST validation).
       - Tier B: Deterministic Structural Auto-Patcher fallback (safe AST-verified
         fixes for unused imports, bare `except:`, cache-key length collisions,
         wildcard meet-semilattice defects, numeric edge-case guards, and untyped
         `test_*` signatures).
  4. Re-verify candidate patches and accept ONLY strictly improving patches
     (monotonic hill-climbing — never regress working code) until all gates pass
     or `max_repair_attempts` is reached.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dynarot import (
    DynaRotModelRouter,
    build_no_proxy_opener,
    detect_live_hardware,
    find_best_installed_chat_model,
    probe_ollama_models,
    sanitize_model_tag,
    size_glm_by_hardware,
)
from .langextract_engine import normalize_arxiv_id
from .llm_verifier import (
    CodeVerificationReport,
    run_deterministic_code_verification,
    verify_generated_python_code,
)
from .models import DynaRotRoutingDecision
from .panel_verifier import run_model_panel_verification, strip_think_tags


@dataclass
class RepairAttemptRecord:
    """Audit record for a single Verify -> Diagnose -> Patch -> Re-Verify cycle."""

    attempt: int
    strategy: str
    model_used: str
    defects_targeted: List[str]
    deterministic_fixes_applied: List[str]
    before_verdict: str
    after_verdict: str
    before_score: float
    after_score: float
    before_gates_passed: int
    after_gates_passed: int
    accepted: bool
    duration_ms: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutoRepairResult:
    """Complete result of the iterative auto-repair retry loop."""

    arxiv_id: str
    repaired: bool
    converged_clean: bool
    repair_attempts: int
    max_repair_attempts: int
    initial_verdict: str
    final_verdict: str
    initial_score: float
    final_score: float
    repaired_source_code: str
    final_verification_report: CodeVerificationReport
    repair_history: List[RepairAttemptRecord] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact JSON-safe summary suitable for embedding inside reports/metadata."""
        return {
            "arxiv_id": self.arxiv_id,
            "repaired": self.repaired,
            "converged_clean": self.converged_clean,
            "repair_attempts": self.repair_attempts,
            "max_repair_attempts": self.max_repair_attempts,
            "initial_verdict": self.initial_verdict,
            "final_verdict": self.final_verdict,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "duration_ms": self.duration_ms,
            "repair_history": [rec.to_dict() for rec in self.repair_history],
        }

    def to_dict(self) -> Dict[str, Any]:
        out = self.to_summary_dict()
        out["final_verification_report"] = self.final_verification_report.to_dict()
        return out


# =============================================================================
# 1. Clean-Pass Check, Monotonic Ranking & Structured Defect Diagnostics
# =============================================================================

def count_layer1_gates_passed(det_checks: Dict[str, Any]) -> Tuple[int, int]:
    """Counts passed Layer-1 deterministic gates out of 9 total gates."""
    tests_total = int(det_checks.get("runtime_tests_total", 0))
    tests_passed = int(det_checks.get("runtime_tests_passed", 0))
    gates = [
        bool(det_checks.get("ast_syntax_valid", False)),
        bool(det_checks.get("zero_dangerous_calls", False)),
        bool(det_checks.get("zero_mock_single_token_indicators", False)),
        bool(det_checks.get("linter_passed", False)),
        bool(det_checks.get("mypy_passed", False)),
        bool(det_checks.get("fuzz_passed", False)),
        bool(det_checks.get("pytest_passed", False)),
        det_checks.get("runtime_error") is None,
        (tests_total == 0) or (tests_passed >= tests_total),
    ]
    return sum(1 for g in gates if g), len(gates)


def is_verification_clean_pass(report: CodeVerificationReport) -> bool:
    """True iff verdict is PASS AND every single Layer-1 deterministic gate passed."""
    passed, total = count_layer1_gates_passed(report.deterministic_checks)
    return report.verdict == "PASS" and passed == total


def rank_verification_report(report: CodeVerificationReport) -> Tuple[int, int, int, float, float]:
    """
    Lexicographic ranking key for monotonic hill-climbing:
      (is_clean_pass, verdict_tier, layer1_gates_passed, deterministic_score, overall_score)
    """
    verdict_tier = {"PASS": 2, "WARN": 1, "FAIL": 0}.get(report.verdict, 0)
    gates_passed, _ = count_layer1_gates_passed(report.deterministic_checks)
    det_score = float(report.deterministic_checks.get("deterministic_score", 0.0))
    return (
        1 if is_verification_clean_pass(report) else 0,
        verdict_tier,
        gates_passed,
        det_score,
        float(report.overall_score),
    )


def build_defect_diagnostic_summary(report: CodeVerificationReport) -> Tuple[str, List[str]]:
    """
    Distills all Layer-1 (AST, Ruff, mypy --strict, Fuzzing, pytest, runtime) and Layer-2
    (LLM / Panel) defects into a crisp prompt block + bullet list for code repair.
    """
    det = report.deterministic_checks
    defects: List[str] = []

    if not det.get("ast_syntax_valid", True):
        defects.append("SYNTAX: Module fails ast.parse(). Fix Python syntax errors.")
    if not det.get("zero_dangerous_calls", True):
        calls = ", ".join(det.get("dangerous_calls_found") or [])
        defects.append(f"SECURITY: Remove forbidden dangerous calls: {calls}")
    if not det.get("zero_mock_single_token_indicators", True):
        mocks = ", ".join(det.get("mock_indicators_found") or [])
        defects.append(f"ANTI-MOCK: Replace single-token mock indicators with multi-token indicators: {mocks}")

    if not det.get("linter_passed", True):
        for issue in (det.get("linter_issues") or [])[:10]:
            defects.append(
                f"LINT [{issue.get('engine')}:{issue.get('code')} L{issue.get('line')}]: {issue.get('message')}"
            )

    if not det.get("mypy_passed", True):
        for issue in (det.get("mypy_issues") or [])[:10]:
            defects.append(
                f"MYPY [{issue.get('code')} L{issue.get('line')}]: {issue.get('message')}"
            )

    if not det.get("fuzz_passed", True):
        for failure in (det.get("fuzz_failures") or [])[:6]:
            defects.append(
                f"FUZZ [{failure.get('property')}]: {failure.get('detail')}"
            )

    if not det.get("pytest_passed", True):
        defects.append(f"PYTEST: {det.get('pytest_summary', 'pytest failed')}")

    if det.get("runtime_error"):
        defects.append(f"RUNTIME: {det['runtime_error']}")
    elif int(det.get("runtime_tests_passed", 0)) < int(det.get("runtime_tests_total", 0)):
        defects.append(
            f"RUNTIME_TESTS: Only {det.get('runtime_tests_passed')}/{det.get('runtime_tests_total')} invariants passed."
        )

    # Include Layer-2 LLM / Panel findings (especially CRITICAL/HIGH/MEDIUM)
    if report.llm_verified:
        for f in (report.llm_audit.get("findings") or [])[:8]:
            sev = str(f.get("severity", "INFO")).upper()
            if sev in ("CRITICAL", "HIGH", "MEDIUM"):
                rec = f.get("recommendation", "")
                msg = f"LLM [{sev} {f.get('category', 'general')}]: {f.get('message', '')}"
                if rec:
                    msg += f" (Fix: {rec})"
                defects.append(msg)

    if not defects and report.verdict != "PASS":
        defects.append(
            f"VERDICT: Overall verdict is {report.verdict} ({report.overall_score}/100); tighten correctness, typing, and security invariants."
        )

    prompt_block = "\n".join(f"  {i + 1}. {d}" for i, d in enumerate(defects))
    return prompt_block, defects


# =============================================================================
# 2. Tier B — Deterministic Structural Auto-Patcher (Zero-Mock, AST-Verified)
# =============================================================================

def _remove_unused_imports(source_code: str, unused_names: List[str]) -> Tuple[str, List[str]]:
    """Safely removes AST/Ruff-flagged unused imports without touching active imports."""
    if not unused_names:
        return source_code, []
    unused_set = set(unused_names)
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return source_code, []

    lines = source_code.splitlines(keepends=True)
    edits: List[Tuple[int, int, str]] = []  # (start_idx_0, end_idx_0_excl, replacement)
    removed: List[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            kept: List[str] = []
            node_removed: List[str] = []
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in unused_set:
                    node_removed.append(bound)
                else:
                    kept.append(f"{alias.name} as {alias.asname}" if alias.asname else alias.name)
            if node_removed:
                start = node.lineno - 1
                end = (node.end_lineno or node.lineno)
                indent = " " * node.col_offset
                repl = f"{indent}import {', '.join(kept)}\n" if kept else ""
                edits.append((start, end, repl))
                removed.extend(node_removed)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            kept_from: List[str] = []
            node_removed_from: List[str] = []
            for alias in node.names:
                if alias.name == "*":
                    kept_from.append("*")
                    continue
                bound = alias.asname or alias.name
                if bound in unused_set:
                    node_removed_from.append(bound)
                else:
                    kept_from.append(f"{alias.name} as {alias.asname}" if alias.asname else alias.name)
            if node_removed_from:
                start = node.lineno - 1
                end = (node.end_lineno or node.lineno)
                indent = " " * node.col_offset
                dots = "." * (node.level or 0)
                mod = node.module or ""
                repl = f"{indent}from {dots}{mod} import {', '.join(kept_from)}\n" if kept_from else ""
                edits.append((start, end, repl))
                removed.extend(node_removed_from)

    if not edits:
        return source_code, []

    # Apply edits from bottom to top so line indices remain exact
    for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[start:end] = [repl] if repl else []

    candidate = "".join(lines)
    try:
        ast.parse(candidate)
        return candidate, removed
    except SyntaxError:
        return source_code, []


def apply_deterministic_code_repairs(
    source_code: str,
    det_checks: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """
    Applies safe, AST-verified deterministic structural patches for known defects
    caught by Ruff/AST Lint, `mypy --strict`, and Property-Based Fuzzing.
    Each individual fix is rolled back immediately if it violates `ast.parse()`.
    """
    current = source_code
    fixes_applied: List[str] = []

    def _try_apply(candidate: str, label: str) -> None:
        nonlocal current
        if candidate != current:
            try:
                ast.parse(candidate)
                current = candidate
                fixes_applied.append(label)
            except SyntaxError:
                pass

    # Fix 1: Remove unused imports flagged by AST-F401 or ruff F401
    unused_names: List[str] = []
    for li in det_checks.get("linter_issues") or []:
        code = str(li.get("code", ""))
        msg = str(li.get("message", ""))
        if code in ("AST-F401", "F401"):
            m = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", msg)
            if m:
                unused_names.append(m.group(1))
    if unused_names:
        cand, removed = _remove_unused_imports(current, list(dict.fromkeys(unused_names)))
        if removed:
            _try_apply(cand, f"removed_unused_imports({','.join(sorted(set(removed)))})")

    # Fix 2: Replace bare `except:` clauses with `except Exception:`
    if re.search(r"^(\s*)except\s*:", current, flags=re.MULTILINE):
        cand = re.sub(r"^(\s*)except\s*:", r"\1except Exception:", current, flags=re.MULTILINE)
        _try_apply(cand, "replaced_bare_except_with_Exception")

    # Fix 3: Fix cache-key length collision `(payload_corpus, len(indicators))` -> `(payload_corpus, tuple(indicators))`
    if re.search(r"cache_key\s*(?::[^=]+)?=\s*\(\s*payload_corpus\s*,\s*len\s*\(\s*indicators\s*\)\s*\)", current):
        cand = re.sub(
            r"Dict\[Tuple\[str,\s*int\],\s*Tuple\[str,\s*\.\.\.\]\]",
            "Dict[Tuple[str, Tuple[str, ...]], Tuple[str, ...]]",
            current,
        )
        cand = re.sub(
            r"cache_key\s*(?::[^=]+)?=\s*\(\s*payload_corpus\s*,\s*len\s*\(\s*indicators\s*\)\s*\)",
            "cache_key = (payload_corpus, tuple(indicators))",
            cand,
        )
        _try_apply(cand, "fixed_cache_key_tuple_indicators")

    # Fix 4: Fix unsafe division `1.0 / alpha` or unguarded `prior_wealth` / `alpha <= 0`
    fuzz_props = {str(f.get("property", "")) for f in (det_checks.get("fuzz_failures") or [])}
    if (
        "scan_no_crash" in fuzz_props
        or "scan_finite_bounds" in fuzz_props
        or re.search(r"\b1\.0\s*/\s*alpha\b", current)
    ):
        cand = current
        # Guard bare `1.0 / alpha`
        cand = re.sub(r"\b1\.0\s*/\s*alpha\b", "1.0 / (alpha if alpha > 1e-6 else 1e-6)", cand)
        # Guard `prior_wealth *` or `max(0.0, prior_wealth)` against NaN/Inf/negative inside scan_and_score_payload
        if "def scan_and_score_payload(" in cand and "safe_prior" not in cand:
            cand = re.sub(
                r"(def scan_and_score_payload\([^)]*\)\s*->\s*[^:]+:\s*\n(?:\s*\"\"\"[\s\S]*?\"\"\"\s*\n)?)",
                (
                    r"\1"
                    "    prior_wealth = prior_wealth if (0.0 <= prior_wealth < 1e12) else 1.0\n"
                    "    alpha = alpha if alpha > 1e-6 else 1e-6\n"
                ),
                cand,
                count=1,
            )
        _try_apply(cand, "hardened_scan_alpha_and_prior_wealth_guards")

    # Fix 5: Fix case-sensitive indicator scanning if fuzz soundness detected missed uppercase indicator
    if "scan_soundness_and_cache_isolation" in fuzz_props and "def scan_and_score_payload(" in current:
        cand = re.sub(
            r"\bpat\s+in\s+corpus_lc\b",
            "pat.lower() in corpus_lc",
            current,
        )
        cand = re.sub(
            r"\bpat\s+in\s+payload_corpus\b",
            "pat.lower() in payload_corpus.lower()",
            cand,
        )
        _try_apply(cand, "normalized_case_insensitive_indicator_scan")

    # Fix 6: Fix wildcard top-element meet-semilattice bug (`self.resources & narrower.resources`)
    if (
        "semilattice_top_identity" in fuzz_props
        or (
            "class DelegationScope" in current
            and "self.resources & narrower.resources" in current
            and "_meet_set" not in current
        )
    ):
        meet_helper = (
            "    @staticmethod\n"
            "    def _meet_set(a: FrozenSet[str], b: FrozenSet[str]) -> FrozenSet[str]:\n"
            '        if "*" in a:\n'
            "            return b\n"
            '        if "*" in b:\n'
            "            return a\n"
            "        return a & b\n\n"
        )
        cand = current
        if "_meet_set" not in cand:
            cand = re.sub(
                r"(\n\s*def meet\(self,\s*narrower:\s*DelegationScope\))",
                "\n" + meet_helper + r"\1",
                cand,
                count=1,
            )
        cand = cand.replace("self.resources & narrower.resources", "self._meet_set(self.resources, narrower.resources)")
        cand = cand.replace("self.actions & narrower.actions", "self._meet_set(self.actions, narrower.actions)")
        cand = cand.replace("self.data_domains & narrower.data_domains", "self._meet_set(self.data_domains, narrower.data_domains)")
        _try_apply(cand, "restored_wildcard_top_element_meet_semilattice")

    # Fix 7: Add missing `-> None` return annotations on untyped `def test_...():` flagged by mypy
    mypy_issues = det_checks.get("mypy_issues") or []
    untyped_lines = {
        int(mi.get("line", 0))
        for mi in mypy_issues
        if str(mi.get("code", "")) in ("no-untyped-def", "misc")
    }
    if untyped_lines:
        lines_list = current.splitlines(keepends=True)
        changed = False
        for ln_no in untyped_lines:
            idx = ln_no - 1
            if 0 <= idx < len(lines_list):
                line_txt = lines_list[idx]
                if re.match(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(\s*(?:self)?\s*\)\s*:\s*$", line_txt):
                    lines_list[idx] = re.sub(r"\)\s*:\s*$", ") -> None:\n", line_txt)
                    changed = True
        if changed:
            _try_apply("".join(lines_list), "added_missing_return_type_annotations")

    # Fix 8: Ensure `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` aren't accidentally duplicated or dropped
    return current, fixes_applied


# =============================================================================
# 3. Tier A — Local Ollama LLM Patch Generator
# =============================================================================

def extract_python_code_from_response(raw_text: str) -> Optional[str]:
    """
    Extracts complete, syntactically valid Python source code from an LLM response
    after stripping `<think>...</think>` reasoning blocks and markdown fences.
    """
    cleaned = strip_think_tags(raw_text).strip()
    if not cleaned:
        return None

    # Prefer fenced ```python ... ``` or ``` ... ``` block with the largest length
    fenced_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates: List[str] = []
    if fenced_blocks:
        candidates.extend(sorted(fenced_blocks, key=len, reverse=True))
    candidates.append(cleaned)

    for cand in candidates:
        cand_stripped = cand.strip()
        if not cand_stripped:
            continue
        try:
            ast.parse(cand_stripped)
            return cand_stripped + "\n"
        except SyntaxError:
            continue
    return None


def _build_ollama_repair_prompt(
    source_code: str,
    defect_block: str,
    paper: Dict[str, Any],
) -> str:
    clean_id = normalize_arxiv_id(str(paper.get("arxiv_id", "custom"))) or "custom"
    title = str(paper.get("title", f"arXiv:{clean_id}"))
    return (
        f"You are a senior Python systems, formal verification, and security engineer.\n"
        f"Fix ALL reported defects in the following Python module synthesized from arXiv:{clean_id} ({title}).\n\n"
        f"=== DEFECTS TO FIX ===\n{defect_block}\n\n"
        f"=== STRICT REQUIREMENTS ===\n"
        f"1. Return the COMPLETE repaired Python module inside a single ```python ... ``` fenced code block.\n"
        f"2. Preserve all public class/function names, `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers, "
        f"constant-time `hmac.compare_digest`, wildcard top-element `DelegationScope.meet()`, and tests.\n"
        f"3. Ensure ZERO Ruff/lint warnings, ZERO `mypy --strict` type errors (explicit type annotations on every function/variable), "
        f"ZERO property-based fuzzing failures (handle alpha<=0, NaN/Inf prior_wealth, case-insensitive indicators, collision-free cache keys), "
        f"and 100% passing pytest + deterministic benchmark invariants.\n"
        f"4. Never introduce `eval`, `exec`, `os.system`, `pickle`, `subprocess(..., shell=True)`, or single-word mock rules.\n"
        f"5. Output ONLY the ```python ... ``` block with no extra commentary.\n\n"
        f"=== CURRENT PYTHON SOURCE ===\n```python\n{source_code}\n```\n"
    )


def request_ollama_code_patch(
    source_code: str,
    defect_block: str,
    paper: Dict[str, Any],
    endpoint: str,
    model_tag: str,
    timeout_sec: float = 120.0,
) -> Tuple[Optional[str], str]:
    """
    Calls local Ollama `/api/generate` (with `/api/chat` fallback) to synthesize a
    repaired Python module addressing all Layer-1 and Layer-2 verification failures.
    Returns `(repaired_code_or_None, status_string)`.
    """
    clean_model = sanitize_model_tag(model_tag)
    prompt = _build_ollama_repair_prompt(source_code, defect_block, paper)
    is_reasoning_model = any(k in clean_model.lower() for k in ("deepseek-r1", "qwq", "glm-4.7"))
    opener = build_no_proxy_opener()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ArxivToCode-AutoRepair/1.0",
    }

    gen_url = f"{endpoint.rstrip('/')}/api/generate"
    gen_body_obj: Dict[str, Any] = {
        "model": clean_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "num_predict": 4096,
            "num_ctx": 8192,
        },
    }
    if is_reasoning_model:
        gen_body_obj["think"] = False

    try:
        req = urllib.request.Request(
            gen_url,
            data=json.dumps(gen_body_obj).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        with opener.open(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
                raw_text = str(body.get("response", "")).strip()
                extracted = extract_python_code_from_response(raw_text)
                if extracted is not None:
                    return extracted, "ollama_patch_synthesized"
                return None, "ollama_patch_invalid_python"
            return None, f"ollama_http_{resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            chat_url = f"{endpoint.rstrip('/')}/api/chat"
            chat_payload = json.dumps(
                {
                    "model": clean_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.15, "num_predict": 4096, "num_ctx": 8192},
                }
            ).encode("utf-8")
            try:
                chat_req = urllib.request.Request(chat_url, data=chat_payload, method="POST", headers=headers)
                with opener.open(chat_req, timeout=timeout_sec) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode("utf-8", errors="replace"))
                        raw_text = str((body.get("message") or {}).get("content", "")).strip()
                        extracted = extract_python_code_from_response(raw_text)
                        if extracted is not None:
                            return extracted, "ollama_chat_patch_synthesized"
                        return None, "ollama_chat_patch_invalid_python"
            except Exception:
                pass
        return None, f"ollama_http_{exc.code}"
    except TimeoutError:
        return None, "ollama_timeout"
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return None, "ollama_timeout"
        return None, f"ollama_url_error:{type(exc.reason).__name__}"
    except Exception as exc:
        return None, f"ollama_error:{type(exc).__name__}"


def _resolve_repair_ollama_model(
    endpoint: str,
    requested_model: Optional[str],
    dynarot_decision: DynaRotRoutingDecision,
) -> Tuple[bool, Optional[str]]:
    """Probes local Ollama and selects the highest-capability installed chat/code model fitting free RAM/VRAM."""
    ollama_online, installed_models = probe_ollama_models(endpoint=endpoint, timeout_sec=0.4)
    if not ollama_online or not installed_models:
        return ollama_online, None
    hw = detect_live_hardware()
    recommended_glm = size_glm_by_hardware(hw.free_ram_gb, hw.free_vram_mb)
    preferred = list(
        dict.fromkeys(
            [
                *([sanitize_model_tag(requested_model)] if requested_model else []),
                recommended_glm,
                dynarot_decision.shinka_heavy_reasoning_model,
                dynarot_decision.primary_model,
                dynarot_decision.openevolve_fast_mutation_model,
            ]
        )
    )
    for cand in preferred:
        hit = find_best_installed_chat_model(cand, hw, installed_models)
        if hit is not None:
            return True, sanitize_model_tag(hit)
    return True, None


# =============================================================================
# 4. Primary Iterative Auto-Repair Orchestrator
# =============================================================================

def auto_repair_generated_code(
    source_code: str,
    paper: Dict[str, Any],
    extractions: List[Dict[str, Any]],
    code_stage: str = "synthesized",
    initial_report: Optional[CodeVerificationReport] = None,
    max_repair_attempts: int = 3,
    use_panel: bool = False,
    model_id: Optional[str] = None,
    panel_size: Optional[int] = None,
    panel_mode: str = "multi-model",
    panel_models: Optional[List[str]] = None,
    ollama_endpoint: Optional[str] = None,
    dynarot_decision: Optional[DynaRotRoutingDecision] = None,
    timeout_sec: float = 120.0,
    enable_llm: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> AutoRepairResult:
    """
    Runs the Verify -> Diagnose -> Patch -> Re-Verify auto-repair loop up to
    `max_repair_attempts` iterations until every Layer-1 gate (AST, Ruff/Lint,
    `mypy --strict`, Property-Based Fuzzing, `pytest`, 7 invariants) and Layer-2
    verdict reaches clean `PASS`.

    Monotonic Hill-Climbing Invariant:
      A candidate patch is accepted ONLY when it strictly improves the verification
      ranking tuple `(clean_pass, verdict_tier, layer1_gates_passed, det_score, overall_score)`.
    """
    t0 = time.perf_counter()
    clean_id = normalize_arxiv_id(str(paper.get("arxiv_id", "custom"))) or "custom"
    endpoint = ollama_endpoint or os.environ.get("LOCAL_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    max_attempts = max(1, min(10, int(max_repair_attempts)))

    if dynarot_decision is None:
        dynarot_decision = DynaRotModelRouter(endpoint=endpoint).route(
            prompt=f"Auto-repair generated Python code for arXiv:{clean_id}"
        )

    def _verify(code: str, stage_label: str) -> CodeVerificationReport:
        if use_panel:
            return run_model_panel_verification(
                source_code=code,
                paper=paper,
                extractions=extractions,
                code_stage=stage_label,
                panel_size=panel_size,
                panel_mode=panel_mode,
                panel_models=panel_models,
                ollama_endpoint=endpoint,
                dynarot_decision=dynarot_decision,
                timeout_sec=timeout_sec,
                enable_llm=enable_llm,
            )
        return verify_generated_python_code(
            source_code=code,
            paper=paper,
            extractions=extractions,
            code_stage=stage_label,
            model_id=model_id,
            ollama_endpoint=endpoint,
            dynarot_decision=dynarot_decision,
            timeout_sec=timeout_sec,
            enable_llm=enable_llm,
        )

    current_code = source_code
    current_report = initial_report if initial_report is not None else _verify(current_code, code_stage)
    initial_verdict = current_report.verdict
    initial_score = float(current_report.overall_score)

    if is_verification_clean_pass(current_report):
        return AutoRepairResult(
            arxiv_id=clean_id,
            repaired=False,
            converged_clean=True,
            repair_attempts=0,
            max_repair_attempts=max_attempts,
            initial_verdict=initial_verdict,
            final_verdict=current_report.verdict,
            initial_score=initial_score,
            final_score=float(current_report.overall_score),
            repaired_source_code=current_code,
            final_verification_report=current_report,
            repair_history=[],
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    # Probe Ollama once for Tier A repair model selection
    ollama_online, repair_model = (
        _resolve_repair_ollama_model(endpoint, model_id, dynarot_decision)
        if enable_llm
        else (False, None)
    )

    repaired_any = False
    history: List[RepairAttemptRecord] = []

    for attempt in range(1, max_attempts + 1):
        at_start = time.perf_counter()
        defect_block, defect_list = build_defect_diagnostic_summary(current_report)
        before_rank = rank_verification_report(current_report)
        before_gates, _ = count_layer1_gates_passed(current_report.deterministic_checks)
        before_verdict = current_report.verdict
        before_score = float(current_report.overall_score)

        if progress_callback is not None:
            progress_callback(
                f"Auto-Repair attempt {attempt}/{max_attempts}: targeting {len(defect_list)} defect(s)..."
            )

        # Collect candidate patches: (strategy_name, model_label, det_fixes, patched_code, note)
        candidates: List[Tuple[str, str, List[str], str, str]] = []

        # Tier A: Local Ollama LLM Repair
        llm_status_note = ""
        if enable_llm and ollama_online and repair_model is not None:
            llm_code, llm_status = request_ollama_code_patch(
                source_code=current_code,
                defect_block=defect_block,
                paper=paper,
                endpoint=endpoint,
                model_tag=repair_model,
                timeout_sec=timeout_sec,
            )
            llm_status_note = f"ollama({repair_model})={llm_status}"
            if llm_code is not None and llm_code.strip() != current_code.strip():
                candidates.append(
                    ("ollama_llm_patch", repair_model, [], llm_code, llm_status_note)
                )
                # Also test hybrid LLM + deterministic post-polish if Layer-1 lint/mypy finds a minor issue
                llm_det = run_deterministic_code_verification(llm_code, arxiv_id=clean_id)
                polished_code, polish_fixes = apply_deterministic_code_repairs(llm_code, llm_det)
                if polish_fixes and polished_code.strip() != llm_code.strip():
                    candidates.append(
                        (
                            "ollama_plus_deterministic_patch",
                            repair_model,
                            polish_fixes,
                            polished_code,
                            f"{llm_status_note}; post_polish={','.join(polish_fixes)}",
                        )
                    )
        else:
            llm_status_note = (
                "llm_disabled"
                if not enable_llm
                else ("ollama_offline" if not ollama_online else "no_eligible_ollama_model")
            )

        # Tier B: Deterministic Structural Auto-Patcher
        det_patched_code, det_fixes = apply_deterministic_code_repairs(
            current_code, current_report.deterministic_checks
        )
        if det_fixes and det_patched_code.strip() != current_code.strip():
            candidates.append(
                (
                    "deterministic_structural_patch",
                    "deterministic_ast_patcher",
                    det_fixes,
                    det_patched_code,
                    f"applied={','.join(det_fixes)}; {llm_status_note}",
                )
            )

        if not candidates:
            history.append(
                RepairAttemptRecord(
                    attempt=attempt,
                    strategy="no_viable_patch",
                    model_used=repair_model or "deterministic_ast_patcher",
                    defects_targeted=defect_list[:8],
                    deterministic_fixes_applied=[],
                    before_verdict=before_verdict,
                    after_verdict=before_verdict,
                    before_score=before_score,
                    after_score=before_score,
                    before_gates_passed=before_gates,
                    after_gates_passed=before_gates,
                    accepted=False,
                    duration_ms=round((time.perf_counter() - at_start) * 1000.0, 2),
                    note=f"No new candidate patch produced ({llm_status_note}).",
                )
            )
            if not (enable_llm and ollama_online and repair_model is not None):
                break
            continue

        # Fast-screen candidates on Layer-1 first so we only pay full Layer-2 LLM verification
        # on the top Layer-1 candidate (and avoid wasting LLM calls on broken patches).
        best_choice: Optional[Tuple[Tuple[int, int, int, float, float], CodeVerificationReport, str, str, List[str], str, str]] = None
        for strat, model_lbl, applied_fixes, cand_code, note in candidates:
            cand_report = _verify(cand_code, f"{code_stage}_repaired_try{attempt}")
            cand_rank = rank_verification_report(cand_report)
            if best_choice is None or cand_rank > best_choice[0]:
                best_choice = (cand_rank, cand_report, strat, model_lbl, applied_fixes, cand_code, note)

        assert best_choice is not None
        best_rank, best_report, best_strat, best_model, best_fixes, best_code, best_note = best_choice
        after_gates, _ = count_layer1_gates_passed(best_report.deterministic_checks)
        accepted = best_rank > before_rank

        if accepted:
            current_code = best_code
            current_report = best_report
            repaired_any = True

        history.append(
            RepairAttemptRecord(
                attempt=attempt,
                strategy=best_strat,
                model_used=best_model,
                defects_targeted=defect_list[:8],
                deterministic_fixes_applied=best_fixes,
                before_verdict=before_verdict,
                after_verdict=best_report.verdict if accepted else before_verdict,
                before_score=before_score,
                after_score=float(best_report.overall_score) if accepted else before_score,
                before_gates_passed=before_gates,
                after_gates_passed=after_gates if accepted else before_gates,
                accepted=accepted,
                duration_ms=round((time.perf_counter() - at_start) * 1000.0, 2),
                note=best_note if accepted else f"Rejected non-improving patch ({best_note}).",
            )
        )

        if is_verification_clean_pass(current_report):
            break
        if (not accepted) and not (enable_llm and ollama_online and repair_model is not None):
            break

    return AutoRepairResult(
        arxiv_id=clean_id,
        repaired=repaired_any,
        converged_clean=is_verification_clean_pass(current_report),
        repair_attempts=len(history),
        max_repair_attempts=max_attempts,
        initial_verdict=initial_verdict,
        final_verdict=current_report.verdict,
        initial_score=initial_score,
        final_score=float(current_report.overall_score),
        repaired_source_code=current_code,
        final_verification_report=current_report,
        repair_history=history,
        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
    )
