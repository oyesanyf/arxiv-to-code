"""
LLM-Powered Verification Engine for Generated & Evolved Python Code (`llm_verifier.py`).

Combines two fail-closed verification layers:
  1. Layer 1 — Deterministic Static AST, Security Invariant & Runtime Benchmark Gate:
     - Parses Python AST syntax (`ast.parse`)
     - Verifies `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers
     - Verifies constant-time cryptographic verification (`hmac.compare_digest`)
     - Verifies asymmetric meet-semilattice delegation narrowing (`DelegationScope.meet`)
     - Blocks dangerous execution sinks (`eval`, `exec`, `os.system`, `subprocess(..., shell=True)`, `pickle.loads`)
     - Blocks naive single-token mock indicators (`powershell.exe`, `python.exe`, `bash`, `sh`, `cmd.exe`)
     - Executes `evaluate_python_candidate()` to verify real runtime unit tests (`4/4`) and metrics
  2. Layer 2 — Open-Weight LLM-as-a-Judge Verification (Local Ollama via ModelFusion DynaRot,
     or optional Gemini REST API when explicitly requested):
     - Audits the generated/evolved Python module against the arXiv paper's grounded spans,
       mathematical formulations, and defense countermeasures.
     - Scores Paper Fidelity (0-100), Algorithmic Correctness (0-100), and Security & Anti-Mock
       Compliance (0-100) with structured JSON findings and actionable recommendations.
     - Never allows an optimistic LLM verdict to override a failed deterministic security or test gate.
     - Falls back gracefully with explicit `ollama pull <model>` guidance if Ollama is offline.
"""

from __future__ import annotations

import ast
import concurrent.futures
import importlib.util
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dynarot import (
    DynaRotModelRouter,
    build_no_proxy_opener,
    detect_live_hardware,
    find_best_installed_chat_model,
    probe_ollama_models,
    sanitize_model_tag,
    size_glm_by_hardware,
)
from .evolve_engine import EVOLVE_END_MARKER, EVOLVE_START_MARKER, evaluate_python_candidate
from .langextract_engine import normalize_arxiv_id
from .models import DynaRotRoutingDecision

FORBIDDEN_SINGLE_TOKEN_MOCKS = frozenset(
    {
        "powershell",
        "powershell.exe",
        "cmd",
        "cmd.exe",
        "python",
        "python.exe",
        "bash",
        "sh",
        "curl",
        "wget",
        "nc",
    }
)

DANGEROUS_CALL_NAMES = frozenset({"eval", "exec", "compile", "__import__"})
DANGEROUS_ATTR_CALLS = frozenset(
    {
        ("os", "system"),
        ("os", "popen"),
        ("pickle", "loads"),
        ("pickle", "load"),
        ("marshal", "loads"),
    }
)


@dataclass
class CodeVerificationReport:
    """Structured result of Deterministic + LLM-as-a-Judge generated code verification."""

    arxiv_id: str
    title: str
    code_stage: str  # "synthesized" | "evolved_champion" | "custom"
    verdict: str  # "PASS" | "WARN" | "FAIL"
    overall_score: float
    verification_mode: str  # "llm_and_deterministic_verified" | "deterministic_static_and_runtime_fallback"
    llm_verified: bool
    verifier_model: str
    verifier_provider: str  # "ollama" | "gemini" | "deterministic_fallback"
    deterministic_checks: Dict[str, Any]
    llm_audit: Dict[str, Any]
    findings: List[Dict[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        det = self.deterministic_checks
        llm = self.llm_audit
        lines = [
            f"## Code Verification Report (`arXiv:{self.arxiv_id}` — `{self.code_stage}`)",
            f"- **Final Verdict**: `{self.verdict}` (`{self.overall_score}/100`)",
            f"- **Verification Mode**: `{self.verification_mode}`",
            f"- **Verifier Model**: `{self.verifier_model}` (`{self.verifier_provider}` | LLM Executed: `{self.llm_verified}`)",
            f"- **Deterministic Checks**: Syntax=`{det.get('ast_syntax_valid')}`, "
            f"Lint=`{'PASS' if det.get('linter_passed') else 'FAIL'}` ({','.join(det.get('linter_engines', []))}), "
            f"Mypy=`{'PASS' if det.get('mypy_passed') else 'FAIL'}` ({det.get('mypy_issue_count', 0)} errs), "
            f"Fuzz=`{'PASS' if det.get('fuzz_passed') else 'FAIL'}` ({det.get('fuzz_properties_checked', 0)} props), "
            f"Pytest=`{'PASS' if det.get('pytest_passed') else 'FAIL'}`, "
            f"Tests=`{det.get('runtime_tests_passed')}/{det.get('runtime_tests_total')}`, "
            f"Constant-Time HMAC=`{det.get('constant_time_hmac_used')}`, "
            f"Zero Dangerous Calls=`{det.get('zero_dangerous_calls')}`, "
            f"Zero Mock Indicators=`{det.get('zero_mock_single_token_indicators')}`",
        ]
        if self.llm_verified:
            lines.extend(
                [
                    f"- **LLM Paper Fidelity Score**: `{llm.get('paper_fidelity_score', 0.0)}/100`",
                    f"- **LLM Correctness Score**: `{llm.get('correctness_score', 0.0)}/100`",
                    f"- **LLM Security Score**: `{llm.get('security_score', 0.0)}/100`",
                    f"- **LLM Summary**: {llm.get('summary', '')}",
                ]
            )
        else:
            guidance = llm.get("ollama_guidance") or {}
            cmd = guidance.get("pull_command", "")
            lines.append(
                f"- **LLM Judge Status**: `{llm.get('status', 'offline')}`"
                + (f" (enable live LLM verification via `{cmd}`)" if cmd else "")
            )
        if self.findings:
            lines.append("### Findings")
            for f in self.findings:
                lines.append(
                    f"- **[{f.get('severity', 'INFO')}] ({f.get('category', 'general')})**: "
                    f"{f.get('message', '')}"
                    + (f" — *Recommendation*: {f['recommendation']}" if f.get("recommendation") else "")
                )
        return "\n".join(lines) + "\n"


# =============================================================================
# Layer 1: Deterministic Static Linters (AST + Ruff), Pytest & Runtime Gate
# =============================================================================

def run_static_code_linters(source_code: str) -> Dict[str, Any]:
    """
    Runs multi-layer static linting over generated Python code:
      1. Built-in zero-dependency strict AST linter:
         - Unused imports (`AST-F401`)
         - Duplicate dictionary literal keys (`AST-F601`)
         - Bare `except:` clauses (`AST-E722`)
         - Mutable default arguments like `def f(x=[])` (`AST-B006`)
         - Cache-key length-collision anti-pattern (`AST-BUG001`: `(payload_corpus, len(indicators))`)
      2. External `ruff check --select E,F,W,B --ignore E501` subprocess linter when `ruff` is installed.
    """
    issues: List[Dict[str, Any]] = []
    engines_used: List[str] = ["builtin_ast_linter"]

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return {
            "linter_passed": False,
            "linter_engines": engines_used,
            "issue_count": 1,
            "issues": [
                {
                    "engine": "builtin_ast_linter",
                    "code": "E999",
                    "line": exc.lineno or 1,
                    "message": f"SyntaxError: {exc.msg}",
                }
            ],
        }

    # 1a. Collect imported names vs referenced names (skipping `from __future__ import ...`)
    imported: Dict[str, int] = {}
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                imported[bound] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                imported[bound] = node.lineno
        elif isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            referenced.add(node.value.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Keep forward-reference string annotations (e.g. "DelegationScope")
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value):
                referenced.add(tok)

        # 1b. Bare except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(
                {
                    "engine": "builtin_ast_linter",
                    "code": "AST-E722",
                    "line": node.lineno,
                    "message": "Bare `except:` clause detected; catch a specific exception type.",
                }
            )

        # 1c. Mutable default arguments
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            for dflt in all_defaults:
                if isinstance(dflt, (ast.List, ast.Dict, ast.Set)):
                    issues.append(
                        {
                            "engine": "builtin_ast_linter",
                            "code": "AST-B006",
                            "line": dflt.lineno,
                            "message": f"Mutable default argument in function `{node.name}`.",
                        }
                    )

        # 1d. Duplicate literal dictionary keys
        if isinstance(node, ast.Dict):
            seen_keys: set[Any] = set()
            for k in node.keys:
                if isinstance(k, ast.Constant):
                    key_repr = (type(k.value).__name__, k.value)
                    if key_repr in seen_keys:
                        issues.append(
                            {
                                "engine": "builtin_ast_linter",
                                "code": "AST-F601",
                                "line": k.lineno,
                                "message": f"Duplicate dictionary literal key `{k.value!r}`.",
                            }
                        )
                    seen_keys.add(key_repr)

    for imp_name, lineno in imported.items():
        if imp_name not in referenced:
            issues.append(
                {
                    "engine": "builtin_ast_linter",
                    "code": "AST-F401",
                    "line": lineno,
                    "message": f"Unused import `{imp_name}`.",
                }
            )

    # 1e. Detect cache-key length collision anti-pattern `(payload_corpus, len(indicators))`
    if re.search(r"cache_key\s*=\s*\(\s*payload_corpus\s*,\s*len\s*\(\s*indicators\s*\)\s*\)", source_code):
        issues.append(
            {
                "engine": "builtin_ast_linter",
                "code": "AST-BUG001",
                "line": 1,
                "message": "Cache key uses `len(indicators)` instead of `tuple(indicators)`, causing stale collisions.",
            }
        )

    # 2. Run external `ruff check` if installed
    if importlib.util.find_spec("ruff") is not None:
        engines_used.append("ruff")
        creationflags = 0x08000000 if platform.system() == "Windows" else 0
        with tempfile.TemporaryDirectory(prefix="arxiv_lint_") as tmpdir:
            tmp_py = Path(tmpdir) / "generated_module.py"
            tmp_py.write_text(source_code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ruff",
                        "check",
                        "--output-format",
                        "json",
                        "--select",
                        "E,F,W,B",
                        "--ignore",
                        "E501",
                        str(tmp_py),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    creationflags=creationflags,
                )
                raw_out = (proc.stdout or "").strip()
                if raw_out:
                    ruff_items = json.loads(raw_out)
                    if isinstance(ruff_items, list):
                        for item in ruff_items[:20]:
                            loc = item.get("location") or {}
                            issues.append(
                                {
                                    "engine": "ruff",
                                    "code": str(item.get("code", "RUFF")),
                                    "line": int(loc.get("row", 1)),
                                    "message": str(item.get("message", "")),
                                }
                            )
            except Exception as exc:
                issues.append(
                    {
                        "engine": "ruff",
                        "code": "RUFF-EXEC",
                        "line": 1,
                        "message": f"Ruff execution warning: {type(exc).__name__}: {str(exc)[:140]}",
                    }
                )

    return {
        "linter_passed": len(issues) == 0,
        "linter_engines": engines_used,
        "issue_count": len(issues),
        "issues": issues,
    }


def run_generated_code_pytest(
    source_code: str,
    arxiv_id: str = "custom",
    timeout_sec: int = 20,
) -> Dict[str, Any]:
    """
    Executes real `python -m pytest` in an isolated temporary directory directly against
    the generated Python module (with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and isolated `--basetemp`).
    """
    has_discoverable_tests = bool(
        re.search(r"^\s*def\s+test_", source_code, flags=re.MULTILINE)
        or re.search(r"^\s*class\s+Test[A-Za-z0-9_]*", source_code, flags=re.MULTILINE)
    )
    if not has_discoverable_tests:
        return {
            "pytest_executed": False,
            "pytest_passed": True,
            "exit_code": 5,
            "summary": "No pytest test_* functions or Test* classes declared in custom snippet.",
        }

    if importlib.util.find_spec("pytest") is None:
        return {
            "pytest_executed": False,
            "pytest_passed": True,
            "exit_code": -1,
            "summary": "pytest package not installed; relied on embedded run_deterministic_benchmark().",
        }

    slug = re.sub(r"[^A-Za-z0-9_]+", "_", arxiv_id or "custom")
    creationflags = 0x08000000 if platform.system() == "Windows" else 0
    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    with tempfile.TemporaryDirectory(prefix="arxiv_pytest_") as tmpdir:
        tmp_root = Path(tmpdir)
        test_file = tmp_root / f"test_generated_{slug}.py"
        basetemp = tmp_root / "basetemp"
        test_file.write_text(source_code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--no-header",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    str(basetemp),
                    str(test_file),
                ],
                cwd=str(tmp_root),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                creationflags=creationflags,
            )
            combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            last_line = [ln.strip() for ln in combined.splitlines() if ln.strip()]
            summary_line = last_line[-1] if last_line else f"exit_code={proc.returncode}"
            return {
                "pytest_executed": True,
                "pytest_passed": proc.returncode == 0,
                "exit_code": proc.returncode,
                "summary": summary_line[:240],
            }
        except Exception as exc:
            return {
                "pytest_executed": True,
                "pytest_passed": False,
                "exit_code": -2,
                "summary": f"pytest execution error: {type(exc).__name__}: {str(exc)[:180]}",
            }


def run_mypy_type_checker(
    source_code: str,
    timeout_sec: int = 25,
) -> Dict[str, Any]:
    """
    Executes strict static type checking (`python -m mypy --strict --ignore-missing-imports`)
    against generated Python code in an isolated temporary directory.
    """
    if importlib.util.find_spec("mypy") is None:
        return {
            "mypy_executed": False,
            "mypy_passed": True,
            "issue_count": 0,
            "issues": [],
            "summary": "mypy not installed; skipped strict static type check.",
        }

    creationflags = 0x08000000 if platform.system() == "Windows" else 0
    with tempfile.TemporaryDirectory(prefix="arxiv_mypy_") as tmpdir:
        tmp_root = Path(tmpdir)
        src_file = tmp_root / "generated_module.py"
        cache_dir = tmp_root / "mypy_cache"
        src_file.write_text(source_code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "--strict",
                    "--ignore-missing-imports",
                    "--no-error-summary",
                    "--no-color-output",
                    "--show-error-codes",
                    "--cache-dir",
                    str(cache_dir),
                    str(src_file),
                ],
                cwd=str(tmp_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                creationflags=creationflags,
            )
            issues: List[Dict[str, Any]] = []
            for raw_line in (proc.stdout or "").splitlines():
                line = raw_line.strip()
                if not line or ": note:" in line:
                    continue
                m = re.match(r"^.+?:(\d+)(?::\d+)?:\s*(error|warning):\s*(.+?)(?:\s+\[([^\]]+)\])?$", line)
                if m:
                    issues.append(
                        {
                            "engine": "mypy",
                            "line": int(m.group(1)),
                            "severity": m.group(2),
                            "code": m.group(4) or "mypy-error",
                            "message": m.group(3).strip(),
                        }
                    )
                elif proc.returncode != 0:
                    issues.append(
                        {
                            "engine": "mypy",
                            "line": 1,
                            "severity": "error",
                            "code": "mypy-error",
                            "message": line[:200],
                        }
                    )
            passed = proc.returncode == 0 and len(issues) == 0
            return {
                "mypy_executed": True,
                "mypy_passed": passed,
                "issue_count": len(issues),
                "issues": issues[:20],
                "summary": "mypy --strict passed (0 errors)" if passed else f"mypy --strict found {len(issues)} error(s)",
            }
        except Exception as exc:
            return {
                "mypy_executed": True,
                "mypy_passed": False,
                "issue_count": 1,
                "issues": [
                    {
                        "engine": "mypy",
                        "line": 1,
                        "severity": "error",
                        "code": "mypy-exec",
                        "message": f"mypy execution failed: {type(exc).__name__}: {str(exc)[:180]}",
                    }
                ],
                "summary": f"mypy execution error: {type(exc).__name__}",
            }


def run_property_based_fuzzing(
    source_code: str,
    seed: int = 2026,
    num_trials: int = 35,
) -> Dict[str, Any]:
    """
    Randomized property-based fuzzer executing directly against the generated Python module:
      1. Algebraic Meet-Semilattice Laws on `DelegationScope.meet()`:
         - Idempotence:      S meet S == S
         - Commutativity:    S1 meet S2 == S2 meet S1
         - Associativity:    (S1 meet S2) meet S3 == S1 meet (S2 meet S3)
         - Top Identity:     Top meet S == S == S meet Top
         - Monotonicity:     allows(S1 meet S2) => allows(S1) AND allows(S2), and restrictions only grow
      2. Adversarial Input Fuzzing on `scan_and_score_payload()`:
         - Empty, null-byte, Unicode/emoji, huge separator strings, alpha <= 0, NaN/Inf prior_wealth
         - Soundness: injected indicator substrings ALWAYS trigger
         - Cache Isolation: distinct same-length random indicator tuples NEVER cross-contaminate
      3. Cryptographic & Scope Fuzzing on `ArxivAlgorithmEngine_*`:
         - Bit-flipped HMAC seals ALWAYS denied (`session_lineage_verified == False`)
         - Replayed `approval_nonce` ALWAYS denied (`approval_token_replay`)
         - Out-of-scope concrete resources ALWAYS denied
    """
    failures: List[Dict[str, str]] = []
    properties_checked = 0

    try:
        compiled = compile(source_code, "<fuzz_target>", "exec")
    except Exception as exc:
        return {
            "fuzz_executed": False,
            "fuzz_passed": False,
            "trials_run": 0,
            "properties_checked": 0,
            "failures": [{"property": "compile", "detail": str(exc)[:200]}],
        }

    mod_name = f"_arxiv_fuzz_mod_{abs(hash(source_code)) % 1000000}"
    mod = types.ModuleType(mod_name)
    ns: Dict[str, Any] = mod.__dict__
    ns["__name__"] = mod_name
    sys.modules[mod_name] = mod
    try:
        exec(compiled, ns)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        return {
            "fuzz_executed": False,
            "fuzz_passed": False,
            "trials_run": 0,
            "properties_checked": 0,
            "failures": [{"property": "module_load", "detail": str(exc)[:200]}],
        }

    rng = random.Random(seed)
    try:
        scope_cls = ns.get("DelegationScope")
        scan_fn = ns.get("scan_and_score_payload")
        engine_cls = next(
            (v for k, v in ns.items() if k.startswith("ArxivAlgorithmEngine_") and isinstance(v, type)),
            None,
        )

        # 1. Fuzz Algebraic Meet-Semilattice Laws on DelegationScope
        if scope_cls is not None and hasattr(scope_cls, "meet"):
            res_universe = ["repo_a", "repo_b", "repo_c", "db_prod"]
            act_universe = ["read_public", "compile_workspace", "read_sensitive", "external_transmit"]
            dom_universe = ["engineering", "finance", "default"]
            comp_universe = [("read_sensitive", "external_transmit"), ("read_public", "external_transmit")]
            top_scope = scope_cls(
                resources=frozenset(["*"]),
                actions=frozenset(["*"]),
                data_domains=frozenset(["*"]),
            )

            def _rand_scope() -> Any:
                r_set = frozenset(["*"]) if rng.random() < 0.25 else frozenset(rng.sample(res_universe, rng.randint(1, 3)))
                a_set = frozenset(["*"]) if rng.random() < 0.25 else frozenset(rng.sample(act_universe, rng.randint(1, 3)))
                d_set = frozenset(["*"]) if rng.random() < 0.25 else frozenset(rng.sample(dom_universe, rng.randint(1, 2)))
                c_set = frozenset(rng.sample(comp_universe, rng.randint(0, len(comp_universe))))
                return scope_cls(resources=r_set, actions=a_set, data_domains=d_set, prohibited_compositions=c_set)

            for trial in range(num_trials):
                s1, s2, s3 = _rand_scope(), _rand_scope(), _rand_scope()
                properties_checked += 5
                # Idempotence
                if s1.meet(s1) != s1:
                    failures.append({"property": "semilattice_idempotence", "detail": f"Trial {trial}: S meet S != S"})
                    break
                # Commutativity
                m12, m21 = s1.meet(s2), s2.meet(s1)
                if m12 != m21:
                    failures.append({"property": "semilattice_commutativity", "detail": f"Trial {trial}: S1 meet S2 != S2 meet S1"})
                    break
                # Associativity
                if m12.meet(s3) != s1.meet(s2.meet(s3)):
                    failures.append({"property": "semilattice_associativity", "detail": f"Trial {trial}: (S1 meet S2) meet S3 != S1 meet (S2 meet S3)"})
                    break
                # Wildcard Top-Element Identity
                if top_scope.meet(s1) != s1 or s1.meet(top_scope) != s1:
                    failures.append({"property": "semilattice_top_identity", "detail": f"Trial {trial}: Top(*) meet S != S"})
                    break
                # Monotonic attenuation & restriction accumulation
                probe_r = rng.choice(res_universe)
                probe_a = rng.choice(act_universe)
                probe_d = rng.choice(dom_universe)
                if m12.allows(probe_r, probe_a, probe_d) and not (
                    s1.allows(probe_r, probe_a, probe_d) and s2.allows(probe_r, probe_a, probe_d)
                ):
                    failures.append({"property": "semilattice_monotonic_attenuation", "detail": f"Trial {trial}: meet escalated privilege for {(probe_r, probe_a, probe_d)}"})
                    break
                if not (s1.prohibited_compositions <= m12.prohibited_compositions and s2.prohibited_compositions <= m12.prohibited_compositions):
                    failures.append({"property": "semilattice_restriction_union", "detail": f"Trial {trial}: prohibited_compositions lost during meet"})
                    break

        # 2. Fuzz scan_and_score_payload() on adversarial edge cases + cache isolation
        if callable(scan_fn):
            adversarial_corpora = [
                "",
                "\x00\x01\x02",
                "/" * 64 + "|" * 64 + ";" * 32,
                "Unicode \u202e\u0301 \U0001f680 safe payload",
                "UPPERCASE_ONLY_TEXT",
            ]
            adversarial_alphas = [0.0, -0.05, 1e-9, 0.01, 0.5, 2.0]
            adversarial_wealths = [0.0, -5.0, 1.0, 1e6, float("nan"), float("inf")]

            for trial in range(num_trials):
                corpus = (
                    adversarial_corpora[trial]
                    if trial < len(adversarial_corpora)
                    else "".join(rng.choices("abcxyz/|;0123 \t\n", k=rng.randint(0, 120)))
                )
                alpha_val = adversarial_alphas[trial % len(adversarial_alphas)]
                wealth_val = adversarial_wealths[trial % len(adversarial_wealths)]
                tok_a = f"fuzz_ind_a_{trial}_{rng.randint(10, 99)}"
                tok_b = f"fuzz_ind_b_{trial}_{rng.randint(10, 99)}"
                ind_a = (tok_a, f"other_a_{trial}")
                ind_b = (tok_b, f"other_b_{trial}")
                properties_checked += 3
                try:
                    m1, nc1, w1, t1 = scan_fn(corpus, ind_a, wealth_val, alpha_val)
                    if not isinstance(m1, tuple) or not isinstance(t1, bool):
                        failures.append({"property": "scan_return_types", "detail": f"Trial {trial}: invalid return types"})
                        break
                    if not (math.isfinite(nc1) and 0.0 <= nc1 <= 1.0 and math.isfinite(w1)):
                        failures.append({"property": "scan_finite_bounds", "detail": f"Trial {trial}: non-finite/out-of-range nc={nc1}, w={w1}"})
                        break

                    # Soundness + same-length cache isolation fuzz
                    injected_corpus = f"{corpus} {tok_b.upper()} tail"
                    ma, _, _, ta = scan_fn(injected_corpus, ind_a, 1.0, 0.01)
                    mb, _, _, tb = scan_fn(injected_corpus, ind_b, 1.0, 0.01)
                    if ta or len(ma) != 0 or (not tb) or tok_b not in mb:
                        failures.append({"property": "scan_soundness_and_cache_isolation", "detail": f"Trial {trial}: cache collision or missed indicator (ta={ta}, tb={tb}, mb={mb})"})
                        break
                except Exception as exc:
                    failures.append({"property": "scan_no_crash", "detail": f"Trial {trial} (alpha={alpha_val}, wealth={wealth_val}): {type(exc).__name__}: {exc}"})
                    break

        # 3. Fuzz ArxivAlgorithmEngine_* security invariants (HMAC bit-flip & nonce replay)
        if engine_cls is not None and scope_cls is not None:
            eng = engine_cls(
                signing_key=b"fuzz-hmac-key-32-bytes-strict!!",
                allowed_actions=["read_public", "compile_workspace", "read_sensitive", "external_transmit"],
            )
            for trial in range(min(15, num_trials)):
                actor = f"actor-{trial}"
                sess = f"sess-fuzz-{trial}"
                act = "read_public"
                seal = eng.issue_envelope_seal(actor, sess, act)
                # Flip first hex char of HMAC seal
                bad_char = "0" if seal[0] != "0" else "f"
                forged = bad_char + seal[1:]
                properties_checked += 2
                v_forged = eng.evaluate_action(actor, sess, act, "benign test text", forged)
                if v_forged.allowed or v_forged.session_lineage_verified:
                    failures.append({"property": "hmac_bitflip_reject", "detail": f"Trial {trial}: forged HMAC seal was allowed!"})
                    break

                nonce = f"fuzz-nonce-{trial}"
                v1 = eng.evaluate_action(actor, sess, act, "benign test text", seal, approval_nonce=nonce)
                v2 = eng.evaluate_action(actor, sess, act, "benign test text", seal, approval_nonce=nonce)
                if not v1.allowed or v2.allowed:
                    failures.append({"property": "approval_nonce_replay_reject", "detail": f"Trial {trial}: replay nonce guard failed (v1={v1.allowed}, v2={v2.allowed})"})
                    break
    finally:
        sys.modules.pop(mod_name, None)

    return {
        "fuzz_executed": True,
        "fuzz_passed": len(failures) == 0,
        "trials_run": num_trials,
        "properties_checked": properties_checked,
        "failures": failures,
    }


def run_deterministic_code_verification(
    source_code: str,
    arxiv_id: str = "custom",
    benchmark_iterations: int = 40,
    run_pytest_subprocess: bool = True,
    run_mypy_subprocess: bool = True,
    run_fuzzing: bool = True,
) -> Dict[str, Any]:
    """
    Performs fail-closed static AST inspection, security invariant checks, anti-mock
    verification, and live deterministic execution via `evaluate_python_candidate()`.
    """
    findings: List[Dict[str, str]] = []
    has_evolve_block = (EVOLVE_START_MARKER in source_code) and (EVOLVE_END_MARKER in source_code)
    if not has_evolve_block:
        findings.append(
            {
                "severity": "MEDIUM",
                "category": "structure",
                "message": "Missing # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END markers.",
                "recommendation": "Wrap core algorithmic kernel inside EVOLVE-BLOCK markers for ShinkaEvolve/OpenEvolve.",
            }
        )

    ast_syntax_valid = False
    constant_time_hmac_used = False
    meet_semilattice_present = False
    dangerous_calls_found: List[str] = []
    mock_indicators_found: List[str] = []
    tree: Optional[ast.AST] = None

    try:
        tree = ast.parse(source_code)
        ast_syntax_valid = True
    except SyntaxError as exc:
        findings.append(
            {
                "severity": "CRITICAL",
                "category": "correctness",
                "message": f"Python AST syntax error at line {exc.lineno}: {exc.msg}",
                "recommendation": "Fix Python syntax before executing or deploying generated module.",
            }
        )

    if tree is not None:
        for node in ast.walk(tree):
            # Check constant-time HMAC verification
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "compare_digest":
                        constant_time_hmac_used = True
                    if isinstance(node.func.value, ast.Name):
                        pair = (node.func.value.id, node.func.attr)
                        if pair in DANGEROUS_ATTR_CALLS:
                            dangerous_calls_found.append(f"{pair[0]}.{pair[1]}")
                        if pair[0] == "subprocess":
                            for kw in node.keywords:
                                if (
                                    kw.arg == "shell"
                                    and isinstance(kw.value, ast.Constant)
                                    and kw.value.value is True
                                ):
                                    dangerous_calls_found.append(f"subprocess.{pair[1]}(shell=True)")
                elif isinstance(node.func, ast.Name):
                    if node.func.id in DANGEROUS_CALL_NAMES:
                        dangerous_calls_found.append(node.func.id)

            # Check DelegationScope.meet()
            if isinstance(node, ast.ClassDef) and node.name == "DelegationScope":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "meet":
                        meet_semilattice_present = True

            # Check MULTI_TOKEN_TECHNICAL_INDICATORS for single-word mock rules
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_name = ""
                val_node = None
                if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                    val_node = node.value
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    target_name = node.target.id
                    val_node = node.value

                if target_name == "MULTI_TOKEN_TECHNICAL_INDICATORS" and isinstance(
                    val_node, (ast.Tuple, ast.List, ast.Set)
                ):
                    for elt in val_node.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            tok = elt.value.strip().lower()
                            if tok in FORBIDDEN_SINGLE_TOKEN_MOCKS:
                                mock_indicators_found.append(tok)

    zero_dangerous_calls = len(dangerous_calls_found) == 0
    zero_mock_indicators = len(mock_indicators_found) == 0

    if not zero_dangerous_calls:
        findings.append(
            {
                "severity": "CRITICAL",
                "category": "security",
                "message": f"Dangerous dynamic/shell calls detected in generated code: {sorted(set(dangerous_calls_found))}",
                "recommendation": "Remove eval/exec/os.system/shell=True calls; use safe structured stdlib primitives.",
            }
        )
    if not zero_mock_indicators:
        findings.append(
            {
                "severity": "HIGH",
                "category": "security",
                "message": f"Naive single-token mock indicators detected: {mock_indicators_found}",
                "recommendation": "Replace single process names with multi-token technical indicators.",
            }
        )
    if ast_syntax_valid and not constant_time_hmac_used:
        findings.append(
            {
                "severity": "MEDIUM",
                "category": "security",
                "message": "No constant-time hmac.compare_digest() verification detected in module.",
                "recommendation": "Verify cryptographic envelope seals using hmac.compare_digest() to prevent timing side-channels.",
            }
        )

    # Run static linters (AST + Ruff), mypy --strict, property-based fuzzing & subprocess pytest concurrently
    clean_id = normalize_arxiv_id(arxiv_id) or "custom"
    linter_report: Dict[str, Any]
    mypy_report: Dict[str, Any]
    pytest_report: Dict[str, Any]
    fuzz_report: Dict[str, Any]
    if ast_syntax_valid and zero_dangerous_calls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            f_lint = pool.submit(run_static_code_linters, source_code)
            f_mypy = (
                pool.submit(run_mypy_type_checker, source_code)
                if run_mypy_subprocess
                else None
            )
            f_pytest = (
                pool.submit(run_generated_code_pytest, source_code, clean_id)
                if run_pytest_subprocess
                else None
            )
            f_fuzz = (
                pool.submit(run_property_based_fuzzing, source_code)
                if run_fuzzing
                else None
            )
            metrics = evaluate_python_candidate(
                source_code=source_code,
                candidate_id=f"verify-{clean_id}",
                benchmark_iterations=benchmark_iterations,
            )
            linter_report = f_lint.result()
            mypy_report = (
                f_mypy.result()
                if f_mypy is not None
                else {"mypy_executed": False, "mypy_passed": True, "issue_count": 0, "issues": [], "summary": "Skipped mypy by caller."}
            )
            pytest_report = (
                f_pytest.result()
                if f_pytest is not None
                else {"pytest_executed": False, "pytest_passed": True, "exit_code": 0, "summary": "Skipped subprocess pytest by caller."}
            )
            fuzz_report = (
                f_fuzz.result()
                if f_fuzz is not None
                else {"fuzz_executed": False, "fuzz_passed": True, "trials_run": 0, "properties_checked": 0, "failures": []}
            )

        runtime_error = metrics.error
        tests_passed = metrics.tests_passed
        tests_total = metrics.tests_total
        runtime_fitness = metrics.total_fitness
        execution_time_us = metrics.execution_time_us
        peak_memory_kib = metrics.peak_memory_kib
    else:
        linter_report = run_static_code_linters(source_code)
        mypy_report = {
            "mypy_executed": False,
            "mypy_passed": False,
            "issue_count": 1,
            "issues": [{"engine": "mypy", "line": 1, "severity": "error", "code": "skip", "message": "Skipped mypy due to syntax/dangerous call failure."}],
            "summary": "Skipped mypy due to syntax or dangerous call failure.",
        }
        fuzz_report = {
            "fuzz_executed": False,
            "fuzz_passed": False,
            "trials_run": 0,
            "properties_checked": 0,
            "failures": [{"property": "precheck", "detail": "Skipped fuzzing due to syntax or dangerous call failure."}],
        }
        runtime_error = "Skipped runtime execution due to syntax error or dangerous AST call."
        tests_passed = 0
        tests_total = 7
        runtime_fitness = 0.0
        execution_time_us = 0.0
        peak_memory_kib = 0.0
        pytest_report = {
            "pytest_executed": False,
            "pytest_passed": False,
            "exit_code": 2,
            "summary": "Skipped pytest due to syntax error or dangerous AST call.",
        }

    linter_passed = bool(linter_report["linter_passed"])
    if not linter_passed:
        for li in linter_report["issues"][:8]:
            findings.append(
                {
                    "severity": "HIGH" if str(li.get("code", "")).startswith(("E9", "F8", "AST-BUG")) else "MEDIUM",
                    "category": "lint",
                    "message": f"[{li['engine']}:{li['code']} L{li['line']}] {li['message']}",
                    "recommendation": "Resolve static linter violation before shipping generated code.",
                }
            )

    mypy_passed = bool(mypy_report["mypy_passed"])
    if not mypy_passed:
        for mi in mypy_report["issues"][:8]:
            findings.append(
                {
                    "severity": "HIGH",
                    "category": "typing",
                    "message": f"[mypy:{mi['code']} L{mi['line']}] {mi['message']}",
                    "recommendation": "Fix strict static type annotation error reported by mypy --strict.",
                }
            )

    fuzz_passed = bool(fuzz_report["fuzz_passed"])
    if not fuzz_passed:
        for fi in fuzz_report["failures"][:6]:
            findings.append(
                {
                    "severity": "CRITICAL",
                    "category": "fuzzing",
                    "message": f"[fuzz:{fi['property']}] {fi['detail']}",
                    "recommendation": "Fix algebraic semilattice, bounds, or security invariant counterexample found by property-based fuzzing.",
                }
            )

    pytest_passed = bool(pytest_report["pytest_passed"])
    if not pytest_passed:
        findings.append(
            {
                "severity": "CRITICAL",
                "category": "correctness",
                "message": f"Subprocess pytest failed on generated module: {pytest_report['summary']}",
                "recommendation": "Fix failing pytest test cases in generated module.",
            }
        )

    if runtime_error:
        findings.append(
            {
                "severity": "CRITICAL",
                "category": "correctness",
                "message": f"Runtime benchmark/test failure: {runtime_error}",
                "recommendation": "Ensure module compiles cleanly and implements passing run_deterministic_benchmark().",
            }
        )
    elif tests_passed < tests_total:
        findings.append(
            {
                "severity": "CRITICAL",
                "category": "correctness",
                "message": f"Only {tests_passed}/{tests_total} deterministic unit tests passed.",
                "recommendation": "Fix failing invariants in scan_and_score_payload() or ArxivAlgorithmEngine.",
            }
        )

    # Deterministic 0-100 composite score (AST + Security + Lint + Mypy + Fuzzing + Benchmark + Pytest)
    score = 0.0
    if ast_syntax_valid:
        score += 10.0
    if zero_dangerous_calls:
        score += 15.0
    if zero_mock_indicators:
        score += 10.0
    if linter_passed:
        score += 10.0
    if mypy_passed:
        score += 10.0
    if fuzz_passed:
        score += 10.0
    if has_evolve_block:
        score += 5.0
    if constant_time_hmac_used:
        score += 5.0
    if meet_semilattice_present:
        score += 5.0
    if tests_total > 0 and runtime_error is None:
        score += 10.0 * (tests_passed / float(tests_total))
    if pytest_passed and runtime_error is None:
        score += 10.0

    return {
        "ast_syntax_valid": ast_syntax_valid,
        "evolve_block_present": has_evolve_block,
        "constant_time_hmac_used": constant_time_hmac_used,
        "meet_semilattice_present": meet_semilattice_present,
        "zero_dangerous_calls": zero_dangerous_calls,
        "dangerous_calls_found": sorted(set(dangerous_calls_found)),
        "zero_mock_single_token_indicators": zero_mock_indicators,
        "mock_indicators_found": mock_indicators_found,
        "linter_passed": linter_passed,
        "linter_engines": linter_report["linter_engines"],
        "linter_issue_count": linter_report["issue_count"],
        "linter_issues": linter_report["issues"],
        "mypy_executed": bool(mypy_report["mypy_executed"]),
        "mypy_passed": mypy_passed,
        "mypy_issue_count": int(mypy_report["issue_count"]),
        "mypy_issues": mypy_report["issues"],
        "mypy_summary": str(mypy_report["summary"]),
        "fuzz_executed": bool(fuzz_report["fuzz_executed"]),
        "fuzz_passed": fuzz_passed,
        "fuzz_trials_run": int(fuzz_report["trials_run"]),
        "fuzz_properties_checked": int(fuzz_report["properties_checked"]),
        "fuzz_failures": fuzz_report["failures"],
        "pytest_executed": bool(pytest_report["pytest_executed"]),
        "pytest_passed": pytest_passed,
        "pytest_exit_code": int(pytest_report["exit_code"]),
        "pytest_summary": str(pytest_report["summary"]),
        "runtime_tests_passed": tests_passed,
        "runtime_tests_total": tests_total,
        "runtime_fitness_score": runtime_fitness,
        "execution_time_us": execution_time_us,
        "peak_memory_kib": peak_memory_kib,
        "runtime_error": runtime_error,
        "deterministic_score": round(min(100.0, score), 2),
        "static_findings": findings,
    }


# =============================================================================
# Layer 2: Open-Weight LLM-as-a-Judge Verification (Ollama / Gemini)
# =============================================================================

def _build_llm_verification_prompt(
    paper: Dict[str, Any],
    extractions: List[Dict[str, Any]],
    source_code: str,
    det_checks: Dict[str, Any],
) -> str:
    """Constructs a structured verification audit prompt for the LLM judge."""
    spans_lines = []
    for idx, ext in enumerate(extractions[:12], start=1):
        spans_lines.append(
            f"  {idx}. [{ext.get('extraction_class', 'unknown')}] "
            f"quote='{ext.get('source_quote', '')}' | "
            f"vector='{ext.get('attack_vector', '')}' | "
            f"math='{ext.get('mathematical_formulation', '')}' | "
            f"mitigation='{ext.get('defense_countermeasure') or ext.get('mitigation', '')}'"
        )
    spans_block = "\n".join(spans_lines) if spans_lines else "  (No extracted spans provided)"

    # Keep code concise for fast VRAM-resident local open-weight model inference while preserving core logic
    code_excerpt = source_code if len(source_code) <= 14000 else (
        source_code[:10500] + "\n# ... [middle truncated for verifier context window] ...\n" + source_code[-3000:]
    )

    return (
        "You are a strict formal-methods, Python security, and ML systems code reviewer.\n"
        "Verify whether the generated Python module faithfully, safely, and correctly implements "
        "the research paper's grounded concepts, mathematical formulations, and security countermeasures.\n"
        "Base your findings strictly on the actual Python source shown below; do not fabricate defects "
        "that contradict passing executable Layer-1 tests/fuzzing unless you cite a concrete failing counterexample.\n\n"
        f"=== PAPER METADATA ===\n"
        f"arXiv ID: {paper.get('arxiv_id', 'custom')}\n"
        f"Title: {paper.get('title', 'Unknown')}\n"
        f"Abstract: {(paper.get('summary', '') or '')[:1200]}\n\n"
        f"=== GOOGLE LANGEXTRACT GROUNDED SPANS ===\n"
        f"{spans_block}\n\n"
        f"=== LAYER-1 DETERMINISTIC CHECK SUMMARY ===\n"
        f"AST Syntax Valid: {det_checks['ast_syntax_valid']} | "
        f"Static Lint (Ruff+AST): {'PASS' if det_checks.get('linter_passed') else 'FAIL'} ({det_checks.get('linter_issue_count', 0)} issues) | "
        f"Mypy --strict: {'PASS' if det_checks.get('mypy_passed') else 'FAIL'} ({det_checks.get('mypy_issue_count', 0)} errors) | "
        f"Property-Based Fuzzing: {'PASS' if det_checks.get('fuzz_passed') else 'FAIL'} "
        f"({det_checks.get('fuzz_properties_checked', 0)} algebraic/adversarial/crypto properties checked) | "
        f"Subprocess Pytest: {'PASS' if det_checks.get('pytest_passed') else 'FAIL'} | "
        f"Unit Tests Passed: {det_checks['runtime_tests_passed']}/{det_checks['runtime_tests_total']} | "
        f"Runtime Fitness: {det_checks['runtime_fitness_score']}/100 | "
        f"Constant-Time HMAC: {det_checks['constant_time_hmac_used']} | "
        f"Zero Dangerous Calls: {det_checks['zero_dangerous_calls']}\n\n"
        f"=== GENERATED PYTHON CODE TO VERIFY ===\n"
        f"```python\n{code_excerpt}\n```\n\n"
        "Respond ONLY with a single valid JSON object matching this exact schema (no prose outside JSON):\n"
        "{\n"
        '  "verdict": "PASS" | "WARN" | "FAIL",\n'
        '  "paper_fidelity_score": <number 0-100>,\n'
        '  "correctness_score": <number 0-100>,\n'
        '  "security_score": <number 0-100>,\n'
        '  "summary": "<2-3 sentence technical audit verdict>",\n'
        '  "grounded_span_coverage": ["<span or concept verified in code>"],\n'
        '  "findings": [\n'
        '    {"severity": "INFO|LOW|MEDIUM|HIGH|CRITICAL", "category": "fidelity|correctness|security|performance", '
        '"message": "<specific observation>", "recommendation": "<concrete fix or enhancement>"}\n'
        "  ]\n"
        "}\n"
    )


def _clamp_score(val: Any, default: float = 0.0) -> float:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(100.0, num)), 2)


def parse_llm_verification_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parses a structured JSON verification verdict from raw LLM output.
    Strips `<think>...</think>` reasoning blocks (DeepSeek-R1 / GLM-4.7-Flash) and
    markdown code fences cleanly.
    """
    if not raw_text or not raw_text.strip():
        return None
    cleaned = raw_text.strip()
    end_think = cleaned.rfind("</think>")
    if end_think != -1:
        cleaned = cleaned[end_think + len("</think>") :].strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    parsed: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            parsed = obj
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(cleaned[start : end + 1])
                if isinstance(obj, dict):
                    parsed = obj
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        return None

    # Reject empty/sparse JSON objects (e.g., `{}`) lacking all required verification keys
    required_signal_keys = (
        "verdict",
        "paper_fidelity_score",
        "correctness_score",
        "security_score",
        "summary",
    )
    if not any(k in parsed and parsed.get(k) not in (None, "") for k in required_signal_keys):
        return None

    verdict_raw = str(parsed.get("verdict", "WARN")).strip().upper()
    if verdict_raw not in ("PASS", "WARN", "FAIL"):
        verdict_raw = "WARN"

    paper_fidelity = _clamp_score(parsed.get("paper_fidelity_score"), 70.0)
    correctness = _clamp_score(parsed.get("correctness_score"), 70.0)
    security = _clamp_score(parsed.get("security_score"), 70.0)
    summary = str(parsed.get("summary", "")).strip() or "LLM code verification completed."

    raw_cov = parsed.get("grounded_span_coverage")
    coverage_list = [str(x).strip() for x in raw_cov if str(x).strip()] if isinstance(raw_cov, list) else []

    raw_findings = parsed.get("findings")
    norm_findings: List[Dict[str, str]] = []
    if isinstance(raw_findings, list):
        for item in raw_findings[:12]:
            if isinstance(item, dict) and item.get("message"):
                sev = str(item.get("severity", "INFO")).strip().upper()
                if sev not in ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"):
                    sev = "INFO"
                cat = str(item.get("category", "general")).strip().lower()
                norm_findings.append(
                    {
                        "severity": sev,
                        "category": cat,
                        "message": str(item.get("message", "")).strip(),
                        "recommendation": str(item.get("recommendation", "")).strip(),
                    }
                )

    return {
        "verdict": verdict_raw,
        "paper_fidelity_score": paper_fidelity,
        "correctness_score": correctness,
        "security_score": security,
        "summary": summary,
        "grounded_span_coverage": coverage_list,
        "findings": norm_findings,
    }


def _invoke_ollama_llm_judge(
    endpoint: str,
    model_tag: str,
    prompt: str,
    timeout_sec: float = 120.0,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    Calls local Ollama `/api/generate` (with automatic `/api/chat` fallback if the model prefers
    chat completions) in strict JSON output mode (`format="json"`) using a no-proxy opener and
    WITHOUT `Origin: null` (which Ollama rejects with HTTP 403).
    Returns `(parsed_dict_or_None, status_code, raw_excerpt)`.
    """
    clean_model = sanitize_model_tag(model_tag)
    is_reasoning_model = any(k in clean_model.lower() for k in ("deepseek-r1", "qwq", "glm-4.7"))
    opener = build_no_proxy_opener()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ArxivToCode-LLMVerifier/1.0",
    }

    gen_url = f"{endpoint.rstrip('/')}/api/generate"
    gen_body_obj: Dict[str, Any] = {
        "model": clean_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 1000 if is_reasoning_model else 700,
            "num_ctx": 8192,
        },
    }
    if is_reasoning_model:
        # Disable multi-thousand-token hidden reasoning when strict JSON output is requested (Ollama >= 0.9)
        gen_body_obj["think"] = False
    gen_payload = json.dumps(gen_body_obj).encode("utf-8")

    try:
        req = urllib.request.Request(gen_url, data=gen_payload, method="POST", headers=headers)
        with opener.open(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
                raw_text = str(body.get("response", "")).strip()
                parsed = parse_llm_verification_json(raw_text)
                if parsed is None:
                    return None, "llm_parse_fallback", raw_text[:400]
                return parsed, "llm_verified", raw_text[:400]
            return None, f"ollama_http_{resp.status}_fallback", ""
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            # Try /api/chat fallback for chat-only Ollama templates (and without `think` for older Ollama)
            chat_url = f"{endpoint.rstrip('/')}/api/chat"
            chat_payload = json.dumps(
                {
                    "model": clean_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1, "num_predict": 900, "num_ctx": 8192},
                }
            ).encode("utf-8")
            try:
                chat_req = urllib.request.Request(chat_url, data=chat_payload, method="POST", headers=headers)
                with opener.open(chat_req, timeout=timeout_sec) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode("utf-8", errors="replace"))
                        msg_obj = body.get("message") or {}
                        raw_text = str(msg_obj.get("content", "")).strip()
                        parsed = parse_llm_verification_json(raw_text)
                        if parsed is None:
                            return None, "llm_parse_fallback", raw_text[:400]
                        return parsed, "llm_verified", raw_text[:400]
            except Exception:
                pass
        return None, f"ollama_http_{exc.code}_fallback", ""
    except TimeoutError:
        return None, "llm_timeout_fallback", ""
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return None, "llm_timeout_fallback", ""
        return None, f"ollama_request_error_fallback:{type(exc).__name__}", ""
    except Exception as exc:
        return None, f"ollama_error_fallback:{type(exc).__name__}", ""


def _invoke_gemini_llm_judge(
    model_id: str,
    api_key: str,
    prompt: str,
    timeout_sec: float = 25.0,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    Optional cloud fallback using Google Gemini Generative Language REST API when
    the caller explicitly passes `--verify-model gemini-...` and has an API key set.
    """
    clean_model = model_id.strip()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
        f"?key={api_key}"
    )
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            candidates = data.get("candidates") or []
            parts = (candidates[0].get("content") or {}).get("parts") if candidates else []
            raw_text = str(parts[0].get("text", "")).strip() if parts else ""
            parsed = parse_llm_verification_json(raw_text)
            if parsed is None:
                return None, "llm_parse_fallback", raw_text[:400]
            return parsed, "llm_verified", raw_text[:400]
    except Exception as exc:
        return None, f"gemini_error_fallback:{type(exc).__name__}", ""


def verify_generated_python_code(
    source_code: str,
    paper: Dict[str, Any],
    extractions: List[Dict[str, Any]],
    code_stage: str = "synthesized",
    model_id: Optional[str] = None,
    ollama_endpoint: Optional[str] = None,
    dynarot_decision: Optional[DynaRotRoutingDecision] = None,
    timeout_sec: float = 120.0,
    enable_llm: bool = True,
) -> CodeVerificationReport:
    """
    Verifies generated or evolved Python code using:
      1. Deterministic AST, security invariant, anti-mock, and live runtime test verification.
      2. Open-weight LLM-as-a-Judge auditing via local Ollama (selected by ModelFusion DynaRot)
         or Gemini REST API when explicitly requested.

    Fail-closed guarantee: An LLM verdict can NEVER override a failing deterministic AST,
    security, or unit-test gate.
    """
    t0 = time.perf_counter()
    clean_id = normalize_arxiv_id(str(paper.get("arxiv_id", "custom"))) or "custom"
    title = str(paper.get("title", f"arXiv:{clean_id}"))
    endpoint = ollama_endpoint or os.environ.get("LOCAL_OLLAMA_ENDPOINT", "http://127.0.0.1:11434")

    # Layer 1: Deterministic Static + Runtime Verification
    det_checks = run_deterministic_code_verification(
        source_code=source_code,
        arxiv_id=clean_id,
    )

    if dynarot_decision is None:
        dynarot_decision = DynaRotModelRouter(endpoint=endpoint).route(
            prompt=f"Verify generated Python code for arXiv:{clean_id}"
        )

    requested_model = (model_id or "").strip()
    is_gemini = requested_model.lower().startswith("gemini")

    llm_verified = False
    verifier_provider = "deterministic_fallback"
    resolved_model = sanitize_model_tag(requested_model or dynarot_decision.shinka_heavy_reasoning_model)
    llm_audit: Dict[str, Any] = {
        "status": "llm_disabled_by_caller" if not enable_llm else "not_executed",
        "verdict": "N/A",
        "paper_fidelity_score": 0.0,
        "correctness_score": 0.0,
        "security_score": 0.0,
        "summary": "",
        "grounded_span_coverage": [],
        "findings": [],
        "raw_response_excerpt": "",
        "ollama_guidance": {},
    }

    # Layer 2: LLM-as-a-Judge Verification
    if enable_llm:
        prompt = _build_llm_verification_prompt(paper, extractions, source_code, det_checks)
        if is_gemini:
            api_key = (
                os.environ.get("GEMINI_API_KEY", "").strip()
                or os.environ.get("LANGEXTRACT_API_KEY", "").strip()
            )
            if api_key:
                parsed, status, excerpt = _invoke_gemini_llm_judge(
                    model_id=requested_model,
                    api_key=api_key,
                    prompt=prompt,
                    timeout_sec=timeout_sec,
                )
                llm_audit["status"] = status
                llm_audit["raw_response_excerpt"] = excerpt
                if parsed is not None:
                    llm_verified = True
                    verifier_provider = "gemini"
                    resolved_model = requested_model
                    llm_audit.update(parsed)
            else:
                llm_audit["status"] = "gemini_api_key_missing_fallback"
                llm_audit["summary"] = (
                    "Requested Gemini verifier without GEMINI_API_KEY/LANGEXTRACT_API_KEY; "
                    "using deterministic verification gate."
                )
        else:
            ollama_online, installed_models = probe_ollama_models(endpoint=endpoint, timeout_sec=0.4)
            hw = detect_live_hardware()
            recommended_glm = size_glm_by_hardware(hw.free_ram_gb, hw.free_vram_mb)
            preferred_candidates = list(
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
            # QuorumRAG-calibrated model selection: strictly excludes embedding/vision/decision-only
            # (e.g. Bespoke-Nimble-9B) and memory-ineligible models from freeform JSON verification.
            chosen_installed: Optional[str] = None
            for pref_cand in preferred_candidates:
                hit = find_best_installed_chat_model(pref_cand, hw, installed_models)
                if hit is not None:
                    chosen_installed = hit
                    break

            recommended_pull = preferred_candidates[0]
            guidance = {
                "ollama_endpoint": endpoint,
                "ollama_online": ollama_online,
                "installed_models": installed_models,
                "recommended_model": recommended_pull,
                "recommended_glm_model": recommended_glm,
                "pull_command": f"ollama pull {recommended_pull}",
            }
            llm_audit["ollama_guidance"] = guidance

            if not ollama_online:
                llm_audit["status"] = "ollama_offline_fallback"
                llm_audit["summary"] = (
                    f"Local Ollama ({endpoint}) is offline; verified via deterministic AST & runtime test gate. "
                    f"Start Ollama (`ollama serve`) and pull `{recommended_pull}` for live LLM auditing."
                )
            elif chosen_installed is None:
                llm_audit["status"] = "model_not_installed_fallback"
                llm_audit["summary"] = (
                    f"Ollama is online (installed: {installed_models}) but no chat/completion model fitting "
                    f"available free memory ({hw.free_ram_gb} GB RAM / {hw.free_vram_mb} MB VRAM) is pulled yet. "
                    f"Run `ollama pull {recommended_pull}` (or `ollama pull glm4:9b`) to enable live LLM verification."
                )
            else:
                resolved_model = sanitize_model_tag(chosen_installed)
                parsed, status, excerpt = _invoke_ollama_llm_judge(
                    endpoint=endpoint,
                    model_tag=resolved_model,
                    prompt=prompt,
                    timeout_sec=timeout_sec,
                )
                llm_audit["status"] = status
                llm_audit["raw_response_excerpt"] = excerpt
                if parsed is not None:
                    llm_verified = True
                    verifier_provider = "ollama"
                    llm_audit.update(parsed)
                else:
                    llm_audit["summary"] = (
                        f"Ollama model `{resolved_model}` returned fallback status `{status}`; "
                        "relying on deterministic AST & runtime test gate."
                    )

    # Combine findings from Layer 1 and Layer 2
    combined_findings: List[Dict[str, str]] = list(det_checks["static_findings"])
    if llm_verified:
        for item in llm_audit.get("findings", []):
            combined_findings.append(item)

    recommendations = list(
        dict.fromkeys(
            f["recommendation"]
            for f in combined_findings
            if f.get("recommendation")
        )
    )

    # Fail-Closed Verdict Reconciliation
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

    if llm_verified:
        llm_avg = (
            float(llm_audit["paper_fidelity_score"])
            + float(llm_audit["correctness_score"])
            + float(llm_audit["security_score"])
        ) / 3.0
        overall_score = round((0.45 * det_score) + (0.55 * llm_avg), 2)
        if det_hard_fail:
            final_verdict = "FAIL"
            overall_score = min(overall_score, 45.0)
        elif llm_audit["verdict"] == "FAIL" or overall_score < 55.0:
            final_verdict = "FAIL"
        elif llm_audit["verdict"] == "WARN" or overall_score < 75.0 or not det_checks["zero_mock_single_token_indicators"]:
            final_verdict = "WARN"
        else:
            final_verdict = "PASS"
        mode = "llm_and_deterministic_verified"
    else:
        overall_score = det_score
        if det_hard_fail:
            final_verdict = "FAIL"
        elif overall_score >= 85.0 and det_checks["zero_mock_single_token_indicators"]:
            final_verdict = "PASS"
        elif overall_score >= 60.0:
            final_verdict = "WARN"
        else:
            final_verdict = "FAIL"
        mode = "deterministic_static_and_runtime_fallback"

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return CodeVerificationReport(
        arxiv_id=clean_id,
        title=title,
        code_stage=code_stage,
        verdict=final_verdict,
        overall_score=overall_score,
        verification_mode=mode,
        llm_verified=llm_verified,
        verifier_model=resolved_model,
        verifier_provider=verifier_provider,
        deterministic_checks=det_checks,
        llm_audit=llm_audit,
        findings=combined_findings,
        recommendations=recommendations,
        duration_ms=duration_ms,
    )
