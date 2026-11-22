"""
Organized Output Directory & Manifest Manager for ArXiv-to-Code (`generated/`).

Default layout created under `generated/` (or custom `--output-dir`):
    generated/
    ├── INDEX.md                           # Human-readable table of all generated papers, scores, speedups & paths
    ├── MANIFEST.json                      # Machine-readable catalog of every paper run & output path
    └── arxiv_<id>_<short_slug>/
        ├── README.md                      # Paper summary, rank breakdown, LangExtract spans & run commands
        ├── metadata.json                  # Full arXiv metadata, rank score & grounded extractions
        ├── extraction_results.jsonl       # Google LangExtract JSONL output
        ├── langextract_report.html        # Interactive Google LangExtract HTML visualization
        ├── verification_report.json       # Created when --verify-llm is active (Deterministic + LLM Judge)
        ├── src/
        │   └── arxiv_<id>_module.py       # Synthesized production Python module (# EVOLVE-BLOCK)
        ├── eval/
        │   └── evaluate_arxiv_<id>.py     # Standalone deterministic scoring script
        └── evolved/                       # Created when --evolves is active
            ├── champion_arxiv_<id>.py     # Best evolved champion Python module
            ├── evolution_report.json      # Full ShinkaEvolve + OpenEvolve MAP-Elites metrics & round history
            └── elites/                    # Individual ShinkaEvolve & OpenEvolve island elite programs
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .langextract_engine import AnnotatedDocument, normalize_arxiv_id, save_langextract_artifacts


def _slugify_title(title: str, max_words: int = 4) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", title.lower())
    stop = {"a", "an", "the", "of", "for", "in", "on", "to", "with", "and", "via", "from"}
    meaningful = [t for t in tokens if t not in stop][:max_words]
    if not meaningful:
        meaningful = tokens[:max_words] or ["paper"]
    return "_".join(meaningful)[:42]


def prepare_paper_workspace(
    output_root: str,
    paper: Dict[str, Any],
) -> Dict[str, Path]:
    """Creates the organized subfolder structure for a single paper inside `output_root`."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    clean_id = normalize_arxiv_id(paper.get("arxiv_id", "custom")) or "custom"
    id_slug = clean_id.replace(".", "_").replace("-", "_")
    title_slug = _slugify_title(paper.get("title", "paper"))
    folder_name = f"arxiv_{id_slug}_{title_slug}"

    paper_dir = root / folder_name
    src_dir = paper_dir / "src"
    eval_dir = paper_dir / "eval"
    evolved_dir = paper_dir / "evolved"
    elites_dir = evolved_dir / "elites"

    for d in (paper_dir, src_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "paper_dir": paper_dir,
        "src_dir": src_dir,
        "eval_dir": eval_dir,
        "evolved_dir": evolved_dir,
        "elites_dir": elites_dir,
        "folder_name": Path(folder_name),
    }


def write_organized_paper_artifacts(
    output_root: str,
    paper: Dict[str, Any],
    extractions: List[Dict[str, Any]],
    annotated_doc: AnnotatedDocument,
    code_bundle: Dict[str, Any],
    evolution_report: Optional[Dict[str, Any]] = None,
    dynarot_decision: Optional[Dict[str, Any]] = None,
    verification_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Writes all organized paper artifacts (`src/`, `eval/`, `evolved/`, `metadata.json`,
    `extraction_results.jsonl`, `langextract_report.html`, `README.md`) and updates
    `MANIFEST.json` + `INDEX.md` at `output_root`.
    """
    paths = prepare_paper_workspace(output_root, paper)
    root = paths["root"]
    paper_dir = paths["paper_dir"]
    src_dir = paths["src_dir"]
    eval_dir = paths["eval_dir"]
    evolved_dir = paths["evolved_dir"]

    clean_id = code_bundle.get("clean_arxiv_id") or normalize_arxiv_id(paper.get("arxiv_id", "custom"))
    id_slug = clean_id.replace(".", "_").replace("-", "_")

    # 1. Save synthesized Python module & deterministic evaluator
    module_path = src_dir / f"arxiv_{id_slug}_module.py"
    module_path.write_text(code_bundle["production_python_code"], encoding="utf-8")

    scorer_path = eval_dir / f"evaluate_arxiv_{id_slug}.py"
    scorer_path.write_text(code_bundle["deterministic_scoring_script"], encoding="utf-8")

    # 2. Save Google LangExtract JSONL + Interactive HTML report
    lx_paths = save_langextract_artifacts(
        annotated_doc=annotated_doc,
        output_dir=paper_dir,
        html_filename="langextract_report.html",
        jsonl_filename="extraction_results.jsonl",
    )

    # 3. Save evolution report & champion pointer if --evolves ran
    champion_path_str = ""
    evo_report_path_str = ""
    if evolution_report is not None:
        evolved_dir.mkdir(parents=True, exist_ok=True)
        champ_dest = evolved_dir / f"champion_arxiv_{id_slug}.py"
        champ_src = evolution_report.get("champion_candidate", {}).get("source_code", "")
        if champ_src:
            champ_dest.write_text(champ_src, encoding="utf-8")
            champion_path_str = str(champ_dest)
            evolution_report["champion_file_path"] = champion_path_str

        evo_json_path = evolved_dir / "evolution_report.json"
        evo_json_path.write_text(json.dumps(evolution_report, indent=2), encoding="utf-8")
        evo_report_path_str = str(evo_json_path)

    # 3b. Save LLM + Deterministic verification report if --verify-llm ran
    verification_report_path_str = ""
    if verification_report is not None:
        ver_json_path = paper_dir / "verification_report.json"
        ver_json_path.write_text(json.dumps(verification_report, indent=2), encoding="utf-8")
        verification_report_path_str = str(ver_json_path)

    # 4. Save metadata.json
    metadata_payload = {
        "arxiv_id": paper.get("arxiv_id", clean_id),
        "clean_arxiv_id": clean_id,
        "title": paper.get("title", ""),
        "published": paper.get("published", ""),
        "url": paper.get("url", ""),
        "code_repo_url": paper.get("code_repo_url", code_bundle.get("code_repo_url", "")),
        "rank_score": paper.get("rank_score", 0.0),
        "priority_tier": paper.get("priority_tier", ""),
        "score_breakdown": paper.get("score_breakdown", {}),
        "grounded_extractions_count": len(extractions),
        "grounded_extractions": extractions,
        "dynarot_routing": dynarot_decision or {},
        "verification_summary": (
            {
                "verdict": verification_report.get("verdict"),
                "overall_score": verification_report.get("overall_score"),
                "verification_mode": verification_report.get("verification_mode"),
                "llm_verified": verification_report.get("llm_verified"),
                "verifier_model": verification_report.get("verifier_model"),
            }
            if verification_report is not None
            else {}
        ),
        "artifacts": {
            "python_module": str(module_path),
            "evaluator_script": str(scorer_path),
            "langextract_html": lx_paths["html_path"],
            "langextract_jsonl": lx_paths["jsonl_path"],
            "evolved_champion": champion_path_str,
            "evolution_report": evo_report_path_str,
            "verification_report": verification_report_path_str,
        },
        "generated_at_epoch": round(time.time(), 2),
    }
    metadata_path = paper_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    # 5. Write per-paper README.md
    evo_section = ""
    if evolution_report is not None:
        base_m = evolution_report["baseline_seed_metrics"]
        champ_m = evolution_report["champion_metrics"]
        evo_section = (
            "## Evolution Results (`ShinkaEvolve` + `OpenEvolve`)\n"
            f"- **Mode**: `{evolution_report.get('evolve_mode', 'hybrid')}` "
            f"(Rounds / `--evolve-count`: `{evolution_report.get('evolve_count', 1)}`)\n"
            f"- **Baseline Fitness**: `{base_m['total_fitness']}/100` (`{base_m['execution_time_us']} us`, `{base_m['peak_memory_kib']} KiB`)\n"
            f"- **Champion Fitness**: `{champ_m['total_fitness']}/100` (`{champ_m['execution_time_us']} us`, `{champ_m['peak_memory_kib']} KiB`)\n"
            f"- **Speedup Ratio**: `{evolution_report.get('speedup_ratio', 1.0)}x`\n"
            f"- **Occupied MAP-Elites Niches**: `{evolution_report['openevolve_map_elites_summary']['occupied_niches_count']}/16`\n"
            f"- **Run Champion Directly**:\n  ```powershell\n  python \"{champion_path_str}\"\n  ```\n\n"
        )

    verify_section = ""
    if verification_report is not None:
        v_llm = verification_report.get("llm_audit", {})
        v_det = verification_report.get("deterministic_checks", {})
        verify_section = (
            "## LLM & Deterministic Code Verification (`--verify-llm`)\n"
            f"- **Verdict**: `{verification_report.get('verdict', 'N/A')}` (`{verification_report.get('overall_score', 0.0)}/100`)\n"
            f"- **Mode**: `{verification_report.get('verification_mode', '')}` "
            f"(Model: `{verification_report.get('verifier_model', '')}`, LLM Executed: `{verification_report.get('llm_verified', False)}`)\n"
            f"- **Deterministic Gate**: Tests `{v_det.get('runtime_tests_passed', 0)}/{v_det.get('runtime_tests_total', 4)}`, "
            f"HMAC Constant-Time=`{v_det.get('constant_time_hmac_used', False)}`, "
            f"Zero Dangerous Calls=`{v_det.get('zero_dangerous_calls', False)}`\n"
            + (
                f"- **LLM Judge Scores**: Fidelity `{v_llm.get('paper_fidelity_score', 0.0)}/100` | "
                f"Correctness `{v_llm.get('correctness_score', 0.0)}/100` | "
                f"Security `{v_llm.get('security_score', 0.0)}/100`\n"
                f"- **LLM Summary**: {v_llm.get('summary', '')}\n"
                if verification_report.get("llm_verified")
                else f"- **LLM Status**: `{v_llm.get('status', 'offline')}` — {v_llm.get('summary', '')}\n"
            )
            + "\n"
        )

    spans_md = "\n".join(
        f"- **`{e['extraction_class']}`** `[{e['source_char_start']}:{e['source_char_end']}]`: "
        f"`{e['source_quote']}` — {e['attack_vector']}"
        for e in extractions
    )

    readme_text = (
        f"# `arXiv:{paper.get('arxiv_id', clean_id)}` — {paper.get('title', '')}\n\n"
        f"- **Rank Score**: `{paper.get('rank_score', 0.0)}/100` (`{paper.get('priority_tier', 'N/A')}`)\n"
        f"- **Source URL**: {paper.get('url', 'N/A')}\n"
        f"- **Code Repo**: {paper.get('code_repo_url') or 'N/A'}\n\n"
        "## Directory Contents\n"
        f"- [`src/arxiv_{id_slug}_module.py`](src/arxiv_{id_slug}_module.py) — Synthesized production Python module (`# EVOLVE-BLOCK`)\n"
        f"- [`eval/evaluate_arxiv_{id_slug}.py`](eval/evaluate_arxiv_{id_slug}.py) — Deterministic multi-objective scoring script\n"
        "- [`langextract_report.html`](langextract_report.html) — Interactive Google LangExtract source-grounded HTML report\n"
        "- [`extraction_results.jsonl`](extraction_results.jsonl) — Google LangExtract JSONL output\n"
        "- [`metadata.json`](metadata.json) — Full paper metadata, rank breakdown & grounded extractions\n"
        + (
            "- [`verification_report.json`](verification_report.json) — Deterministic + LLM-as-a-Judge verification scorecard\n"
            if verification_report is not None
            else ""
        )
        + (
            f"- [`evolved/champion_arxiv_{id_slug}.py`](evolved/champion_arxiv_{id_slug}.py) — Evolved champion Python module\n"
            "- [`evolved/evolution_report.json`](evolved/evolution_report.json) — Full ShinkaEvolve + OpenEvolve metrics\n"
            if evolution_report is not None
            else ""
        )
        + "\n## How to Run\n"
        f"```powershell\npython \"{module_path}\"\n```\n\n"
        + evo_section
        + verify_section
        + "## Google LangExtract Grounded Spans\n"
        + spans_md
        + "\n"
    )
    readme_path = paper_dir / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")

    # 6. Update root MANIFEST.json and INDEX.md so user always knows what is inside `generated/`
    manifest_path = root / "MANIFEST.json"
    manifest: Dict[str, Any] = {"output_root": str(root), "papers": {}}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("papers"), dict):
                manifest = loaded
        except Exception:
            pass

    rel_folder = paper_dir.name
    manifest["papers"][clean_id] = {
        "arxiv_id": paper.get("arxiv_id", clean_id),
        "title": paper.get("title", ""),
        "folder": rel_folder,
        "rank_score": paper.get("rank_score", 0.0),
        "priority_tier": paper.get("priority_tier", ""),
        "grounded_spans": len(extractions),
        "python_module": f"{rel_folder}/src/arxiv_{id_slug}_module.py",
        "langextract_html": f"{rel_folder}/langextract_report.html",
        "evolved": evolution_report is not None,
        "evolve_count": evolution_report.get("evolve_count", 0) if evolution_report else 0,
        "champion_fitness": (
            evolution_report["champion_metrics"]["total_fitness"] if evolution_report else None
        ),
        "speedup_ratio": evolution_report.get("speedup_ratio") if evolution_report else None,
        "champion_module": (
            f"{rel_folder}/evolved/champion_arxiv_{id_slug}.py" if evolution_report else ""
        ),
        "verified": verification_report is not None,
        "verification_verdict": verification_report.get("verdict") if verification_report else None,
        "verification_score": verification_report.get("overall_score") if verification_report else None,
        "llm_verified": verification_report.get("llm_verified", False) if verification_report else False,
        "verification_report": (
            f"{rel_folder}/verification_report.json" if verification_report else ""
        ),
        "updated_at": round(time.time(), 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    index_lines = [
        "# Generated ArXiv-to-Code Workspace Index",
        "",
        "This directory is automatically organized by `arixv_to_code.py`. Each paper has its own subfolder containing `src/`, `eval/`, `langextract_report.html`, `extraction_results.jsonl`, `metadata.json`, `README.md`, and (when enabled) `evolved/` and `verification_report.json`.",
        "",
        "| arXiv ID | Score | Title | Folder | Python Module | Evolved Champion | Speedup | Verification | LangExtract HTML |",
        "|:---|:---:|:---|:---|:---|:---|:---:|:---:|:---|",
    ]
    for pid, entry in sorted(
        manifest["papers"].items(),
        key=lambda kv: float(kv[1].get("rank_score", 0.0)),
        reverse=True,
    ):
        t_short = entry["title"][:42] + ("..." if len(entry["title"]) > 42 else "")
        champ_cell = (
            f"[`champion`]({entry['champion_module']}) ({entry['champion_fitness']}/100)"
            if entry.get("evolved") and entry.get("champion_module")
            else "—"
        )
        spd_cell = f"{entry['speedup_ratio']}x" if entry.get("speedup_ratio") else "—"
        if entry.get("verified") and entry.get("verification_report"):
            v_tag = "LLM+Det" if entry.get("llm_verified") else "Det"
            ver_cell = (
                f"[`{entry['verification_verdict']}`]({entry['verification_report']}) "
                f"({entry['verification_score']}/100 {v_tag})"
            )
        else:
            ver_cell = "—"
        index_lines.append(
            f"| `{entry['arxiv_id']}` | `{entry['rank_score']}/100` | {t_short} | "
            f"[`{entry['folder']}/`]({entry['folder']}/README.md) | "
            f"[`module`]({entry['python_module']}) | {champ_cell} | {spd_cell} | {ver_cell} | "
            f"[`HTML`]({entry['langextract_html']}) |"
        )

    index_path = root / "INDEX.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    return {
        "output_root": str(root),
        "paper_dir": str(paper_dir),
        "python_module": str(module_path),
        "evaluator_script": str(scorer_path),
        "langextract_html": lx_paths["html_path"],
        "langextract_jsonl": lx_paths["jsonl_path"],
        "metadata_json": str(metadata_path),
        "paper_readme": str(readme_path),
        "evolved_champion": champion_path_str,
        "evolution_report": evo_report_path_str,
        "verification_report": verification_report_path_str,
        "manifest_json": str(manifest_path),
        "index_md": str(index_path),
    }
