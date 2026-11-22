"""Tests for arXiv search, 3-class Google LangExtract source grounding, and multi-factor ranking."""

from pathlib import Path
from arxiv_to_code.langextract_engine import (
    audit_defense_coverage_multi_target,
    compute_article_rank,
    extract_grounded_breakouts,
    extract_text_from_document_path,
    fetch_paper_by_id,
    generate_interactive_html,
    normalize_arxiv_id,
    search_and_rank_articles,
)


def test_normalize_arxiv_id() -> None:
    assert normalize_arxiv_id("arXiv:2610.09167v2") == "2610.09167"
    assert normalize_arxiv_id("https://arxiv.org/abs/2603.02277v3") == "2603.02277"
    assert normalize_arxiv_id("2608.15888") == "2608.15888"


def test_3class_langextract_character_interval_grounding() -> None:
    paper = fetch_paper_by_id("2603.02277")
    assert paper is not None
    extractions, annotated_doc = extract_grounded_breakouts(paper)
    assert len(extractions) >= 3
    for ext in extractions:
        s = ext["source_char_start"]
        e = ext["source_char_end"]
        quote = ext["source_quote"]
        assert paper["summary"][s:e] == quote
    rank = compute_article_rank(paper, extractions)
    assert rank["total_score"] >= 80.0
    assert rank["priority_tier"] == "TIER-1 CRITICAL"
    assert rank["is_novel_to_codebase"] is True
    assert rank["enhances_codebase"] is True
    assert rank["has_code_repo"] is True


def test_strict_novelty_rejection_of_implemented_paper() -> None:
    paper = fetch_paper_by_id("2610.09167")
    assert paper is not None
    exts, _ = extract_grounded_breakouts(paper)
    rank = compute_article_rank(paper, exts)
    assert rank["is_novel_to_codebase"] is False
    assert rank["enhances_codebase"] is False
    assert rank["novelty_delta_score"] == 0.0


def test_search_and_rank_descending_order() -> None:
    ranked = search_and_rank_articles(
        query="container sandbox escape",
        max_results=4,
        only_novel_enhancements=True,
    )
    assert len(ranked) >= 1
    scores = [r["rank_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    for item in ranked:
        assert item["is_novel_to_codebase"] is True


def test_html_report_and_document_extraction(tmp_path: Path) -> None:
    paper = fetch_paper_by_id("2608.15888")
    assert paper is not None
    _, annotated_doc = extract_grounded_breakouts(paper)
    html_file = tmp_path / "grounded_report.html"
    saved = generate_interactive_html(annotated_doc, str(html_file))
    assert Path(saved).exists()
    assert html_file.stat().st_size > 250

    sample_doc = tmp_path / "sample_paper.txt"
    sample_doc.write_text(
        "We present a container sandbox escape defense with conformal martingale bounds.",
        encoding="utf-8",
    )
    extracted_txt = extract_text_from_document_path(str(sample_doc))
    assert "martingale" in extracted_txt


def test_audit_defense_coverage_multi_target() -> None:
    res = audit_defense_coverage_multi_target("2603.02277")
    assert res["success"] is True
    assert res["overall_coverage_percentage"] == 100.0
    assert len(res["codebase_enhancement_recommendations"]) >= 1


def test_real_google_langextract_artifacts_and_ollama_glm_provider(tmp_path: Path) -> None:
    from arxiv_to_code.langextract_engine import save_langextract_artifacts

    paper = fetch_paper_by_id("2603.02277")
    assert paper is not None
    _, annotated_doc = extract_grounded_breakouts(paper)

    saved = save_langextract_artifacts(annotated_doc, tmp_path)
    jsonl_path = Path(saved["jsonl_path"])
    html_path = Path(saved["html_path"])
    assert jsonl_path.exists() and jsonl_path.stat().st_size > 100
    assert html_path.exists() and html_path.stat().st_size > 1000
    # Verify real Google LangExtract interactive visualization was rendered (not minimal fallback)
    html_text = html_path.read_text(encoding="utf-8")
    assert "<script" in html_text.lower() or "highlight" in html_text.lower()

    # Verify direct OllamaLanguageModel accepts GLM tags (bypassing static OLLAMA_PATTERNS limitation)
    from langextract.providers.ollama import OllamaLanguageModel
    from langextract.core.data import FormatType

    glm_model = OllamaLanguageModel(
        model_id="glm-4.7-flash",
        model_url="http://127.0.0.1:11434",
        format_type=FormatType.JSON,
        timeout=30,
    )
    assert glm_model is not None
