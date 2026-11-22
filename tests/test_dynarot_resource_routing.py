"""Tests for ModelFusion DynaRot Dynamic Available Memory Law, Pareto Knee-Point, and UCB1 routing."""

from arxiv_to_code.cli import main as cli_main
from arxiv_to_code.dynarot import (
    DynaRotModelRouter,
    ParetoCandidate,
    calculate_provision_plan,
    compute_pareto_front,
    find_best_installed_chat_model,
    is_decision_only_model,
    is_synthesis_chat_model,
    model_fits_free_memory,
    sanitize_model_tag,
    select_knee_point,
    select_model_calibrated,
    size_gemma_by_hardware,
    size_glm_by_hardware,
    size_models_by_available_memory,
)
from arxiv_to_code.models import SystemHardwareResources


def test_dynamic_available_memory_law_all_tiers() -> None:
    # Tier 1: Free RAM >= 48 GB OR Free VRAM >= 24 GB (24,000 MB) -> qwen2.5:32b
    _, p1, s1, o1, ens1, _ = size_models_by_available_memory(free_ram_gb=52.0, free_vram_mb=8000.0)
    assert p1 == "qwen2.5:32b"
    assert s1 == "qwen2.5:32b"
    assert o1 == "qwen2.5:7b"
    assert ens1 == 7

    # Tier 1 via VRAM >= 24,000 MB even when Free RAM is 16 GB
    _, p1v, _, _, _, _ = size_models_by_available_memory(free_ram_gb=16.0, free_vram_mb=24576.0)
    assert p1v == "qwen2.5:32b"

    # Tier 2: Free RAM >= 24 GB OR Free VRAM >= 14 GB (14,000 MB) -> qwen2.5:14b
    _, p2, s2, o2, ens2, _ = size_models_by_available_memory(free_ram_gb=26.0, free_vram_mb=4000.0)
    assert p2 == "qwen2.5:14b"
    assert s2 == "qwen2.5:14b"
    assert o2 == "qwen2.5:3b"
    assert ens2 == 5

    # Tier 3: Free RAM >= 12 GB OR Free VRAM >= 4.5 GB (4,500 MB) -> qwen2.5:7b
    _, p3, s3, o3, ens3, _ = size_models_by_available_memory(free_ram_gb=15.0, free_vram_mb=2000.0)
    assert p3 == "qwen2.5:7b"
    assert s3 == "qwen2.5:7b"
    assert o3 == "qwen2.5:1.5b"
    assert ens3 == 4

    # Tier 4: Free RAM >= 6 GB OR Free VRAM >= 2 GB (2,000 MB) -> qwen2.5:3b
    _, p4, s4, o4, ens4, _ = size_models_by_available_memory(free_ram_gb=7.5, free_vram_mb=1000.0)
    assert p4 == "qwen2.5:3b"
    assert s4 == "qwen2.5:3b"
    assert o4 == "qwen2.5:1.5b"
    assert ens4 == 3

    # Tier 5: Free RAM < 6 GB -> qwen2.5:1.5b
    _, p5, s5, o5, ens5, _ = size_models_by_available_memory(free_ram_gb=4.2, free_vram_mb=512.0)
    assert p5 == "qwen2.5:1.5b"
    assert s5 == "qwen2.5:1.5b"
    assert o5 == "qwen2.5:0.5b"
    assert ens5 == 2


def test_never_append_coder_suffix() -> None:
    assert sanitize_model_tag("qwen2.5-coder:7b") == "qwen2.5:7b"
    assert sanitize_model_tag("qwen2.5-coder:32b") == "qwen2.5:32b"
    assert sanitize_model_tag("qwen2.5-coder") == "qwen2.5"


def test_pareto_front_and_knee_selection() -> None:
    candidates = [
        ParetoCandidate("high-quality", [0.96, 1.8, 0.85, 18.0]),
        ParetoCandidate("balanced-knee", [0.91, 1.9, 0.94, 5.0]),
        ParetoCandidate("dominated-slow", [0.70, 0.5, 0.50, 20.0]),
    ]
    front = compute_pareto_front(candidates)
    front_ids = [c.model_id for c in front]
    assert "high-quality" in front_ids
    assert "balanced-knee" in front_ids
    assert "dominated-slow" not in front_ids

    knee = select_knee_point(front)
    assert knee is not None
    assert knee.model_id == "balanced-knee"


def test_dynarot_router_and_provisioning_plan() -> None:
    router = DynaRotModelRouter()
    decision = router.route(
        prompt="Synthesize and evolve Python algorithm from arXiv:2608.15888",
        override_free_ram_gb=30.0,
        override_free_vram_mb=8000.0,
    )
    assert decision.primary_model == "qwen2.5:14b"
    assert decision.shinka_heavy_reasoning_model == "qwen2.5:14b"
    assert decision.openevolve_fast_mutation_model == "qwen2.5:3b"
    assert "-coder" not in decision.primary_model

    prov = router.provision_models_for_resources(
        override_free_ram_gb=16.0,
        override_free_vram_mb=6000.0,
        auto_pull=False,
    )
    assert prov["dynarot_decision"]["primary_model"] == "qwen2.5:7b"
    assert "ollama pull qwen2.5:7b" in prov["guidance_card"]["terminal_commands"]


def test_quorumrag_parity_glm_sizing_disk_downsizing_and_calibrated_selection(capsys) -> None:
    # 1. GLM family sizing strictly by runtime free RAM / free VRAM
    assert size_glm_by_hardware(free_ram_gb=20.0, free_vram_mb=0.0) == "glm-4.7-flash"
    assert size_glm_by_hardware(free_ram_gb=8.0, free_vram_mb=16_384.0) == "glm-4.7-flash"
    assert size_glm_by_hardware(free_ram_gb=14.0, free_vram_mb=0.0) == "glm4:9b"
    assert size_glm_by_hardware(free_ram_gb=8.0, free_vram_mb=0.0) == "qwen2.5:3b"
    assert size_glm_by_hardware(free_ram_gb=4.0, free_vram_mb=0.0) == "qwen2.5:1.5b"

    # 2. Gemma family sizing strictly by runtime free RAM / free VRAM
    assert size_gemma_by_hardware(free_ram_gb=22.0, free_vram_mb=0.0) == "gemma4:26b"
    assert size_gemma_by_hardware(free_ram_gb=14.0, free_vram_mb=0.0) == "gemma2:9b"
    assert size_gemma_by_hardware(free_ram_gb=7.0, free_vram_mb=0.0) == "gemma3:4b"
    assert size_gemma_by_hardware(free_ram_gb=4.0, free_vram_mb=0.0) == "gemma2:2b"

    # 3. Decision-only (Bespoke-Nimble-9B), embedding, and vision models excluded from chat/JSON synthesis
    assert is_decision_only_model("hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q4_K_M") is True
    assert is_synthesis_chat_model("hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q4_K_M") is False
    assert is_synthesis_chat_model("nomic-embed-text:latest") is False
    assert is_synthesis_chat_model("glm-ocr:latest") is False
    assert is_synthesis_chat_model("glm-4.7-flash") is True
    assert is_synthesis_chat_model("glm4:9b") is True

    # 4. Calibrated installed selection respects free RAM/VRAM & prefers matching installed GLM model
    hw_14g = SystemHardwareResources(
        total_ram_gb=32.0,
        free_ram_gb=14.0,
        cpu_cores=16,
        gpu_name=None,
        gpu_vram_total_gb=0.0,
        gpu_vram_free_gb=0.0,
        free_disk_gb=50.0,
    )
    installed = [
        "hf.co/ggml-org/Bespoke-Nimble-9B-v3-GGUF:Q4_K_M",
        "qwen2.5:32b",  # ineligible on 14 GB free RAM
        "glm4:9b",      # eligible!
        "nomic-embed-text:latest",
    ]
    assert model_fits_free_memory("qwen2.5:32b", 14.0, 0.0) is False
    assert model_fits_free_memory("glm4:9b", 14.0, 0.0) is True
    assert find_best_installed_chat_model("glm4:9b", hw_14g, installed) == "glm4:9b"
    fam, tag, _ = select_model_calibrated("glm", "extract paper spans", hw_14g, installed)
    assert fam == "glm" and tag == "glm4:9b"

    # 5. Pre-provisioning disk headroom downsizing (QuorumRAG ModelProvisionPlan)
    hw_tight_disk = SystemHardwareResources(
        total_ram_gb=64.0,
        free_ram_gb=52.0,
        cpu_cores=16,
        gpu_name=None,
        gpu_vram_total_gb=0.0,
        gpu_vram_free_gb=0.0,
        free_disk_gb=7.0,  # Enough for glm4:9b (6 GB), NOT enough for glm-4.7-flash (18 GB)
    )
    plan_ok = calculate_provision_plan("glm4:9b", hw_tight_disk, [])
    assert plan_ok["effective_tag"] == "glm4:9b"
    assert plan_ok["downsized_for_resources"] is False
    assert plan_ok["disk_sufficient"] is True

    plan_down = calculate_provision_plan("glm-4.7-flash", hw_tight_disk, [])
    assert plan_down["downsized_for_resources"] is True
    assert plan_down["effective_tag"] != "glm-4.7-flash"

    # 6. CLI `--dynarot --model-family glm` selects GLM tier
    rc = cli_main(["--dynarot", "--model-family", "glm", "--no-progress"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "recommended_glm_model" in out
    assert "provision_plan" in out
