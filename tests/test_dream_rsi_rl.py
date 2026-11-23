"""Tests for Dream-RSI RL Optimization (--rl / --dream-rsi, arXiv:2609.14858)."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from arxiv_to_code.cli import main as cli_main, normalize_cli_argv
from arxiv_to_code.dream_rsi_engine import (
    DreamPolicyOptimizer,
    ReplayWorldPool,
    TraceEnv,
    compute_student_t_ci95,
    create_baseline_heuristic_policy,
    detect_dream_rsi_backend,
    run_dream_rsi_empirical_benchmark,
    run_dream_rsi_rl_optimization,
)
from arxiv_to_code.evolve_engine import evaluate_python_candidate
from arxiv_to_code.langextract_engine import extract_grounded_breakouts, fetch_paper_by_id
from arxiv_to_code.mcp_server import optimize_code_with_dream_rsi_rl
from arxiv_to_code.python_code_generator import synthesize_production_python_code


def _sample_python_code() -> str:
    paper = fetch_paper_by_id("2608.15888")
    assert paper is not None
    exts, _ = extract_grounded_breakouts(paper)
    bundle = synthesize_production_python_code(paper, exts)
    return bundle["production_python_code"]


def test_explicit_backend_labeling_and_student_t_ci95() -> None:
    backend = detect_dream_rsi_backend()
    assert backend["environment_backend"] in (
        "native_dream_rsi_replay_world_model",
        "official_google_research_dream_rsi",
    )
    assert isinstance(backend["is_official_external_package"], bool)
    assert "2609.14858" in backend["reference_paper"]

    ci = compute_student_t_ci95([70.0, 72.0, 74.0, 71.0, 73.0])
    assert ci.n == 5
    assert abs(ci.mean - 72.0) < 1e-4
    assert ci.ci95_low < ci.mean < ci.ci95_high
    assert ci.ci95_half_width > 0.0


def test_trace_env_online_and_zero_execution_offline_replay(tmp_path: Path) -> None:
    seed_code = _sample_python_code()
    eval_calls = {"count": 0}

    def _counting_eval(src: str, cid: str):
        eval_calls["count"] += 1
        return evaluate_python_candidate(src, candidate_id=cid)

    policy = create_baseline_heuristic_policy()
    tree, stats = TraceEnv.run_online_rollout(
        seed_python_code=seed_code,
        arxiv_id="2608.15888",
        policy=policy,
        budget=8,
        cycle_index=1,
        seed=42,
        evaluator_fn=_counting_eval,
    )
    assert eval_calls["count"] >= 3
    assert len(tree.nodes) >= 3
    assert stats["best_fitness"] >= stats["seed_fitness"]

    # Persist to trace_pool.jsonl and reload cleanly
    pool = ReplayWorldPool([tree])
    jsonl_path = tmp_path / "trace_pool.jsonl"
    pool.save_jsonl(jsonl_path)
    reloaded = ReplayWorldPool.load_jsonl(jsonl_path)
    assert len(reloaded) == 1
    assert reloaded.total_recorded_nodes() == len(tree.nodes)

    # Offline dreaming over the frozen replay world MUST invoke zero live code executions
    calls_before_dream = eval_calls["count"]
    promoted, dream_summary = DreamPolicyOptimizer.dream_and_promote(
        incumbent=policy,
        pool=reloaded,
        budget=8,
        cycle_index=1,
        dream_candidates=16,
        seed=1000,
    )
    assert eval_calls["count"] == calls_before_dream
    assert dream_summary["live_code_executions_during_dreaming"] == 0
    assert dream_summary["offline_replay_rollouts_executed"] >= 16
    assert dream_summary["promoted_replay_reward"] >= dream_summary["incumbent_replay_reward"]

    # Standardized 5-baseline empirical benchmark with 95% Student-t CIs
    bench = run_dream_rsi_empirical_benchmark(
        pool=reloaded,
        learned_policy=promoted,
        budget=8,
        n_seeds=5,
    )
    assert bench["environment_backend"] == detect_dream_rsi_backend()["environment_backend"]
    for baseline_name in ("Disabled", "Random", "Heuristic", "NoLearning", "DreamRSI_Learned"):
        assert baseline_name in bench["baselines"]
        rw = bench["baselines"][baseline_name]["cumulative_replay_reward_ci95"]
        assert rw["n"] == 5
        assert rw["ci95_low"] <= rw["mean"] <= rw["ci95_high"]
    assert bench["superiority_deltas"]["learned_minus_disabled"] >= 0.0
    assert bench["superiority_deltas"]["learned_minus_random"] >= 0.0


def test_live_loopback_ollama_llm_dreaming_policy_rewrite() -> None:
    """Tests --rl-llm offline dreaming meta-policy proposal against a real local HTTP server."""
    seed_code = _sample_python_code()
    tree, _ = TraceEnv.run_online_rollout(
        seed_python_code=seed_code,
        arxiv_id="2608.15888",
        policy=create_baseline_heuristic_policy(),
        budget=6,
        cycle_index=1,
        seed=77,
    )
    pool = ReplayWorldPool([tree])

    class DreamLLMHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/api/tags":
                payload = json.dumps({"models": [{"name": "glm4:9b"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            if self.path == "/api/generate":
                proposed: Dict[str, Any] = {
                    "uct_exploration_c": 0.55,
                    "depth_bonus": 0.35,
                    "structural_vs_micro_prob": 0.65,
                    "parallel_batch_width": 3,
                    "early_stop_patience": 2,
                    "min_gain_threshold": 0.25,
                    "novelty_weight": 0.2,
                    "rationale": "Focus on structural leaps followed by rapid micro refinement and smart early stop.",
                }
                body = json.dumps({"response": json.dumps(proposed)}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), DreamLLMHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    endpoint = f"http://127.0.0.1:{port}"

    try:
        _, dream_summary = DreamPolicyOptimizer.dream_and_promote(
            incumbent=create_baseline_heuristic_policy(),
            pool=pool,
            budget=6,
            cycle_index=1,
            dream_candidates=10,
            use_llm_dreaming=True,
            ollama_endpoint=endpoint,
            llm_model_tag="glm4:9b",
        )
        assert dream_summary["llm_meta_policy_proposed"] is True
        assert dream_summary["live_code_executions_during_dreaming"] == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_rl_flags_aliases_workspace_and_mcp_tool(tmp_path: Path, capsys) -> None:
    # 1. Alias normalization
    assert normalize_cli_argv(["--dream-rsi", "--dreamrsi", "--use-rl", "--rl-evolve"]) == [
        "--rl",
        "--rl",
        "--rl",
        "--rl",
    ]

    # 2. Full CLI run with --rl + --rl-benchmark + persistent --rl-trace-pool
    out_dir = tmp_path / "rl_out"
    shared_pool = tmp_path / "shared_worlds"
    rc = cli_main(
        [
            "--arxiv-id",
            "2608.15888",
            "--rl",
            "--rl-cycles",
            "2",
            "--rl-budget",
            "6",
            "--rl-dream-candidates",
            "10",
            "--rl-benchmark",
            "--rl-seeds",
            "5",
            "--rl-trace-pool",
            str(shared_pool),
            "--output-dir",
            str(out_dir),
            "--no-progress",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "[DREAM-RSI RL]" in out
    assert "5-Baseline Matched-Seed Benchmark" in out

    # Verify organized rl/ artifacts & persistent trace pool
    assert (shared_pool / "trace_pool.jsonl").exists()
    rl_champs = list(out_dir.glob("arxiv_2608_15888_*/rl/rl_champion_arxiv_2608_15888.py"))
    rl_reports = list(out_dir.glob("arxiv_2608_15888_*/rl/rl_report.json"))
    rl_policies = list(out_dir.glob("arxiv_2608_15888_*/rl/promoted_policy.json"))
    rl_pools = list(out_dir.glob("arxiv_2608_15888_*/rl/trace_pool.jsonl"))
    assert len(rl_champs) == 1
    assert len(rl_reports) == 1
    assert len(rl_policies) == 1
    assert len(rl_pools) == 1

    rep = json.loads(rl_reports[0].read_text(encoding="utf-8"))
    assert rep["replay_world_pool_summary"]["final_worlds_count"] == 2
    assert rep["champion_metrics"]["tests_passed"] == 7
    index_md = (out_dir / "INDEX.md").read_text(encoding="utf-8")
    assert "Dream-RSI RL" in index_md

    # 3. Verify warm-start loading from persistent trace_pool.jsonl increases world count
    res_warm = run_dream_rsi_rl_optimization(
        seed_python_code=_sample_python_code(),
        arxiv_id="2608.15888",
        rl_cycles=1,
        rl_budget=4,
        dream_candidates=8,
        benchmark_seeds=5,
        trace_pool_dir=str(shared_pool),
        show_progress=False,
    )
    assert res_warm["replay_world_pool_summary"]["initial_worlds_loaded"] == 2
    assert res_warm["replay_world_pool_summary"]["final_worlds_count"] == 3

    # 4. MCP Tool 10 direct test
    mcp_res = json.loads(
        asyncio.run(
            optimize_code_with_dream_rsi_rl(
                arxiv_id="2608.15888",
                rl_cycles=2,
                rl_budget=6,
                dream_candidates=10,
                benchmark_seeds=5,
            )
        )
    )
    assert mcp_res["champion_metrics"]["tests_passed"] == 7
    assert "empirical_benchmark_95ci" in mcp_res
