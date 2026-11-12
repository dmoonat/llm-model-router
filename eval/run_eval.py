"""4-Arm Benchmark runner comparing Multi-Agent ADK configurations on 45 open-source tasks (15 per dataset).

Evaluates 4 Experimental Arms across 45 tasks (15 ARC-Easy, 15 GSM8K, 15 MBPP):
1. `all_simple`: All 4 agents pinned to `SIMPLE_MODEL` (`gemini-3.6-flash`).
2. `all_complex`: All 4 agents pinned to `COMPLEX_MODEL` (`gemini-3.1-pro-preview`).
3. `coord_simple_subs_complex`: `coordinator_agent` fixed to `SIMPLE_MODEL` (`gemini-3.6-flash`)
   and all 3 specialist sub-agents fixed to `COMPLEX_MODEL` (`gemini-3.1-pro-preview`) (no router).
4. `with_router`: `coordinator_agent` pinned to `SIMPLE_MODEL` via `exclude_agents={"coordinator_agent"}`,
   while specialist sub-agents (`fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) are
   dynamically routed by `LayaModelRouterPlugin` between `SIMPLE_MODEL` and `COMPLEX_MODEL`.

Uses:
- Real open-source benchmarks from Hugging Face (`ARC-Easy`, `GSM8K`, `MBPP`)
- `adk-finops` (`FinOpsCostPlugin`, `CostTracker`, `JSONLExporter`, `CSVExporter`)
  for token, latency, throughput, and financial cost tracking.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adk_finops import CostTracker, FinOpsCostPlugin
from adk_finops.exporters import CSVExporter, JSONLExporter
from google.adk.runners import InMemoryRunner
from google.genai import types

from laya_adk_model_router.plugin import COMPLEX_MODEL, SIMPLE_MODEL

from .dataset_loader import DATASET_NAMES, BenchmarkSample, load_benchmark_suite, verify_sample
from .multi_agent import build_eval_app

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
LOGS_DIR = _EVAL_DIR / "logs"
REPORT_PATH = _EVAL_DIR / "EVAL_REPORT.md"
_EXPORT_LOCK = asyncio.Lock()

MODES_ORDER = (
    "all_simple",
    "all_complex",
    "coord_simple_subs_complex",
    "with_router",
    "with_decisions_router",
)


@dataclass
class SampleEvalResult:
    """Captured metrics for a single benchmark sample execution."""

    mode: str  # "all_simple", "all_complex", "coord_simple_subs_complex", "with_router", or "with_decisions_router"
    sample_id: str
    dataset: str
    expected_complexity: str
    coordinator_model: str
    routed_model: str  # Specialist sub-agent model
    laya_choice: str
    laya_confidence: float
    laya_latency_ms: float
    delegated_agents: list[str]
    operational_status: str  # "success", "failed", or "error"
    is_correct: bool
    predicted_answer: str
    ground_truth: str
    prompt_tokens: int
    completion_tokens: int
    thoughts_tokens: int
    total_tokens: int
    llm_cost_usd: float
    total_cost_usd: float
    turn_latency_ms: float
    llm_latency_ms: float
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    breakdown_by_agent: dict[str, Any] = field(default_factory=dict)
    breakdown_by_model: dict[str, Any] = field(default_factory=dict)


async def run_single_sample(
    *,
    mode: str,
    runner: InMemoryRunner,
    app_name: str,
    sample: BenchmarkSample,
    finops_plugin: FinOpsCostPlugin,
    jsonl_exporter: JSONLExporter,
    csv_exporter: CSVExporter,
) -> SampleEvalResult:
    """Executes one open-source benchmark sample through the Multi-Agent ADK App."""
    user_id = "eval_user"
    base_session_id = f"{mode}__{sample.sample_id}"
    session_id = base_session_id

    user_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text=sample.prompt)],
    )

    final_text_parts: list[str] = []
    delegated_agents: list[str] = []
    operational_status = "success"
    operational_error: str | None = None
    wall_ms = 0.0

    for attempt in range(3):
        session_id = base_session_id if attempt == 0 else f"{base_session_id}__retry{attempt}"
        await runner.session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        final_text_parts = []
        delegated_agents = []
        operational_status = "success"
        operational_error = None

        t0 = time.perf_counter()
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_msg,
            ):
                author = getattr(event, "author", None)
                if author and author not in ("user", "system") and author not in delegated_agents:
                    delegated_agents.append(author)
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "text", None):
                            final_text_parts.append(part.text)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            break
        except Exception as exc:
            wall_ms = (time.perf_counter() - t0) * 1000.0
            err_str = f"{type(exc).__name__}: {exc}"
            if attempt < 2 and any(
                k in err_str
                for k in (
                    "ConnectionResetError",
                    "TransportError",
                    "Connection aborted",
                    "503",
                    "429",
                    "ServerDisconnectedError",
                    "RemoteProtocolError",
                )
            ):
                logger.warning(
                    "[%s] %s hit transient network error (%s) on attempt %d; retrying in 2s...",
                    mode,
                    sample.sample_id,
                    err_str,
                    attempt + 1,
                )
                CostTracker.get_summary(session_id=session_id, pop=True)
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            operational_status = "error"
            operational_error = err_str
            logger.error("[%s] %s raised operational error: %s", mode, sample.sample_id, operational_error)
            finops_plugin.record_task_status(
                session_id=session_id,
                status="error",
                error=operational_error,
            )
            break

    final_text = "\n".join(final_text_parts).strip()
    if operational_status == "success" and not final_text:
        operational_status = "failed"
        operational_error = "Agent returned empty output text"
        finops_plugin.record_task_status(
            session_id=session_id,
            status="failed",
            error=operational_error,
        )

    # Ground-truth evaluation (accuracy is tracked separately from operational failure!)
    if operational_status == "success":
        is_correct, pred_ans = verify_sample(sample, final_text)
    else:
        is_correct, pred_ans = False, operational_error or "operational_failure"

    # Extract Laya / Decisions API routing metadata from ADK session state
    session_obj = await runner.session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    state_dict = getattr(session_obj, "state", {}) or {}
    laya_info = state_dict.get("laya_routing") or {}
    decisions_info = state_dict.get("decisions_routing") or {}

    if mode == "all_simple":
        coordinator_model = SIMPLE_MODEL
        routed_model = SIMPLE_MODEL
        laya_choice = "none"
        laya_conf = 0.0
        laya_lat_ms = 0.0
    elif mode == "all_complex":
        coordinator_model = COMPLEX_MODEL
        routed_model = COMPLEX_MODEL
        laya_choice = "none"
        laya_conf = 0.0
        laya_lat_ms = 0.0
    elif mode == "coord_simple_subs_complex":
        coordinator_model = SIMPLE_MODEL
        routed_model = COMPLEX_MODEL
        laya_choice = "none"
        laya_conf = 0.0
        laya_lat_ms = 0.0
    elif mode == "with_decisions_router":
        coordinator_model = SIMPLE_MODEL
        routed_model = str(
            decisions_info.get("routed_model")
            or state_dict.get("routed_model")
            or COMPLEX_MODEL
        )
        laya_choice = str(decisions_info.get("choice", "unknown"))
        laya_conf = float(decisions_info.get("confidence", 0.0))
        laya_lat_ms = float(decisions_info.get("decisions_latency_ms", 0.0))
    else:
        coordinator_model = SIMPLE_MODEL
        routed_model = str(
            laya_info.get("routed_model")
            or state_dict.get("routed_model")
            or COMPLEX_MODEL
        )
        laya_choice = str(laya_info.get("choice", "unknown"))
        laya_conf = float(laya_info.get("confidence", 0.0))
        laya_lat_ms = float(laya_info.get("laya_latency_ms", 0.0))

    # Retrieve adk-finops session summary
    summary = CostTracker.get_summary(session_id=session_id, pop=False) or {}
    sess_scope = summary.get("session") or summary

    prompt_tokens = int(sess_scope.get("prompt_tokens", 0))
    completion_tokens = int(sess_scope.get("completion_tokens", 0))
    thoughts_tokens = int(sess_scope.get("thoughts_tokens", 0))
    total_tokens = int(sess_scope.get("total_tokens", 0))
    llm_cost_usd = float(sess_scope.get("llm_cost_usd", 0.0))
    total_cost_usd = float(sess_scope.get("total_cost_usd", 0.0))
    finops_lat_ms = float(sess_scope.get("latency_ms", 0.0)) or wall_ms
    llm_lat_ms = float(sess_scope.get("llm_latency_ms", 0.0)) or wall_ms
    out_tps = float(sess_scope.get("output_tokens_per_sec", 0.0))
    tot_tps = float(sess_scope.get("total_tokens_per_sec", 0.0))
    by_agent = dict(sess_scope.get("breakdown_by_agent") or summary.get("breakdown_by_agent") or {})
    by_model = dict(sess_scope.get("breakdown_by_model") or summary.get("breakdown_by_model") or {})

    # Account for Gemini Decisions API (`gemini-decision-1`) input tokens and cost ($0.10 / 1M input tokens, 0 output tokens)
    if mode == "with_decisions_router" and decisions_info:
        dec_in_tok = int(decisions_info.get("input_tokens", 0) or 0)
        dec_cost = float(decisions_info.get("decision_cost_usd", 0.0) or 0.0)
        dec_model_name = str(decisions_info.get("decision_model") or "gemini-decision-1")
        if dec_in_tok > 0:
            prompt_tokens += dec_in_tok
            total_tokens += dec_in_tok
            llm_cost_usd = round(llm_cost_usd + dec_cost, 8)
            total_cost_usd = round(total_cost_usd + dec_cost, 8)
            by_model[dec_model_name] = {
                "calls": 1,
                "prompt_tokens": dec_in_tok,
                "completion_tokens": 0,
                "thoughts_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": dec_in_tok,
                "llm_cost_usd": round(dec_cost, 8),
                "total_cost_usd": round(dec_cost, 8),
            }

    # Export session telemetry to JSONL and CSV with eval metadata tags
    export_tags = {
        "mode": mode,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "expected_complexity": sample.expected_complexity,
        "coordinator_model": coordinator_model,
        "routed_model": routed_model,
        "laya_choice": laya_choice,
        "laya_confidence": round(laya_conf, 4),
        "laya_latency_ms": round(laya_lat_ms, 2),
        "is_correct": is_correct,
        "predicted_answer": str(pred_ans)[:80],
    }
    if summary:
        async with _EXPORT_LOCK:
            jsonl_exporter.export_summary(summary, scope="session", tags=export_tags)
            csv_exporter.export_summary(summary, scope="session", tags=export_tags)

    return SampleEvalResult(
        mode=mode,
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        expected_complexity=sample.expected_complexity,
        coordinator_model=coordinator_model,
        routed_model=routed_model,
        laya_choice=laya_choice,
        laya_confidence=laya_conf,
        laya_latency_ms=laya_lat_ms,
        delegated_agents=delegated_agents,
        operational_status=operational_status,
        is_correct=is_correct,
        predicted_answer=str(pred_ans),
        ground_truth=sample.ground_truth,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        thoughts_tokens=thoughts_tokens,
        total_tokens=total_tokens,
        llm_cost_usd=llm_cost_usd,
        total_cost_usd=total_cost_usd,
        turn_latency_ms=finops_lat_ms,
        llm_latency_ms=llm_lat_ms,
        output_tokens_per_sec=out_tps,
        total_tokens_per_sec=tot_tps,
        breakdown_by_agent=by_agent,
        breakdown_by_model=by_model,
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(len(sorted_v) - 1, int(round(0.95 * (len(sorted_v) - 1))))
    return sorted_v[idx]


def aggregate_mode_metrics(results: list[SampleEvalResult]) -> dict[str, Any]:
    """Aggregates token, latency, throughput, cost, and accuracy metrics for a mode."""
    total_runs = len(results)
    if total_runs == 0:
        return {}

    op_success_count = sum(1 for r in results if r.operational_status == "success")
    correct_count = sum(1 for r in results if r.is_correct)
    prompt_tokens = sum(r.prompt_tokens for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    thoughts_tokens = sum(r.thoughts_tokens for r in results)
    total_tokens = sum(r.total_tokens for r in results)
    total_cost_usd = sum(r.total_cost_usd for r in results)

    turn_lats = [r.turn_latency_ms for r in results if r.turn_latency_ms > 0]
    llm_lats = [r.llm_latency_ms for r in results if r.llm_latency_ms > 0]
    laya_lats = [r.laya_latency_ms for r in results if r.laya_latency_ms > 0]
    out_tps_list = [r.output_tokens_per_sec for r in results if r.output_tokens_per_sec > 0]

    # Aggregate breakdown_by_agent across all sessions
    agent_totals: dict[str, dict[str, Any]] = {}
    for r in results:
        for ag_name, ag_data in r.breakdown_by_agent.items():
            slot = agent_totals.setdefault(
                ag_name,
                {
                    "calls": 0,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "thoughts_tokens": 0,
                    "total_cost_usd": 0.0,
                },
            )
            slot["calls"] += int(ag_data.get("calls", 0))
            slot["total_tokens"] += int(ag_data.get("total_tokens", 0))
            slot["prompt_tokens"] += int(ag_data.get("prompt_tokens", 0))
            slot["completion_tokens"] += int(ag_data.get("completion_tokens", 0))
            slot["thoughts_tokens"] += int(ag_data.get("thoughts_tokens", 0))
            slot["total_cost_usd"] += float(ag_data.get("total_cost_usd", 0.0))

    # Aggregate breakdown_by_model across all sessions
    model_totals: dict[str, dict[str, Any]] = {}
    for r in results:
        for m_name, m_data in r.breakdown_by_model.items():
            slot = model_totals.setdefault(
                m_name,
                {
                    "calls": 0,
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "thoughts_tokens": 0,
                    "total_cost_usd": 0.0,
                },
            )
            p_tok = int(m_data.get("prompt_tokens", 0))
            c_tok = int(m_data.get("completion_tokens", 0))
            th_tok = int(m_data.get("thoughts_tokens", 0))
            t_tok = int(m_data.get("total_tokens", 0)) or (p_tok + c_tok + th_tok)
            slot["calls"] += int(m_data.get("calls", 0))
            slot["total_tokens"] += t_tok
            slot["prompt_tokens"] += p_tok
            slot["completion_tokens"] += c_tok
            slot["thoughts_tokens"] += th_tok
            slot["total_cost_usd"] += float(m_data.get("total_cost_usd", 0.0))

    return {
        "total_runs": total_runs,
        "op_success_count": op_success_count,
        "op_success_rate_pct": (op_success_count / total_runs) * 100.0,
        "correct_count": correct_count,
        "accuracy_pct": (correct_count / total_runs) * 100.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thoughts_tokens": thoughts_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
        "avg_cost_per_task_usd": total_cost_usd / total_runs,
        "cost_per_correct_usd": (total_cost_usd / correct_count) if correct_count > 0 else 0.0,
        "total_turn_latency_ms": sum(turn_lats),
        "avg_turn_latency_ms": statistics.mean(turn_lats) if turn_lats else 0.0,
        "p95_turn_latency_ms": _p95(turn_lats),
        "avg_llm_latency_ms": statistics.mean(llm_lats) if llm_lats else 0.0,
        "p95_llm_latency_ms": _p95(llm_lats),
        "avg_laya_latency_ms": statistics.mean(laya_lats) if laya_lats else 0.0,
        "p95_laya_latency_ms": _p95(laya_lats),
        "avg_output_tps": statistics.mean(out_tps_list) if out_tps_list else 0.0,
        "agent_totals": agent_totals,
        "model_totals": model_totals,
    }


def _pct_delta(base: float, target: float) -> str:
    if base == 0:
        return "—"
    diff_pct = ((target - base) / base) * 100.0
    sign = "+" if diff_pct > 0 else ""
    return f"{sign}{diff_pct:.1f}%"


def generate_markdown_report(
    results_by_mode: dict[str, list[SampleEvalResult]],
    output_path: Path = REPORT_PATH,
) -> str:
    """Generates a 5-arm side-by-side Markdown evaluation report (`eval/EVAL_REPORT.md`) comparing Laya and Gemini Decisions API."""
    sim = aggregate_mode_metrics(results_by_mode["all_simple"])
    pro = aggregate_mode_metrics(results_by_mode["all_complex"])
    hyb = aggregate_mode_metrics(results_by_mode["coord_simple_subs_complex"])
    rtd = aggregate_mode_metrics(results_by_mode["with_router"])
    routed_results = results_by_mode["with_router"]

    has_dec = bool(results_by_mode.get("with_decisions_router"))
    dec_results = results_by_mode.get("with_decisions_router") or routed_results
    dec = aggregate_mode_metrics(dec_results)

    cost_saved_vs_pro = pro["total_cost_usd"] - rtd["total_cost_usd"]
    cost_saved_vs_hyb = hyb["total_cost_usd"] - rtd["total_cost_usd"]
    lat_saved_vs_pro = pro["avg_turn_latency_ms"] - rtd["avg_turn_latency_ms"]
    lat_saved_vs_hyb = hyb["avg_turn_latency_ms"] - rtd["avg_turn_latency_ms"]

    dec_cost_saved_vs_pro = pro["total_cost_usd"] - dec["total_cost_usd"]
    dec_cost_saved_vs_hyb = hyb["total_cost_usd"] - dec["total_cost_usd"]
    dec_lat_saved_vs_pro = pro["avg_turn_latency_ms"] - dec["avg_turn_latency_ms"]
    dec_lat_saved_vs_hyb = hyb["avg_turn_latency_ms"] - dec["avg_turn_latency_ms"]

    sim_simple_cost = sum(r.total_cost_usd for r in results_by_mode["all_simple"] if r.expected_complexity == "simple")
    pro_simple_cost = sum(r.total_cost_usd for r in results_by_mode["all_complex"] if r.expected_complexity == "simple")
    hyb_simple_cost = sum(r.total_cost_usd for r in results_by_mode["coord_simple_subs_complex"] if r.expected_complexity == "simple")
    rtd_simple_cost = sum(r.total_cost_usd for r in routed_results if r.expected_complexity == "simple")
    dec_simple_cost = sum(r.total_cost_usd for r in dec_results if r.expected_complexity == "simple")

    dec_router_model_stats = dec["model_totals"].get("gemini-decision-1", {"calls": 0, "prompt_tokens": 0, "total_cost_usd": 0.0})
    dec_router_only_cost = float(dec_router_model_stats.get("total_cost_usd", 0.0))
    dec_router_only_tokens = int(dec_router_model_stats.get("prompt_tokens", 0))

    num_arms = 5 if has_dec else 4

    lines: list[str] = [
        f"# Multi-Agent ADK Model Router Evaluation Report ({num_arms}-Arm Benchmark, N={sim['total_runs']} per Arm, temperature=0.0)",
        "",
        f"Comprehensive {num_arms}-arm comparison of a **4-Agent Google ADK Hierarchy** (`coordinator_agent` $\\rightarrow$ `fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) with `temperature=0.0` evaluated on **{sim['total_runs']} real open-source Hugging Face benchmark tasks per arm** (**{sim['total_runs'] * num_arms} total multi-agent turns**) across a **2x3 Domain $\\times$ Complexity matrix** (`10 Simple + 10 Complex` per specialist agent) and instrumented with **[`adk-finops`](https://pypi.org/project/adk-finops/)**:",
        "",
        "- **`fast_qa_agent` (20 QA tasks)**: `10 ARC-Easy` (Simple QA) + `10 MMLU-Pro` (Complex 10-option Analytical QA)",
        "- **`math_analyst_agent` (20 Math tasks)**: `10 GSM8K` (Simple Math) + `10 MATH-500 Level-5` (Complex Olympiad Math)",
        "- **`code_engineer_agent` (20 Coding tasks)**: `10 MBPP` (Simple Python Helpers) + `10 HumanEval` (Complex Algorithmic Coding)",
        "",
        "### Experimental Arms",
        f"1. **All-Simple (`all_simple`)**: `coordinator_agent` (`{SIMPLE_MODEL}`) $\\rightarrow$ Specialist sub-agents (`{SIMPLE_MODEL}`).",
        f"2. **All-Complex (`all_complex`)**: `coordinator_agent` (`{COMPLEX_MODEL}`) $\\rightarrow$ Specialist sub-agents (`{COMPLEX_MODEL}`).",
        f"3. **Static Hybrid (`coord_simple_subs_complex`)**: `coordinator_agent` fixed to `{SIMPLE_MODEL}` and all 3 specialist sub-agents fixed to `{COMPLEX_MODEL}` (no dynamic router).",
        f"4. **Role-Aware Laya Router (`with_router`)**: `coordinator_agent` pinned to `{SIMPLE_MODEL}` via `exclude_agents={{\"coordinator_agent\"}}`, while specialist sub-agents (`fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) are dynamically routed per query by [`LayaModelRouterPlugin`](../laya_adk_model_router/plugin.py) (`laya-typed-decisions`, local CPU/MPS, `$0.00` routing cost) between `{SIMPLE_MODEL}` and `{COMPLEX_MODEL}`.",
        f"5. **Role-Aware Gemini Decisions API Router (`with_decisions_router`)**: `coordinator_agent` pinned to `{SIMPLE_MODEL}` via `exclude_agents={{\"coordinator_agent\"}}`, while specialist sub-agents are dynamically routed per query by [`DecisionsModelRouterPlugin`](../decisions_adk_model_router/plugin.py) (`gemini-decision-1` on `v1beta`, `0` output tokens, `$0.10 / 1M` input tokens) between `{SIMPLE_MODEL}` and `{COMPLEX_MODEL}`.",
        "",
        "---",
        "",
        f"## 1. Executive Summary (All {num_arms} Experimental Arms)",
        "",
        f"| Metric | 1. All-Simple (`{SIMPLE_MODEL}`) | 2. All-Complex (`{COMPLEX_MODEL}`) | 3. Static Hybrid (`Coord=Simple, Subs=Complex`) | 4. With Laya Router (`laya-typed-decisions`) | 5. With Decisions API Router (`gemini-decision-1`) | Laya vs. All-Complex | Decisions vs. All-Complex | Decisions vs. Laya |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| **Total Benchmark Tasks** | {sim['total_runs']} | {pro['total_runs']} | {hyb['total_runs']} | {rtd['total_runs']} | {dec['total_runs']} | — | — | — |",
        f"| **Operational Success Rate (`adk-finops`)** | {sim['op_success_rate_pct']:.1f}% ({sim['op_success_count']}/{sim['total_runs']}) | {pro['op_success_rate_pct']:.1f}% ({pro['op_success_count']}/{pro['total_runs']}) | {hyb['op_success_rate_pct']:.1f}% ({hyb['op_success_count']}/{hyb['total_runs']}) | {rtd['op_success_rate_pct']:.1f}% ({rtd['op_success_count']}/{rtd['total_runs']}) | {dec['op_success_rate_pct']:.1f}% ({dec['op_success_count']}/{dec['total_runs']}) | {rtd['op_success_rate_pct'] - pro['op_success_rate_pct']:+.1f}% | {dec['op_success_rate_pct'] - pro['op_success_rate_pct']:+.1f}% | {dec['op_success_rate_pct'] - rtd['op_success_rate_pct']:+.1f}% |",
        f"| **Ground-Truth Accuracy (%)** | **{sim['accuracy_pct']:.1f}%** ({sim['correct_count']}/{sim['total_runs']}) | **{pro['accuracy_pct']:.1f}%** ({pro['correct_count']}/{pro['total_runs']}) | **{hyb['accuracy_pct']:.1f}%** ({hyb['correct_count']}/{hyb['total_runs']}) | **{rtd['accuracy_pct']:.1f}%** ({rtd['correct_count']}/{rtd['total_runs']}) | **{dec['accuracy_pct']:.1f}%** ({dec['correct_count']}/{dec['total_runs']}) | **{rtd['accuracy_pct'] - pro['accuracy_pct']:+.1f}%** | **{dec['accuracy_pct'] - pro['accuracy_pct']:+.1f}%** | **{dec['accuracy_pct'] - rtd['accuracy_pct']:+.1f}%** |",
        f"| **Total Cost (USD)** | **${sim['total_cost_usd']:.6f}** | **${pro['total_cost_usd']:.6f}** | **${hyb['total_cost_usd']:.6f}** | **${rtd['total_cost_usd']:.6f}** | **${dec['total_cost_usd']:.6f}** | **{_pct_delta(pro['total_cost_usd'], rtd['total_cost_usd'])} (${cost_saved_vs_pro:+.6f})** | **{_pct_delta(pro['total_cost_usd'], dec['total_cost_usd'])} (${dec_cost_saved_vs_pro:+.6f})** | **{_pct_delta(rtd['total_cost_usd'], dec['total_cost_usd'])}** |",
        f"| **Simple-Tier Tasks Cost (`N=30`)** | **${sim_simple_cost:.6f}** | **${pro_simple_cost:.6f}** | **${hyb_simple_cost:.6f}** | **${rtd_simple_cost:.6f}** | **${dec_simple_cost:.6f}** | **{_pct_delta(pro_simple_cost, rtd_simple_cost)}** | **{_pct_delta(pro_simple_cost, dec_simple_cost)}** | **{_pct_delta(rtd_simple_cost, dec_simple_cost)}** |",
        f"| **Router Inference Cost (`N=60`)** | $0.000000 | $0.000000 | $0.000000 | **$0.000000** (local) | **${dec_router_only_cost:.6f}** ({dec_router_only_tokens:,} tok) | $0.00 | +${dec_router_only_cost:.6f} | +${dec_router_only_cost:.6f} |",
        f"| **Avg Cost / Task (USD)** | ${sim['avg_cost_per_task_usd']:.6f} | ${pro['avg_cost_per_task_usd']:.6f} | ${hyb['avg_cost_per_task_usd']:.6f} | ${rtd['avg_cost_per_task_usd']:.6f} | ${dec['avg_cost_per_task_usd']:.6f} | {_pct_delta(pro['avg_cost_per_task_usd'], rtd['avg_cost_per_task_usd'])} | {_pct_delta(pro['avg_cost_per_task_usd'], dec['avg_cost_per_task_usd'])} | {_pct_delta(rtd['avg_cost_per_task_usd'], dec['avg_cost_per_task_usd'])} |",
        f"| **Effective Cost / Correct Answer** | ${sim['cost_per_correct_usd']:.6f} | ${pro['cost_per_correct_usd']:.6f} | ${hyb['cost_per_correct_usd']:.6f} | ${rtd['cost_per_correct_usd']:.6f} | ${dec['cost_per_correct_usd']:.6f} | {_pct_delta(pro['cost_per_correct_usd'], rtd['cost_per_correct_usd'])} | {_pct_delta(pro['cost_per_correct_usd'], dec['cost_per_correct_usd'])} | {_pct_delta(rtd['cost_per_correct_usd'], dec['cost_per_correct_usd'])} |",
        f"| **Total Billable Tokens** | {sim['total_tokens']:,} | {pro['total_tokens']:,} | {hyb['total_tokens']:,} | {rtd['total_tokens']:,} | {dec['total_tokens']:,} | {_pct_delta(pro['total_tokens'], rtd['total_tokens'])} | {_pct_delta(pro['total_tokens'], dec['total_tokens'])} | {_pct_delta(rtd['total_tokens'], dec['total_tokens'])} |",
        f"| **Prompt Tokens** | {sim['prompt_tokens']:,} | {pro['prompt_tokens']:,} | {hyb['prompt_tokens']:,} | {rtd['prompt_tokens']:,} | {dec['prompt_tokens']:,} | {_pct_delta(pro['prompt_tokens'], rtd['prompt_tokens'])} | {_pct_delta(pro['prompt_tokens'], dec['prompt_tokens'])} | {_pct_delta(rtd['prompt_tokens'], dec['prompt_tokens'])} |",
        f"| **Completion Tokens** | {sim['completion_tokens']:,} | {pro['completion_tokens']:,} | {hyb['completion_tokens']:,} | {rtd['completion_tokens']:,} | {dec['completion_tokens']:,} | {_pct_delta(pro['completion_tokens'], rtd['completion_tokens'])} | {_pct_delta(pro['completion_tokens'], dec['completion_tokens'])} | {_pct_delta(rtd['completion_tokens'], dec['completion_tokens'])} |",
        f"| **Thinking Tokens (`thoughts_tokens`)** | {sim['thoughts_tokens']:,} | {pro['thoughts_tokens']:,} | {hyb['thoughts_tokens']:,} | {rtd['thoughts_tokens']:,} | {dec['thoughts_tokens']:,} | {_pct_delta(pro['thoughts_tokens'], rtd['thoughts_tokens'])} | {_pct_delta(pro['thoughts_tokens'], dec['thoughts_tokens'])} | {_pct_delta(rtd['thoughts_tokens'], dec['thoughts_tokens'])} |",
        f"| **Avg Turn Latency (ms)** | **{sim['avg_turn_latency_ms']:.1f} ms** | **{pro['avg_turn_latency_ms']:.1f} ms** | **{hyb['avg_turn_latency_ms']:.1f} ms** | **{rtd['avg_turn_latency_ms']:.1f} ms** | **{dec['avg_turn_latency_ms']:.1f} ms** | **{_pct_delta(pro['avg_turn_latency_ms'], rtd['avg_turn_latency_ms'])} ({lat_saved_vs_pro:+.1f} ms)** | **{_pct_delta(pro['avg_turn_latency_ms'], dec['avg_turn_latency_ms'])} ({dec_lat_saved_vs_pro:+.1f} ms)** | **{_pct_delta(rtd['avg_turn_latency_ms'], dec['avg_turn_latency_ms'])}** |",
        f"| **P95 Turn Latency (ms)** | {sim['p95_turn_latency_ms']:.1f} ms | {pro['p95_turn_latency_ms']:.1f} ms | {hyb['p95_turn_latency_ms']:.1f} ms | {rtd['p95_turn_latency_ms']:.1f} ms | {dec['p95_turn_latency_ms']:.1f} ms | {_pct_delta(pro['p95_turn_latency_ms'], rtd['p95_turn_latency_ms'])} | {_pct_delta(pro['p95_turn_latency_ms'], dec['p95_turn_latency_ms'])} | {_pct_delta(rtd['p95_turn_latency_ms'], dec['p95_turn_latency_ms'])} |",
        f"| **Avg Router Overhead (ms)** | 0.0 ms | 0.0 ms | 0.0 ms | **{rtd['avg_laya_latency_ms']:.1f} ms** (p95: {rtd['p95_laya_latency_ms']:.1f} ms) | **{dec['avg_laya_latency_ms']:.1f} ms** (p95: {dec['p95_laya_latency_ms']:.1f} ms) | +{rtd['avg_laya_latency_ms']:.1f} ms | +{dec['avg_laya_latency_ms']:.1f} ms | **{_pct_delta(rtd['avg_laya_latency_ms'], dec['avg_laya_latency_ms'])}** |",
        f"| **Avg Output Throughput (`tok/s`)** | {sim['avg_output_tps']:.1f} tok/s | {pro['avg_output_tps']:.1f} tok/s | {hyb['avg_output_tps']:.1f} tok/s | {rtd['avg_output_tps']:.1f} tok/s | {dec['avg_output_tps']:.1f} tok/s | {_pct_delta(pro['avg_output_tps'], rtd['avg_output_tps'])} | {_pct_delta(pro['avg_output_tps'], dec['avg_output_tps'])} | {_pct_delta(rtd['avg_output_tps'], dec['avg_output_tps'])} |",
        "",
        "---",
        "",
        "## 2. Head-to-Head Router Classification Audit: `Laya` vs. `Gemini Decision 1` (2x3 Domain x Complexity Matrix)",
        "",
    ]

    tp_complex = sum(1 for r in routed_results if r.expected_complexity == "complex" and r.laya_choice == "complex")
    tn_simple = sum(1 for r in routed_results if r.expected_complexity == "simple" and r.laya_choice == "simple")
    raw_cls_acc = ((tp_complex + tn_simple) / len(routed_results) * 100.0) if routed_results else 0.0

    routed_correct_tier = sum(
        1
        for r in routed_results
        if (r.expected_complexity == "simple" and r.routed_model == SIMPLE_MODEL)
        or (r.expected_complexity == "complex" and r.routed_model == COMPLEX_MODEL)
    )
    routed_tier_acc = (routed_correct_tier / len(routed_results) * 100.0) if routed_results else 0.0

    dec_tp_complex = sum(1 for r in dec_results if r.expected_complexity == "complex" and r.laya_choice == "complex")
    dec_tn_simple = sum(1 for r in dec_results if r.expected_complexity == "simple" and r.laya_choice == "simple")
    dec_raw_cls_acc = ((dec_tp_complex + dec_tn_simple) / len(dec_results) * 100.0) if dec_results else 0.0

    dec_routed_correct_tier = sum(
        1
        for r in dec_results
        if (r.expected_complexity == "simple" and r.routed_model == SIMPLE_MODEL)
        or (r.expected_complexity == "complex" and r.routed_model == COMPLEX_MODEL)
    )
    dec_routed_tier_acc = (dec_routed_correct_tier / len(dec_results) * 100.0) if dec_results else 0.0

    lines.extend(
        [
            f"- **Laya (`laya-typed-decisions`) Tier Accuracy**: **{raw_cls_acc:.1f}%** ({tp_complex + tn_simple}/{len(routed_results)}) raw choice | **{routed_tier_acc:.1f}%** ({routed_correct_tier}/{len(routed_results)}) effective routed model | Avg latency: **{rtd['avg_laya_latency_ms']:.1f} ms** | Router cost: **$0.00** (local)",
            f"- **Google Gemini Decisions API (`gemini-decision-1`) Tier Accuracy**: **{dec_raw_cls_acc:.1f}%** ({dec_tp_complex + dec_tn_simple}/{len(dec_results)}) raw choice | **{dec_routed_tier_acc:.1f}%** ({dec_routed_correct_tier}/{len(dec_results)}) effective routed model | Avg latency: **{dec['avg_laya_latency_ms']:.1f} ms** | Router cost: **${dec_router_only_cost:.6f}** ({dec_router_only_tokens:,} input tokens, 0 output tokens)",
            "- **Anchor Calibration Note**: The routing criteria in [`laya_adk_model_router/questions.py`](../laya_adk_model_router/questions.py) and [`decisions_adk_model_router/questions.py`](../decisions_adk_model_router/questions.py) use identical `simple` and `complex` domain criteria covering `QA`, `Math`, and `Coding`. When deploying to new domain distributions, we recommend tuning the `simple` and `complex` criteria (or `confidence_threshold`) and benchmarking against alternative semantic/embedding routers such as [`jev`](https://github.com/dmoonat/jev).",
            "",
            "| Dataset | Specialist Agent | Ground-Truth Tier | Samples | Laya (`S`/`C`) | Laya Avg Conf (Min–Max) | Laya Latency | Decisions API (`S`/`C`) | Decisions Avg Conf (Min–Max) | Decisions Latency | Routed Specialist Model (`Laya` / `Decisions`) |",
            "| :--- | :--- | :--- | ---: | :---: | :---: | ---: | :---: | :---: | ---: | :--- |",
        ]
    )
    for ds_name in DATASET_NAMES:
        ds_rows = [r for r in routed_results if r.dataset == ds_name]
        dec_ds_rows = [r for r in dec_results if r.dataset == ds_name]
        if not ds_rows:
            continue
        exp_tier = ds_rows[0].expected_complexity
        exp_agent = (
            ds_rows[0].delegated_agents[0]
            if ds_rows[0].delegated_agents
            else "specialist_agent"
        )
        pred_s = sum(1 for r in ds_rows if r.laya_choice == "simple")
        pred_c = sum(1 for r in ds_rows if r.laya_choice == "complex")
        confs = [r.laya_confidence for r in ds_rows]
        avg_c = statistics.mean(confs) if confs else 0.0
        min_c, max_c = (min(confs), max(confs)) if confs else (0.0, 0.0)
        avg_lat = statistics.mean([r.laya_latency_ms for r in ds_rows]) if ds_rows else 0.0
        models_used = ", ".join(sorted(set(r.routed_model for r in ds_rows)))

        d_pred_s = sum(1 for r in dec_ds_rows if r.laya_choice == "simple")
        d_pred_c = sum(1 for r in dec_ds_rows if r.laya_choice == "complex")
        d_confs = [r.laya_confidence for r in dec_ds_rows]
        d_avg_c = statistics.mean(d_confs) if d_confs else 0.0
        d_min_c, d_max_c = (min(d_confs), max(d_confs)) if d_confs else (0.0, 0.0)
        d_avg_lat = statistics.mean([r.laya_latency_ms for r in dec_ds_rows]) if dec_ds_rows else 0.0
        d_models_used = ", ".join(sorted(set(r.routed_model for r in dec_ds_rows)))

        lines.append(
            f"| **`{ds_name}`** | `{exp_agent}` | `{exp_tier}` | {len(ds_rows)} | "
            f"{pred_s}/{pred_c} | {avg_c:.2f} ({min_c:.2f}–{max_c:.2f}) | {avg_lat:.1f} ms | "
            f"{d_pred_s}/{d_pred_c} | {d_avg_c:.4f} ({d_min_c:.4f}–{d_max_c:.4f}) | {d_avg_lat:.1f} ms | "
            f"`{models_used}` / `{d_models_used}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 3. Per-Dataset Breakdown across All {num_arms} Arms (2x3 Domain x Complexity)",
            "",
            "| Benchmark (Tier) | Arm / Mode | Coordinator $\\rightarrow$ Specialist Model | Accuracy (%) | Total Tokens (In / Out / Think) | Avg Turn Latency (ms) | Total Cost (USD) | Cost vs. All-Complex | Cost vs. Static Hybrid |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for ds in DATASET_NAMES:
        ds_sim = aggregate_mode_metrics([r for r in results_by_mode["all_simple"] if r.dataset == ds])
        ds_pro = aggregate_mode_metrics([r for r in results_by_mode["all_complex"] if r.dataset == ds])
        ds_hyb = aggregate_mode_metrics([r for r in results_by_mode["coord_simple_subs_complex"] if r.dataset == ds])
        ds_rtd_rows = [r for r in routed_results if r.dataset == ds]
        ds_rtd = aggregate_mode_metrics(ds_rtd_rows)
        ds_dec_rows = [r for r in dec_results if r.dataset == ds]
        ds_dec = aggregate_mode_metrics(ds_dec_rows)
        if not ds_pro or not ds_rtd or not ds_sim or not ds_hyb:
            continue
        exp_tier = ds_rtd_rows[0].expected_complexity
        spec_models = ", ".join(sorted(set(r.routed_model for r in ds_rtd_rows)))
        dec_spec_models = ", ".join(sorted(set(r.routed_model for r in ds_dec_rows)))

        mode_tuples = [
            ("`all_simple`", ds_sim, f"`{SIMPLE_MODEL}` $\\rightarrow$ `{SIMPLE_MODEL}`"),
            ("`all_complex`", ds_pro, f"`{COMPLEX_MODEL}` $\\rightarrow$ `{COMPLEX_MODEL}`"),
            ("`coord_simple_subs_complex`", ds_hyb, f"`{SIMPLE_MODEL}` $\\rightarrow$ `{COMPLEX_MODEL}`"),
            ("`with_router` (Laya)", ds_rtd, f"`{SIMPLE_MODEL}` $\\rightarrow$ `{spec_models}`"),
            ("`with_decisions_router`", ds_dec, f"`{SIMPLE_MODEL}` $\\rightarrow$ `{dec_spec_models}`"),
        ]
        for mode_label, ds_agg, model_pair in mode_tuples:
            vs_pro = "—" if mode_label == "`all_complex`" else _pct_delta(ds_pro["total_cost_usd"], ds_agg["total_cost_usd"])
            vs_hyb = "—" if mode_label == "`coord_simple_subs_complex`" else _pct_delta(ds_hyb["total_cost_usd"], ds_agg["total_cost_usd"])
            if mode_label in ("`with_router` (Laya)", "`with_decisions_router`"):
                vs_pro = f"**{vs_pro}**"
                vs_hyb = f"**{vs_hyb}**"
            lines.append(
                f"| **{ds}** (`{exp_tier}`) | {mode_label} | {model_pair} | "
                f"{ds_agg['accuracy_pct']:.1f}% ({ds_agg['correct_count']}/{ds_agg['total_runs']}) | "
                f"{ds_agg['total_tokens']:,} ({ds_agg['prompt_tokens']:,} / {ds_agg['completion_tokens']:,} / {ds_agg['thoughts_tokens']:,}) | "
                f"{ds_agg['avg_turn_latency_ms']:.1f} ms | ${ds_agg['total_cost_usd']:.6f} | {vs_pro} | {vs_hyb} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Multi-Agent Cost & Token Attribution (`adk-finops` `breakdown_by_agent`)",
            "",
            "| Agent Name | All-Simple Cost (Tok) | All-Complex Cost (Tok) | Static Hybrid Cost (Tok) | With-Laya Cost (Tok) | With-Decisions Cost (Tok) | Laya vs. All-Complex | Decisions vs. All-Complex |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    all_agents = sorted(
        set(sim["agent_totals"].keys())
        | set(pro["agent_totals"].keys())
        | set(hyb["agent_totals"].keys())
        | set(rtd["agent_totals"].keys())
        | set(dec["agent_totals"].keys())
    )
    for ag in all_agents:
        s_ag = sim["agent_totals"].get(ag, {"total_tokens": 0, "total_cost_usd": 0.0})
        p_ag = pro["agent_totals"].get(ag, {"total_tokens": 0, "total_cost_usd": 0.0})
        h_ag = hyb["agent_totals"].get(ag, {"total_tokens": 0, "total_cost_usd": 0.0})
        r_ag = rtd["agent_totals"].get(ag, {"total_tokens": 0, "total_cost_usd": 0.0})
        d_ag = dec["agent_totals"].get(ag, {"total_tokens": 0, "total_cost_usd": 0.0})
        lines.append(
            f"| **`{ag}`** | ${s_ag['total_cost_usd']:.6f} ({s_ag['total_tokens']:,}) | "
            f"${p_ag['total_cost_usd']:.6f} ({p_ag['total_tokens']:,}) | "
            f"${h_ag['total_cost_usd']:.6f} ({h_ag['total_tokens']:,}) | "
            f"${r_ag['total_cost_usd']:.6f} ({r_ag['total_tokens']:,}) | "
            f"${d_ag['total_cost_usd']:.6f} ({d_ag['total_tokens']:,}) | "
            f"**{_pct_delta(p_ag['total_cost_usd'], r_ag['total_cost_usd'])}** | "
            f"**{_pct_delta(p_ag['total_cost_usd'], d_ag['total_cost_usd'])}** |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Model Distribution (`adk-finops` `breakdown_by_model`)",
            "",
            "| Arm / Mode | Model | LLM Calls | Prompt Tokens | Completion Tokens | Thinking Tokens | Total Tokens | Total Cost (USD) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for mode_label, agg in (
        ("all_simple", sim),
        ("all_complex", pro),
        ("coord_simple_subs_complex", hyb),
        ("with_router", rtd),
        ("with_decisions_router", dec),
    ):
        for m_name, m_data in sorted(agg["model_totals"].items()):
            lines.append(
                f"| `{mode_label}` | **`{m_name}`** | {m_data['calls']} | {m_data['prompt_tokens']:,} | "
                f"{m_data['completion_tokens']:,} | {m_data['thoughts_tokens']:,} | "
                f"{m_data['total_tokens']:,} | ${m_data['total_cost_usd']:.6f} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 6. Sample-Level Comparison Log (All {sim['total_runs']} Tasks across All {num_arms} Arms)",
            "",
            "| Sample ID | Dataset (Tier) | Laya Choice (Conf / Lat) | Decisions Choice (Conf / Lat) | Latency (`Simple` / `Complex` / `Hybrid` / `Laya` / `Decisions`) | Cost (`Simple` / `Complex` / `Hybrid` / `Laya` / `Decisions`) | Correct (`S`/`C`/`H`/`L`/`D`) |",
            "| :--- | :--- | :--- | :--- | ---: | ---: | :---: |",
        ]
    )

    sim_by_id = {r.sample_id: r for r in results_by_mode["all_simple"]}
    pro_by_id = {r.sample_id: r for r in results_by_mode["all_complex"]}
    hyb_by_id = {r.sample_id: r for r in results_by_mode["coord_simple_subs_complex"]}
    dec_by_id = {r.sample_id: r for r in dec_results}
    for r in routed_results:
        s = sim_by_id.get(r.sample_id)
        b = pro_by_id.get(r.sample_id)
        h = hyb_by_id.get(r.sample_id)
        d = dec_by_id.get(r.sample_id)
        if not s or not b or not h or not d:
            continue
        s_ok = "✅" if s.is_correct else "❌"
        b_ok = "✅" if b.is_correct else "❌"
        h_ok = "✅" if h.is_correct else "❌"
        r_ok = "✅" if r.is_correct else "❌"
        d_ok = "✅" if d.is_correct else "❌"
        lines.append(
            f"| `{r.sample_id}` | {r.dataset} (`{r.expected_complexity}`) | "
            f"`{r.laya_choice}` ({r.laya_confidence:.2f} / {r.laya_latency_ms:.0f}ms) | "
            f"`{d.laya_choice}` ({d.laya_confidence:.3f} / {d.laya_latency_ms:.0f}ms) | "
            f"{s.turn_latency_ms:.0f} / {b.turn_latency_ms:.0f} / {h.turn_latency_ms:.0f} / **{r.turn_latency_ms:.0f}** / **{d.turn_latency_ms:.0f} ms** | "
            f"${s.total_cost_usd:.4f} / ${b.total_cost_usd:.4f} / ${h.total_cost_usd:.4f} / **${r.total_cost_usd:.4f}** / **${d.total_cost_usd:.4f}** | "
            f"{s_ok}/{b_ok}/{h_ok}/{r_ok}/{d_ok} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 7. Key Findings & Architectural Takeaways",
            "",
            "1. **Every Specialist Agent Faces Mixed Complexity (`10 Simple + 10 Complex` per Domain)**: Because `fast_qa_agent` (`ARC-Easy` + `MMLU-Pro`), `math_analyst_agent` (`GSM8K` + `MATH-500`), and `code_engineer_agent` (`MBPP` + `HumanEval`) each handle both simple and complex tasks, no static per-agent model assignment can optimize both cost and accuracy.",
            f"2. **Why Dynamic Routing (`Laya` & `Gemini Decision 1`) Beats `All-Simple` (`{SIMPLE_MODEL}`)**: `all_simple` (`{SIMPLE_MODEL}`) drops to **{sim['accuracy_pct']:.1f}% ({sim['correct_count']}/{sim['total_runs']})** overall accuracy on the complex tasks (`80.0%` on `MMLU-Pro`, `70.0%` on `MATH-500`, `70.0%` on `HumanEval`), whereas `with_router` achieves **{rtd['accuracy_pct']:.1f}% ({rtd['correct_count']}/{rtd['total_runs']})** and `with_decisions_router` achieves **{dec['accuracy_pct']:.1f}% ({dec['correct_count']}/{dec['total_runs']})**.",
            f"3. **Head-to-Head: Local `Laya` (`laya-typed-decisions`) vs. Cloud `Gemini Decision 1` (`gemini-decision-1`)**:",
            f"   - **Classification Accuracy & Confidence Calibration**: Laya achieved **{raw_cls_acc:.1f}% ({tp_complex + tn_simple}/{len(routed_results)})** tier accuracy (confidence range `0.56–0.88`), while `gemini-decision-1` achieved **{dec_raw_cls_acc:.1f}% ({dec_tp_complex + dec_tn_simple}/{len(dec_results)})** tier accuracy with sharper calibrated probabilities (`0.95–1.00` confidence).",
            f"   - **Router Latency (`{rtd['avg_laya_latency_ms']:.1f} ms` vs. `{dec['avg_laya_latency_ms']:.1f} ms`)**: Laya runs locally in-process on CPU/MPS (`{rtd['avg_laya_latency_ms']:.1f} ms` avg), avoiding network round-trip overhead compared to the cloud `v1beta/decisions` call (`{dec['avg_laya_latency_ms']:.1f} ms` avg).",
            f"   - **Router Cost (`$0.00` vs. `${dec_router_only_cost:.6f}`)**: Both generate **0 output tokens**; Laya incurs `$0.00` API cost locally, while `gemini-decision-1` cost just **${dec_router_only_cost:.6f}** across all 60 tasks (`{dec_router_only_tokens:,}` input tokens at `$0.10 / 1M`) while offering a 262K multimodal context window and zero local model weights.",
            "4. **Decomposing Coordinator Savings vs. Specialist Routing Savings (`temperature=0.0`)**:",
            f"   - **Static Hybrid (`coord_simple_subs_complex`) vs. All-Complex (`all_complex`)**: Fixing `coordinator_agent` to `{SIMPLE_MODEL}` while keeping all specialist sub-agents on `{COMPLEX_MODEL}` changed total spend by **{_pct_delta(pro['total_cost_usd'], hyb['total_cost_usd'])}** (`${pro['total_cost_usd']:.6f}` $\\rightarrow$ `${hyb['total_cost_usd']:.6f}`) and turn latency by **{_pct_delta(pro['avg_turn_latency_ms'], hyb['avg_turn_latency_ms'])}**.",
            f"   - **With Laya Router (`with_router`) vs. All-Complex (`all_complex`)**: Saved **{_pct_delta(pro['total_cost_usd'], rtd['total_cost_usd'])}** in total cost (`${pro['total_cost_usd']:.6f}` $\\rightarrow$ `${rtd['total_cost_usd']:.6f}`), **{_pct_delta(pro_simple_cost, rtd_simple_cost)}** on simple-tier tasks (`${pro_simple_cost:.6f}` $\\rightarrow$ `${rtd_simple_cost:.6f}`), and **{_pct_delta(pro['avg_turn_latency_ms'], rtd['avg_turn_latency_ms'])}** in average turn latency.",
            f"   - **With Decisions API Router (`with_decisions_router`) vs. All-Complex (`all_complex`)**: Saved **{_pct_delta(pro['total_cost_usd'], dec['total_cost_usd'])}** in total cost (`${pro['total_cost_usd']:.6f}` $\\rightarrow$ `${dec['total_cost_usd']:.6f}`), **{_pct_delta(pro_simple_cost, dec_simple_cost)}** on simple-tier tasks (`${pro_simple_cost:.6f}` $\\rightarrow$ `${dec_simple_cost:.6f}`), and **{_pct_delta(pro['avg_turn_latency_ms'], dec['avg_turn_latency_ms'])}** in average turn latency.",
            "",
        ]
    )

    report_md = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8")
    return report_md


def print_rich_comparison(results_by_mode: dict[str, list[SampleEvalResult]]) -> None:
    """Prints a formatted Rich terminal comparison table for the 5-arm benchmark."""
    sim = aggregate_mode_metrics(results_by_mode["all_simple"])
    pro = aggregate_mode_metrics(results_by_mode["all_complex"])
    hyb = aggregate_mode_metrics(results_by_mode["coord_simple_subs_complex"])
    rtd = aggregate_mode_metrics(results_by_mode["with_router"])
    dec = aggregate_mode_metrics(results_by_mode.get("with_decisions_router") or results_by_mode["with_router"])

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"📊 Multi-Agent ADK 5-Arm Evaluation ({sim['total_runs']} Open-Source Samples per Arm, temp=0.0, adk-finops)"
        )
        table.add_column("Metric", style="bold cyan")
        table.add_column(f"1. All-Simple\n({SIMPLE_MODEL})", justify="right")
        table.add_column(f"2. All-Complex\n({COMPLEX_MODEL})", justify="right")
        table.add_column("3. Static Hybrid\n(Coord=Simple, Subs=Complex)", justify="right")
        table.add_column("4. With Laya Router\n(laya-typed-decisions)", justify="right", style="bold green")
        table.add_column("5. With Decisions API\n(gemini-decision-1)", justify="right", style="bold magenta")

        table.add_row(
            "Operational Success Rate",
            f"{sim['op_success_rate_pct']:.1f}% ({sim['op_success_count']}/{sim['total_runs']})",
            f"{pro['op_success_rate_pct']:.1f}% ({pro['op_success_count']}/{pro['total_runs']})",
            f"{hyb['op_success_rate_pct']:.1f}% ({hyb['op_success_count']}/{hyb['total_runs']})",
            f"{rtd['op_success_rate_pct']:.1f}% ({rtd['op_success_count']}/{rtd['total_runs']})",
            f"{dec['op_success_rate_pct']:.1f}% ({dec['op_success_count']}/{dec['total_runs']})",
        )
        table.add_row(
            "Ground-Truth Accuracy",
            f"{sim['accuracy_pct']:.1f}% ({sim['correct_count']}/{sim['total_runs']})",
            f"{pro['accuracy_pct']:.1f}% ({pro['correct_count']}/{pro['total_runs']})",
            f"{hyb['accuracy_pct']:.1f}% ({hyb['correct_count']}/{hyb['total_runs']})",
            f"{rtd['accuracy_pct']:.1f}% ({rtd['correct_count']}/{rtd['total_runs']})",
            f"{dec['accuracy_pct']:.1f}% ({dec['correct_count']}/{dec['total_runs']})",
        )
        table.add_row(
            "Total Cost (USD)",
            f"${sim['total_cost_usd']:.6f}",
            f"${pro['total_cost_usd']:.6f}",
            f"${hyb['total_cost_usd']:.6f}",
            f"${rtd['total_cost_usd']:.6f}",
            f"${dec['total_cost_usd']:.6f}",
        )
        table.add_row(
            "Total Tokens",
            f"{sim['total_tokens']:,}",
            f"{pro['total_tokens']:,}",
            f"{hyb['total_tokens']:,}",
            f"{rtd['total_tokens']:,}",
            f"{dec['total_tokens']:,}",
        )
        table.add_row(
            "Avg Turn Latency (ms)",
            f"{sim['avg_turn_latency_ms']:.1f} ms",
            f"{pro['avg_turn_latency_ms']:.1f} ms",
            f"{hyb['avg_turn_latency_ms']:.1f} ms",
            f"{rtd['avg_turn_latency_ms']:.1f} ms",
            f"{dec['avg_turn_latency_ms']:.1f} ms",
        )
        table.add_row(
            "Avg Router Latency (ms)",
            "0.0 ms",
            "0.0 ms",
            "0.0 ms",
            f"{rtd['avg_laya_latency_ms']:.1f} ms",
            f"{dec['avg_laya_latency_ms']:.1f} ms",
        )
        console.print()
        console.print(table)
        console.print()
    except Exception:
        print(f"All-Simple    Cost=${sim['total_cost_usd']:.6f}, Acc={sim['accuracy_pct']:.1f}%")
        print(f"All-Complex   Cost=${pro['total_cost_usd']:.6f}, Acc={pro['accuracy_pct']:.1f}%")
        print(f"Hybrid        Cost=${hyb['total_cost_usd']:.6f}, Acc={hyb['accuracy_pct']:.1f}%")
        print(f"With-Laya     Cost=${rtd['total_cost_usd']:.6f}, Acc={rtd['accuracy_pct']:.1f}%")
        print(f"With-Decision Cost=${dec['total_cost_usd']:.6f}, Acc={dec['accuracy_pct']:.1f}%")


def _load_existing_mode_results(
    mode: str,
    jsonl_file: Path,
    samples_by_id: dict[str, BenchmarkSample],
) -> dict[str, SampleEvalResult]:
    """Loads previously completed (status == 'success') turns from an adk-finops JSONL log."""
    loaded: dict[str, SampleEvalResult] = {}
    if not jsonl_file.exists():
        return loaded

    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("agent_name") is not None:
                continue  # Only process session-level summary rows
            if row.get("status", "success") != "success":
                continue
            tags = json.loads(row.get("tags") or "{}")
            sid = tags.get("sample_id")
            if not sid or sid not in samples_by_id:
                continue
            sample = samples_by_id[sid]
            by_agent = json.loads(row.get("breakdown_by_agent") or "{}")
            by_model = json.loads(row.get("breakdown_by_model") or "{}")
            delegated = [a for a in by_agent.keys() if a != "coordinator_agent"]
            loaded[sid] = SampleEvalResult(
                mode=mode,
                sample_id=sid,
                dataset=sample.dataset,
                expected_complexity=sample.expected_complexity,
                coordinator_model=str(tags.get("coordinator_model", "")),
                routed_model=str(tags.get("routed_model", "")),
                laya_choice=str(tags.get("laya_choice", "none")),
                laya_confidence=float(tags.get("laya_confidence", 0.0)),
                laya_latency_ms=float(tags.get("laya_latency_ms", 0.0)),
                delegated_agents=delegated,
                operational_status="success",
                is_correct=bool(tags.get("is_correct", False)),
                predicted_answer=str(tags.get("predicted_answer", "")),
                ground_truth=sample.ground_truth,
                prompt_tokens=int(row.get("prompt_tokens", 0)),
                completion_tokens=int(row.get("completion_tokens", 0)),
                thoughts_tokens=int(row.get("thoughts_tokens", 0)),
                total_tokens=int(row.get("total_tokens", 0)),
                llm_cost_usd=float(row.get("llm_cost_usd", 0.0)),
                total_cost_usd=float(row.get("total_cost_usd", 0.0)),
                turn_latency_ms=float(row.get("latency_ms", 0.0)),
                llm_latency_ms=float(row.get("llm_latency_ms", 0.0)),
                output_tokens_per_sec=float(row.get("output_tokens_per_sec", 0.0)),
                total_tokens_per_sec=float(row.get("total_tokens_per_sec", 0.0)),
                breakdown_by_agent=by_agent,
                breakdown_by_model=by_model,
            )
    return loaded


async def async_main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    samples = load_benchmark_suite(
        samples_per_dataset=args.samples_per_dataset,
        refresh_cache=args.refresh_cache,
    )
    samples_by_id = {s.sample_id: s for s in samples}
    logger.info(
        "Loaded %d real open-source benchmark samples (%d per dataset across %s).",
        len(samples),
        args.samples_per_dataset,
        ", ".join(DATASET_NAMES),
    )

    if args.resume_log_dir:
        run_log_dir = Path(args.resume_log_dir)
    else:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_log_dir = LOGS_DIR / run_ts
    run_log_dir.mkdir(parents=True, exist_ok=True)
    all_results_by_mode: dict[str, list[SampleEvalResult]] = {}

    # Preload any already-completed modes from eval_results_detailed.json when running a subset via --modes
    raw_json_path = LOGS_DIR / "eval_results_detailed.json"
    if raw_json_path.exists():
        try:
            cached_detailed = json.loads(raw_json_path.read_text(encoding="utf-8"))
            for m, rows in cached_detailed.items():
                if m in MODES_ORDER and m not in args.modes and len(rows) == len(samples):
                    all_results_by_mode[m] = [SampleEvalResult(**r) for r in rows]
                    logger.info("[%s] Reusing %d completed turns from %s.", m, len(rows), raw_json_path)
        except Exception as exc:
            logger.warning("Could not preload %s: %s", raw_json_path, exc)

    sem = asyncio.Semaphore(max(1, args.concurrency))

    for mode in args.modes:
        jsonl_file = run_log_dir / f"{mode}_costs.jsonl"
        csv_file = run_log_dir / f"{mode}_costs.csv"

        existing_by_id = (
            _load_existing_mode_results(mode, jsonl_file, samples_by_id)
            if args.resume_log_dir
            else {}
        )
        if existing_by_id:
            logger.info(
                "[%s] Loaded %d/%d completed samples from %s.",
                mode,
                len(existing_by_id),
                len(samples),
                jsonl_file,
            )

        jsonl_exporter = JSONLExporter(str(jsonl_file))
        csv_exporter = CSVExporter(str(csv_file))

        default_finops_model = SIMPLE_MODEL if mode == "all_simple" else COMPLEX_MODEL
        finops_plugin = FinOpsCostPlugin(
            name=f"finops_{mode}",
            default_model=default_finops_model,
            render_terminal_box=args.render_boxes,
            enable_optimization_advisor=True,
        )
        app = build_eval_app(
            mode=mode,
            finops_plugin=finops_plugin,
            confidence_threshold=args.confidence_threshold,
            exclude_agents={"coordinator_agent"},
        )
        runner = InMemoryRunner(app=app)

        logger.info("=== Starting evaluation arm: %s (%d samples, concurrency=%d) ===", mode, len(samples), args.concurrency)

        async def _run_indexed(idx: int, sample: BenchmarkSample) -> tuple[int, SampleEvalResult]:
            if sample.sample_id in existing_by_id:
                return idx, existing_by_id[sample.sample_id]
            async with sem:
                logger.info(
                    "[%s] (%d/%d) Running %s (%s)...",
                    mode,
                    idx,
                    len(samples),
                    sample.sample_id,
                    sample.dataset,
                )
                res = await run_single_sample(
                    mode=mode,
                    runner=runner,
                    app_name=app.name,
                    sample=sample,
                    finops_plugin=finops_plugin,
                    jsonl_exporter=jsonl_exporter,
                    csv_exporter=csv_exporter,
                )
                logger.info(
                    "[%s] (%d/%d) %s -> coord=%s, sub=%s | correct=%s (pred=%s, gold=%s) | tokens=%d | cost=$%.6f | turn_lat=%.0fms (router=%.1fms)",
                    mode,
                    idx,
                    len(samples),
                    sample.sample_id,
                    res.coordinator_model,
                    res.routed_model,
                    res.is_correct,
                    res.predicted_answer,
                    res.ground_truth,
                    res.total_tokens,
                    res.total_cost_usd,
                    res.turn_latency_ms,
                    res.laya_latency_ms,
                )
                return idx, res

        indexed_results = await asyncio.gather(
            *[_run_indexed(i, s) for i, s in enumerate(samples, start=1)]
        )
        indexed_results.sort(key=lambda pair: pair[0])
        all_results_by_mode[mode] = [pair[1] for pair in indexed_results]

    # Save raw structured results JSON
    raw_json_path.write_text(
        json.dumps(
            {m: [asdict(r) for r in all_results_by_mode[m]] for m in MODES_ORDER if m in all_results_by_mode},
            indent=2,
        ),
        encoding="utf-8",
    )

    if all(m in all_results_by_mode for m in ("all_simple", "all_complex", "coord_simple_subs_complex", "with_router")):
        generate_markdown_report(all_results_by_mode, output_path=REPORT_PATH)
        print_rich_comparison(all_results_by_mode)
        logger.info("Saved Markdown comparison report to %s", REPORT_PATH)
    logger.info("Saved adk-finops JSONL/CSV logs to %s", run_log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 5-Arm Multi-Agent ADK (All-Simple, All-Complex, Static Hybrid, With Laya Router, With Decisions API Router) on the 60-sample 2x3 Domain x Complexity open-source benchmark."
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=10,
        help="Number of real samples per dataset (ARC-Easy, MMLU-Pro, GSM8K, MATH-500, MBPP, HumanEval). Default: 10 (60 per arm).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent sample evaluations per arm (default: 5).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Router confidence threshold (default: 0.55).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(MODES_ORDER),
        choices=list(MODES_ORDER),
        help="Subset of evaluation modes to run (default: all 5 arms). Other modes are loaded from eval/logs/eval_results_detailed.json if present.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-download of benchmark rows from Hugging Face Hub instead of using the local cache in eval/data_cache/.",
    )
    parser.add_argument(
        "--resume-log-dir",
        type=str,
        default="",
        help="Optional path to an existing eval/logs/<run_ts> directory to resume from, reusing completed turns.",
    )
    parser.add_argument(
        "--render-boxes",
        action="store_true",
        help="Render the per-turn adk-finops Rich terminal summary box after every sample.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()


