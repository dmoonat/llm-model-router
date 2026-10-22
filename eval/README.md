# Multi-Agent ADK Model Router Evaluation (`eval/`)

Reproducible benchmark suite comparing a **4-Agent Google ADK application** with and without **[`LayaModelRouterPlugin`](../laya_adk_model_router/plugin.py)** on **real open-source Hugging Face datasets** (`ARC-Easy`, `MATH-500 Level 5`, `HumanEval`), instrumented with **[`adk-finops`](https://pypi.org/project/adk-finops/)**.

> **Note:** The multi-agent use case and benchmark tasks in this folder are a simplified test setup for demonstration purposes and are **not conclusive for all scenarios**. Because routing gains depend on your specific traffic distribution, agent architecture, and domain prompts, it is recommended to customize [`LAYA_COMPLEXITY_QUESTIONS`](../laya_adk_model_router/questions.py), test the router on your own use cases, and explore other routing alternatives (such as `jev`) for comparison.

---

## What This Evaluates

### 1. Multi-Agent ADK Hierarchy ([`multi_agent.py`](multi_agent.py))
- **`coordinator_agent`** *(Root Orchestrator)*: Triages incoming user queries and delegates via `transfer_to_agent` to one of three specialist sub-agents:
  - **`fast_qa_agent`**: Handles factual lookups, science questions, and multiple-choice questions (`ARC-Easy`).
  - **`math_analyst_agent`**: Solves competition/olympiad mathematical problems step-by-step (`MATH-500 Level 5`).
  - **`code_engineer_agent`**: Implements self-contained Python functions and algorithmic complexity analysis (`HumanEval`).

### 2. Compared Experimental Arms (4-Arm Benchmark)
| Arm / Mode | Plugins Attached | Multi-Agent Routing Behavior |
| :--- | :--- | :--- |
| **`all_simple` (All-Flash-Lite Baseline)** | `[FinOpsCostPlugin()]` | All 4 agents pinned to `SIMPLE_MODEL` (`gemini-3.1-flash-lite`). |
| **`all_complex` (All-Pro Baseline)** | `[FinOpsCostPlugin()]` | All 4 agents pinned to `COMPLEX_MODEL` (`gemini-3.1-pro-preview`). |
| **`coord_simple_subs_complex` (Static Hybrid Baseline)** | `[FinOpsCostPlugin()]` | `coordinator_agent` fixed to `SIMPLE_MODEL` (`gemini-3.1-flash-lite`) and all 3 specialist sub-agents fixed to `COMPLEX_MODEL` (`gemini-3.1-pro-preview`) without dynamic routing. |
| **`with_router` (Role-Aware Laya Router)** | `[LayaModelRouterPlugin(exclude_agents={"coordinator_agent"}), FinOpsCostPlugin()]` | `coordinator_agent` stays on `SIMPLE_MODEL` (`gemini-3.1-flash-lite`) for fast triage (`transfer_to_agent`), while specialist sub-agents (`fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) are dynamically routed by Laya (`laya-typed-decisions`) between `SIMPLE_MODEL` and `COMPLEX_MODEL`. |

### 3. Open-Source Benchmarks ([`dataset_loader.py`](dataset_loader.py))
Loaded from the local cache (`eval/data_cache/benchmark_samples_v2.json`) if already present on disk, or fetched from the [Hugging Face Datasets Server REST API](https://huggingface.co/docs/dataset-viewer/rows) on first run (or when `--refresh-cache` is passed):
- **[ARC-Easy (`allenai/ai2_arc`)](https://huggingface.co/datasets/allenai/ai2_arc)**: Real grade-school science multiple-choice questions (`test` split). Verified via exact match against `answerKey`.
- **[MATH-500 Level 5 (`HuggingFaceH4/MATH-500`)](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)**: Real Level-5 competition mathematics problems (`test` split). Verified against the exact numeric ground truth.
- **[HumanEval (`openai/openai_humaneval`)](https://huggingface.co/datasets/openai/openai_humaneval)**: Real algorithmic Python coding tasks (`test` split, tasks `HumanEval/120`–`134`). Verified by executing generated Python code against the full `check(candidate)` assertion suite.

### 4. Operational Status vs. Answer Accuracy
- **`adk-finops` Task Status (`status`)**: Reserved strictly for **operational health** (`"success"` vs. `"error"`/`"failed"` when an API/tool exception occurs, a budget circuit-breaker trips, or an agent returns empty output).
- **Ground-Truth Accuracy (`is_correct`)**: Evaluated separately by the deterministic ground-truth verifiers to report **Accuracy (%)** and **Effective Cost per Correct Answer ($)** without conflating wrong answers with system failures.

---

## Installation & Running

From the `model_router/` root directory:

```bash
# 1. Install evaluation dependencies (adk-finops[rich])
uv pip install -r eval/requirements-eval.txt

# 2. Run the 4-arm evaluation benchmark (uses cached data in eval/data_cache/benchmark_samples_v2.json if present)
python -m eval.run_eval --samples-per-dataset 15 --concurrency 5

# Optional: Force re-downloading dataset samples from Hugging Face Hub
python -m eval.run_eval --samples-per-dataset 15 --refresh-cache

# Optional: Render per-turn adk-finops Rich summary boxes during execution
python -m eval.run_eval --samples-per-dataset 15 --render-boxes
```

---

## Inspecting Telemetry & Reports

Running `eval.run_eval` produces:
1. **[`eval/EVAL_REPORT.md`](EVAL_REPORT.md)**: Side-by-side 4-arm Markdown report comparing **All-Simple**, **All-Complex**, **Static Hybrid (`Coord=Simple, Subs=Complex`)**, and **With Laya Router** across Executive KPIs, **Laya Router Classification Audit (Confusion Matrix & Confidence Distribution)**, Per-Benchmark Breakdown, Per-Agent Attribution (`breakdown_by_agent`), Model Distribution (`breakdown_by_model`), and Sample-Level Logs.
2. **`eval/logs/<timestamp>/{all_simple,all_complex,coord_simple_subs_complex,with_router}_costs.{jsonl,csv}`** (plus `eval/logs/eval_results_detailed.json`).

You can also run the `adk-finops` CLI report or interactive web dashboard directly on `eval/logs`:

```bash
# Terminal summary report across exported evaluation logs
adk-finops report --log-dir eval/logs --no-bigquery

# Launch the Near-Live FinOps Web Dashboard on port 8088
adk-finops dashboard --log-dir eval/logs --port 8088
```
