# Multi-Agent ADK Model Router Evaluation (`eval/`)

Reproducible **4-arm benchmark suite** comparing a **4-Agent Google ADK application** (`temperature=0.0` pinned across all agents) with and without **[`LayaModelRouterPlugin`](../laya_adk_model_router/plugin.py)** across a **2x3 Domain $\times$ Complexity Matrix** of **60 real open-source Hugging Face benchmark tasks per arm** (`10 ARC-Easy`, `10 MMLU-Pro`, `10 GSM8K`, `10 MATH-500 Level 5`, `10 MBPP`, `10 HumanEval` = **240 multi-agent turns / 480 LLM calls**), instrumented with **[`adk-finops`](https://pypi.org/project/adk-finops/)**.

> [!NOTE]
> **Illustrative Benchmark Disclaimer & Exploring Alternatives:**
> The multi-agent benchmark in this folder is a **simple, illustrative test use case** designed to validate routing mechanics and telemetry—it is **not conclusive across all production workloads or domain distributions**. We encourage users to:
> 1. **Test `laya_adk_model_router` on their own use cases**, customizing [`LAYA_COMPLEXITY_QUESTIONS`](../laya_adk_model_router/questions.py) and `confidence_threshold` for their domain.
> 2. **Explore and compare other routing alternatives**—such as **`jev`** and other lightweight classifiers or semantic/embedding routers—alongside Laya on their specific workloads.

> **Note:** The multi-agent use case and benchmark tasks in this folder are a simplified test setup for demonstration purposes and are **not conclusive for all scenarios**. Because routing gains depend on your specific traffic distribution, agent architecture, and domain prompts, it is recommended to customize [`LAYA_COMPLEXITY_QUESTIONS`](../laya_adk_model_router/questions.py), test the router on your own use cases, and explore other routing alternatives (such as `jev`) for comparison.

---

## What This Evaluates

### 1. Multi-Agent ADK Hierarchy ([`multi_agent.py`](multi_agent.py))
All 4 agents are configured with `generate_content_config=types.GenerateContentConfig(temperature=0.0)` for deterministic evaluation:
- **`coordinator_agent`** *(Root Orchestrator)*: Triages incoming user queries and delegates via `transfer_to_agent` to one of three specialist sub-agents, each of which faces **both Simple (`10`) and Complex (`10`) tasks within its own domain**:
  - **`fast_qa_agent` (20 QA tasks)**: Handles `10 ARC-Easy` (`simple`) science MCQs and `10 MMLU-Pro` (`complex`) 10-option multi-paragraph analytical case reasoning questions.
  - **`math_analyst_agent` (20 Math tasks)**: Solves `10 GSM8K` (`simple`) elementary word problems and `10 MATH-500 Level 5` (`complex`) Olympiad mathematics problems.
  - **`code_engineer_agent` (20 Coding tasks)**: Implements `10 MBPP` (`simple`) basic Python helper functions and `10 HumanEval` (`complex`) algorithmic coding functions.

### 2. Compared Experimental Arms (4-Arm Benchmark)
| Arm / Mode | Plugins Attached | Multi-Agent Routing Behavior |
| :--- | :--- | :--- |
| **`all_simple` (All-Flash-Lite Baseline)** | `[FinOpsCostPlugin()]` | All 4 agents pinned to `SIMPLE_MODEL` (`gemini-3.1-flash-lite`). |
| **`all_complex` (All-Pro Baseline)** | `[FinOpsCostPlugin()]` | All 4 agents pinned to `COMPLEX_MODEL` (`gemini-3.1-pro-preview`). |
| **`coord_simple_subs_complex` (Static Hybrid Baseline)** | `[FinOpsCostPlugin()]` | `coordinator_agent` fixed to `SIMPLE_MODEL` (`gemini-3.1-flash-lite`) and all 3 specialist sub-agents fixed to `COMPLEX_MODEL` (`gemini-3.1-pro-preview`) without dynamic routing. |
| **`with_router` (Role-Aware Laya Router)** | `[LayaModelRouterPlugin(exclude_agents={"coordinator_agent"}), FinOpsCostPlugin()]` | `coordinator_agent` stays on `SIMPLE_MODEL` (`gemini-3.1-flash-lite`) for fast triage (`transfer_to_agent`), while specialist sub-agents (`fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) are dynamically routed per query by Laya (`laya-typed-decisions`) between `SIMPLE_MODEL` (`gemini-3.1-flash-lite`) and `COMPLEX_MODEL` (`gemini-3.1-pro-preview`). |

### 3. Open-Source Benchmarks ([`dataset_loader.py`](dataset_loader.py))
Loaded from the local cache (`eval/data_cache/benchmark_samples_v2.json`) if already present on disk, or fetched from the [Hugging Face Datasets Server REST API](https://huggingface.co/docs/dataset-viewer/rows) on first run (or when `--refresh-cache` is passed):
- **[ARC-Easy (`allenai/ai2_arc`)](https://huggingface.co/datasets/allenai/ai2_arc)** (`simple`, `fast_qa_agent`): Grade-school science 4-option multiple-choice questions (`test` split).
- **[MMLU-Pro (`TIGER-Lab/MMLU-Pro`)](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)** (`complex`, `fast_qa_agent`): 10-option (`A`–`J`) multi-paragraph graduate/professional analytical case questions (`test` split).
- **[GSM8K (`openai/gsm8k`)](https://huggingface.co/datasets/openai/gsm8k)** (`simple`, `math_analyst_agent`): Elementary grade-school arithmetic word problems (`test` split).
- **[MATH-500 Level 5 (`HuggingFaceH4/MATH-500`)](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)** (`complex`, `math_analyst_agent`): Level-5 competition mathematics problems (`test` split).
- **[MBPP (`google-research-datasets/mbpp`)](https://huggingface.co/datasets/google-research-datasets/mbpp)** (`simple`, `code_engineer_agent`): Simple Python helper functions (`test` split), verified against unit-test `assert` statements.
- **[HumanEval (`openai/openai_humaneval`)](https://huggingface.co/datasets/openai/openai_humaneval)** (`complex`, `code_engineer_agent`): Algorithmic Python coding tasks (`test` split, tasks `HumanEval/120`–`129`), verified against the full `check(candidate)` assertion suite.

### 4. Operational Status vs. Answer Accuracy
- **`adk-finops` Task Status (`status`)**: Reserved strictly for **operational health** (`"success"` vs. `"error"`/`"failed"` when an API/tool exception occurs, a budget circuit-breaker trips, or an agent returns empty output).
- **Ground-Truth Accuracy (`is_correct`)**: Evaluated separately by deterministic ground-truth verifiers to report **Accuracy (%)** and **Effective Cost per Correct Answer ($)** without conflating wrong answers with system failures.

---

## Installation & Running

From the `model_router/` root directory:

```bash
# 1. Install evaluation dependencies (adk-finops[rich])
uv pip install -r eval/requirements-eval.txt

# 2. Run the 60-sample 4-arm evaluation benchmark (uses cached data in eval/data_cache/benchmark_samples_v2.json if present)
python -m eval.run_eval --samples-per-dataset 10 --concurrency 5

# Optional: Resume from an existing run log directory (reusing completed turns)
python -m eval.run_eval --samples-per-dataset 10 --resume-log-dir eval/logs/<run_timestamp>

# Optional: Force re-downloading dataset samples from Hugging Face Hub
python -m eval.run_eval --samples-per-dataset 10 --refresh-cache

# Optional: Render per-turn adk-finops Rich summary boxes during execution
python -m eval.run_eval --samples-per-dataset 10 --render-boxes
```

---

## Inspecting Telemetry & Reports

Running `eval.run_eval` produces:
1. **[`eval/EVAL_REPORT.md`](EVAL_REPORT.md)**: Side-by-side 4-arm Markdown report comparing **All-Simple**, **All-Complex**, **Static Hybrid (`Coord=Simple, Subs=Complex`)**, and **With Laya Router** across Executive KPIs, **Laya Router Classification Audit (2x3 Domain $\times$ Complexity Matrix)**, Per-Benchmark Breakdown, Per-Agent Attribution (`breakdown_by_agent`), Model Distribution (`breakdown_by_model`), and Sample-Level Logs.
2. **`eval/logs/<timestamp>/{all_simple,all_complex,coord_simple_subs_complex,with_router}_costs.{jsonl,csv}`** (plus `eval/logs/eval_results_detailed.json`).

You can also run the `adk-finops` CLI report or interactive web dashboard directly on `eval/logs`:

```bash
# Terminal summary report across exported evaluation logs
adk-finops report --log-dir eval/logs --no-bigquery

# Launch the Near-Live FinOps Web Dashboard on port 8088
adk-finops dashboard --log-dir eval/logs --port 8088
```
