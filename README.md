# Laya ADK Model Router (`laya_adk_model_router`)

Dynamic LLM model routing for [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) powered by **[Laya (`convaiinnovations/laya`)](https://huggingface.co/convaiinnovations/laya)** — a fast (~33–100ms), non-autoregressive System-1 decision model.

Instead of sending every prompt to a heavy reasoning model or paying the latency and cost of an LLM router call, `laya_adk_model_router` evaluates incoming user prompts in a single forward pass and dynamically routes ADK's `LlmRequest.model` between **`gemini-3.1-flash-lite`** (`SIMPLE_MODEL`) and **`gemini-3.1-pro-preview`** (`COMPLEX_MODEL`).

---

## Directory Structure

```text
model_router/
├── .gitignore
├── README.md
├── requirements.txt
├── laya_adk_model_router/          # Local System-1 Laya Router (`convaiinnovations/laya`)
│   ├── __init__.py
│   ├── .env
│   ├── agent.py
│   ├── plugin.py
│   └── questions.py
├── decisions_adk_model_router/     # Cloud Google Gemini Decisions API Router (`gemini-decision-1`)
│   ├── __init__.py
│   ├── .env
│   ├── .env.example
│   ├── README.md
│   ├── agent.py
│   ├── plugin.py
│   └── questions.py
└── eval/
    ├── README.md
    ├── EVAL_REPORT.md
    ├── requirements-eval.txt
    ├── dataset_loader.py
    ├── multi_agent.py
    ├── run_eval.py
    └── data_cache/
        └── benchmark_samples_v2.json
```

> **Also in this repository**: See **[`decisions_adk_model_router/README.md`](decisions_adk_model_router/README.md)** for the sibling ADK router powered by Google's **[Gemini Decisions API](https://ai.google.dev/gemini-api/docs/decisions-api)** (`gemini-decision-1`, 0 output tokens, `$0.10 / 1M` input tokens).

---

## How It Works

```mermaid
flowchart LR
    U["User Prompt + Recent Context"] --> L["Laya System-1 Router (~100ms)"]
    L -->|"simple & conf >= threshold"| F["gemini-3.1-flash-lite (SIMPLE_MODEL)"]
    L -->|"complex or conf < threshold"| P["gemini-3.1-pro-preview (COMPLEX_MODEL)"]
```

1. **Intercept (`before_model_callback`)**: When a new user message arrives, [`LayaModelRouterPlugin`](laya_adk_model_router/plugin.py) (or the standalone [`laya_router`](laya_adk_model_router/plugin.py) callback) extracts `user_message` and up to 4 turns of `recent_context`.
2. **Role-Aware Agent Exclusions (`exclude_agents`)**: In multi-agent hierarchies, lightweight orchestrator/triage agents (e.g., `coordinator_agent`) can be listed in `exclude_agents={"coordinator_agent"}` so they remain pinned to `SIMPLE_MODEL` (`gemini-3.1-flash-lite`), while specialist sub-agents are dynamically routed by Laya.
3. **System-1 Classification & Latency Tracking**: [`laya.Router`](https://huggingface.co/convaiinnovations/laya) evaluates [`LAYA_COMPLEXITY_QUESTIONS`](laya_adk_model_router/questions.py) in a single forward pass (protected by an `asyncio.Lock` for safe concurrent execution), returning a calibrated `choice` (`"simple"` or `"complex"`), `answer_confidence`, and recording `laya_latency_ms` inside `callback_context.state["laya_routing"]`.
4. **Dynamic Model Swap**:
   - **`simple`** (with `answer_confidence >= confidence_threshold`) $\rightarrow$ routes to **`gemini-3.1-flash-lite`** (`SIMPLE_MODEL`).
   - **`complex`** (or low confidence `< confidence_threshold`) $\rightarrow$ upgrades to **`gemini-3.1-pro-preview`** (`COMPLEX_MODEL`).
5. **Turn Persistence**: The selected model is saved in `callback_context.state["routed_model"]` keyed by `invocation_id` so intermediate tool-response or sub-agent delegation steps within the same turn reuse the classification without redundant router calls.

> **Disclaimer — Customize [`LAYA_COMPLEXITY_QUESTIONS`](laya_adk_model_router/questions.py) for Your Use Case:**
> The default `simple` and `complex` criteria in [`laya_adk_model_router/questions.py`](laya_adk_model_router/questions.py) are general-purpose reasoning, math, coding, and IT examples. Because Laya conditions directly on your question `instructions` and `criteria` descriptions at inference time, **you should update `LAYA_COMPLEXITY_QUESTIONS` in `questions.py` (as well as `confidence_threshold`) to reflect the exact task boundaries, domain terminology, and escalation rules of your specific application or use case.**

---

## Laya Model Choice

The [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) repository holds all three checkpoints and serves as the hub for the family. The English checkpoint is at the repo root; the other two are bundled subfolders, and only the one you request is downloaded:

| Checkpoint | Backbone Encoder | Params | Context | Best at |
| :--- | :--- | :--- | :--- | :--- |
| **[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)** (repo root) | ModernBERT-large | 421M | 512 | English text, guardrails, email triage |
| **[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)** | mmBERT-base | 322M | 1024 (up to 8k) | 100+ languages, ~2.2x faster |
| **[`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions)** | ModernBERT-large | 421M | 1024 | The four typed-decisions workflows (0.766 acc) |

By default, [`plugin.py`](laya_adk_model_router/plugin.py) preloads **`laya-typed-decisions`**:

```python
laya_engine = Router(default="laya-typed-decisions").preload(["laya-typed-decisions"])
```

You can switch to `"english"` or `"multilingual"` (or pass `model="multilingual"` to `LayaModelRouterPlugin(model="multilingual")`) depending on your workload.

---

## Prerequisites & Installation

From the `llm-model-router/` root directory:

```bash
uv venv
uv pip install -r requirements.txt
```

Configure your `.env` file inside `laya_adk_model_router/.env`:

```env
# Vertex AI via Google Cloud Application Default Credentials (ADC)
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global

# Target Models for Laya Router
SIMPLE_MODEL=gemini-3.1-flash-lite
COMPLEX_MODEL=gemini-3.1-pro-preview
```

*(Alternatively, set `GOOGLE_API_KEY=...` with `GOOGLE_GENAI_USE_VERTEXAI=FALSE` to use the Gemini Developer API.)*

---

## Running the Agent

From the `llm-model-router/` root directory:

```bash
# Launch the ADK Dev UI and select "laya_adk_model_router"
adk web

# Or run interactively in the terminal
adk run laya_adk_model_router
```

> **Note:** At startup, `Router(default="laya-typed-decisions").preload(["laya-typed-decisions"])` downloads (if not already cached in `~/.cache/huggingface/hub/`) and loads the `laya-typed-decisions` checkpoint into memory so the very first user request has zero cold-start latency.

---

## Example Queries (Simple vs. Complex)

Use the following prompts in `adk web` or `adk run laya_adk_model_router` to test how the router classifies and dispatches requests:

### Simple Queries (Routed to `gemini-3.1-flash-lite`)
These prompts require no multi-step reasoning, architectural tradeoffs, or deep debugging (`choice="simple"` with `confidence >= 0.70`):

- `"Hi! What are the standard IT helpdesk operating hours?"`
- `"Convert 72 degrees Fahrenheit to Celsius."`
- `"What is the capital of Australia and what currency do they use?"`
- `"Summarize this sentence in 5 words: The deployment completed successfully with zero downtime."`

**Expected terminal log:**
```text
[LayaRouter] agent=root_agent Routed to gemini-3.1-flash-lite (choice=simple, conf=0.81, laya_latency=64.2ms, laya_model=typed-decisions)
```

### Complex Queries (Routed to `gemini-3.1-pro-preview`)
These prompts trigger complex classification (`choice="complex"` or `confidence < 0.70`) such as code debugging, system architecture, or multi-step analytical reasoning:

- `"We're seeing a sporadic DeadlockException in our PostgreSQL order processing service when two workers update inventory rows concurrently. Walk through the lock ordering issue and show how to fix it in SQLAlchemy using SELECT FOR UPDATE SKIP LOCKED."`
- `"Design a multi-region active-active event-driven architecture on GCP for processing 50,000 financial transactions per second with strict idempotency, RPO=0, and sub-100ms p99 latency. Compare Spanner vs. Bigtable for the ledger layer."`
- `"Analyze the time and space complexity of a concurrent lock-free ring buffer in Rust, and write a thread-safe implementation that avoids the ABA problem."`
- `"Our Kubernetes pod is OOMKilled every 6 hours under steady traffic. Give me a step-by-step root-cause diagnostic plan for profiling Python asyncio memory leaks in production without taking down the service."`

**Expected terminal log:**
```text
[LayaRouter] agent=root_agent Upgraded to gemini-3.1-pro-preview (choice=complex, conf=0.68, laya_latency=112.4ms, laya_model=typed-decisions)
```

---

## Usage Patterns

### 1. Single-Agent ADK App Plugin (Default)

```python
from google.adk.agents import Agent
from google.adk.apps import App
from laya_adk_model_router import LayaModelRouterPlugin

root_agent = Agent(
    name="root_agent",
    model="gemini-3.1-pro-preview",
    instruction="You are an AI assistant.",
)

app = App(
    name="laya_adk_model_router",
    root_agent=root_agent,
    plugins=[LayaModelRouterPlugin(confidence_threshold=0.7)],
)
```

### 2. Multi-Agent Hierarchy with Role-Aware Exclusions (`exclude_agents`)

Keep lightweight orchestrator/triage agents pinned to `SIMPLE_MODEL` (`gemini-3.1-flash-lite`) while Laya dynamically routes specialist sub-agents between `SIMPLE_MODEL` and `COMPLEX_MODEL`:

```python
from google.adk.agents import Agent
from google.adk.apps import App
from laya_adk_model_router import COMPLEX_MODEL, SIMPLE_MODEL, LayaModelRouterPlugin

coordinator_agent = Agent(
    name="coordinator_agent",
    model=SIMPLE_MODEL,  # Always runs on gemini-3.1-flash-lite
    instruction="Route user requests to the right specialist sub-agent.",
    sub_agents=[fast_qa_agent, math_analyst_agent, code_engineer_agent],
)

app = App(
    name="multi_agent_router_app",
    root_agent=coordinator_agent,
    plugins=[
        LayaModelRouterPlugin(
            confidence_threshold=0.55,
            exclude_agents={"coordinator_agent"},
        )
    ],
)
```

### 3. Standalone `before_model_callback`

```python
from google.adk.agents import Agent
from laya_adk_model_router import laya_router

root_agent = Agent(
    name="root_agent",
    model="gemini-3.1-pro-preview",
    instruction="You are an AI assistant.",
    before_model_callback=laya_router,
)
```

---

## Multi-Agent Benchmark Evaluation (`eval/`)

The [`eval/`](eval/README.md) directory contains a reproducible **4-arm evaluation suite** comparing a 4-Agent ADK hierarchy (`coordinator_agent` $\rightarrow$ `fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) with `temperature=0.0` pinned across all agents on **60 real open-source Hugging Face benchmark tasks per arm** (**240 multi-agent turns / 480 LLM calls**) structured as a **2x3 Domain $\times$ Complexity Matrix** (`10 Simple + 10 Complex` per specialist agent), instrumented with **[`adk-finops`](https://pypi.org/project/adk-finops/)**:

- **`fast_qa_agent` (20 QA tasks)**: `10 ARC-Easy` (`simple`) + `10 MMLU-Pro` (`complex`)
- **`math_analyst_agent` (20 Math tasks)**: `10 GSM8K` (`simple`) + `10 MATH-500 Level 5` (`complex`)
- **`code_engineer_agent` (20 Coding tasks)**: `10 MBPP` (`simple`) + `10 HumanEval` (`complex`)

| Metric | 1. All-Simple (`gemini-3.1-flash-lite`) | 2. All-Complex (`gemini-3.1-pro-preview`) | 3. Static Hybrid (`Coord=Simple, Subs=Complex`) | 4. With Laya Router (`Coord=Simple, Subs=Laya`) | Router vs. All-Simple | Router vs. All-Complex | Router vs. Static Hybrid |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Operational Success Rate** | 100.0% (60/60) | 100.0% (60/60) | 100.0% (60/60) | **100.0% (60/60)** | +0.0% | +0.0% | +0.0% |
| **Ground-Truth Accuracy (%)** | 86.7% (52/60) | 100.0% (60/60) | 100.0% (60/60) | **100.0% (60/60)** | **+13.3% (+8 tasks)** | **+0.0%** | **+0.0%** |
| **Laya Tier Classification Accuracy** | — | — | — | **100.0% (60/60)** | — | — | — |
| **Total Cost (USD)** | $0.053324 | $1.543620 | $1.260022 | **$1.119405** | +1999.2% | **-27.5% ($0.424 saved)** | **-11.2% ($0.141 saved)** |
| **Effective Cost / Correct Answer** | $0.001025 | $0.025727 | $0.021000 | **$0.018657** | +1719.3% | **-27.5%** | **-11.2%** |
| **Simple-Tier Tasks Cost (`N=30`)** | $0.016474 | $0.313328 | $0.250687 | **$0.016566** | +0.6% | **-94.7%** | **-93.4%** |
| **Coordinator Agent Cost (`N=60`)** | $0.012187 | $0.164512 | $0.012218 | **$0.012218** | +0.3% | **-92.6%** | **0.0%** |
| **Thinking Tokens (`thoughts_tokens`)** | 0 | 92,742 | 77,628 | **69,963** | — | **-24.6%** | **-9.9%** |
| **Avg Turn Latency (ms)** | 4,696.1 ms | 26,118.9 ms | 22,831.1 ms | **17,747.2 ms** | +277.9% | **-32.1% (-8.4s)** | **-22.3% (-5.1s)** |
| **Avg Laya Overhead (ms)** | 0.0 ms | 0.0 ms | 0.0 ms | **102.7 ms** | +102.7 ms | **~0.58% of turn** | **~0.58% of turn** |

> [!NOTE]
> **Illustrative Benchmark Disclaimer & Exploring Alternatives:**
> The multi-agent benchmark shown above is a **simple, illustrative test use case** designed to validate routing mechanics and telemetry—it is **not conclusive across all production workloads or domain distributions**. Because routing effectiveness depends heavily on your prompt distribution, latency SLOs, and task boundaries, we strongly encourage you to:
> 1. **Evaluate `laya_adk_model_router` on your own use cases and production traffic**, customizing [`LAYA_COMPLEXITY_QUESTIONS`](laya_adk_model_router/questions.py) and `confidence_threshold` for your domain.
> 2. **Explore and benchmark other routing alternatives**—such as **`jev`** and other lightweight classifiers or semantic/embedding routers—alongside Laya to compare accuracy, latency overhead, and cost savings on your specific workload.

- **Full Evaluation Report**: [`eval/EVAL_REPORT.md`](eval/EVAL_REPORT.md)
- **Evaluation Suite Instructions**: [`eval/README.md`](eval/README.md)

> **Note — Illustrative Test Use Case:**
> The multi-agent hierarchy and 45-sample benchmark (`ARC-Easy`, `MATH-500`, `HumanEval`) shown here are a simplified test setup designed to illustrate how `LayaModelRouterPlugin` and `adk-finops` work together, and the results are **not conclusive across all workloads**. Real-world cost, latency, and accuracy trade-offs depend heavily on your traffic mix (e.g., the ratio of simple vs. complex queries), agent/tool topologies, and domain-specific routing criteria. It is strongly encouraged to customize [`LAYA_COMPLEXITY_QUESTIONS`](laya_adk_model_router/questions.py), evaluate the router directly on your own production use cases, and explore other routing alternatives (such as `jev`) for side-by-side comparison.

```bash
# Install evaluation dependencies and run the 60-sample 4-arm benchmark (uses local cache in eval/data_cache/)
uv pip install -r eval/requirements-eval.txt
python -m eval.run_eval --samples-per-dataset 10 --concurrency 5
```
