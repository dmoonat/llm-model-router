# `decisions_adk_model_router` — Google Gemini Decisions API Model Router for ADK

Dynamic model routing plugin and callback for [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) powered by Google's **[Gemini Decisions API](https://ai.google.dev/gemini-api/docs/decisions-api)** (`gemini-decision-1`).

Instead of generating conversational text tokens and parsing them with regex, `decisions_adk_model_router` calls `gemini-decision-1` (`v1beta`) to evaluate typed `choice` / `noul` / `score` questions against the user prompt and recent conversation context in a single forward pass with **0 output tokens** (`$0.10 / 1M` input tokens, `$0.00` output tokens).

---

## Key Features

- **Zero Output Tokens (`gemini-decision-1`)**: Evaluates typed routing rubrics in a single forward pass with `total_output_tokens = 0`, returning calibrated option probabilities (`probabilities`) and `confidence`.
- **SDK + REST Dual Compatibility**: Uses `google.genai.Client(http_options={"api_version": "v1beta"}).decisions.create(...)` when `google-genai >= 3.2` is installed, and automatically falls back to `POST https://generativelanguage.googleapis.com/v1beta/decisions` on older `google-genai` versions.
- **Hybrid Vertex AI + Gemini Developer API Support**: Works seamlessly even when your ADK agents run on Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=TRUE`) while `gemini-decision-1` authenticates via `GEMINI_API_KEY` (or `DECISIONS_API_KEY`).
- **Role-Aware Multi-Agent Routing (`exclude_agents`)**: Pin lightweight orchestrator/triage agents (e.g., `exclude_agents={"coordinator_agent"}`) to `SIMPLE_MODEL` (`gemini-3.1-flash-lite`) while dynamically routing specialist sub-agents (`fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) between `SIMPLE_MODEL` and `COMPLEX_MODEL` (`gemini-3.1-pro-preview`) with per-invocation caching (`invocation_id`).
- **Session State FinOps Telemetry**: Stores per-turn routing metadata in `callback_context.state["decisions_routing"]` (`routed_model`, `choice`, `confidence`, `probabilities`, `input_tokens`, `output_tokens=0`, `decision_cost_usd`, and `decisions_latency_ms`).

---

## Configuration (`.env`)

Copy `.env.example` to `.env` inside `decisions_adk_model_router/` and set your `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey):

```env
GEMINI_API_KEY=your_gemini_api_key_here

# Target Models
DECISION_MODEL=gemini-decision-1
SIMPLE_MODEL=gemini-3.1-flash-lite
COMPLEX_MODEL=gemini-3.1-pro-preview
```

---

## Quickstart

### 1. App-Level Plugin (`DecisionsModelRouterPlugin`)

```python
from google.adk.agents import Agent
from google.adk.apps import App
from decisions_adk_model_router import DecisionsModelRouterPlugin

root_agent = Agent(
    name="root_agent",
    model="gemini-3.1-pro-preview",
    instruction="You are a helpful assistant.",
)

app = App(
    name="my_adk_app",
    root_agent=root_agent,
    plugins=[
        DecisionsModelRouterPlugin(
            confidence_threshold=0.7,
            simple_model="gemini-3.1-flash-lite",
            complex_model="gemini-3.1-pro-preview",
        )
    ],
)
```

### 2. Multi-Agent Hierarchy with `exclude_agents`

```python
from google.adk.apps import App
from decisions_adk_model_router import DecisionsModelRouterPlugin

app = App(
    name="multi_agent_app",
    root_agent=coordinator_agent,
    plugins=[
        DecisionsModelRouterPlugin(
            confidence_threshold=0.7,
            exclude_agents={"coordinator_agent"},  # Coordinator stays on SIMPLE_MODEL; sub-agents are dynamically routed
        )
    ],
)
```

### 3. Standalone `before_model_callback` (`decisions_router`)

```python
from google.adk.agents import Agent
from decisions_adk_model_router import decisions_router

root_agent = Agent(
    name="root_agent",
    model="gemini-3.1-pro-preview",
    instruction="You are a helpful assistant.",
    before_model_callback=decisions_router,
)
```

---

## Empirical Head-to-Head Benchmark (`Laya` vs. `Gemini Decision 1`, N=60 Tasks)

Evaluated on the 60-sample `2x3 Domain x Complexity Matrix` (`ARC-Easy`, `MMLU-Pro`, `GSM8K`, `MATH-500`, `MBPP`, `HumanEval`) in [`eval/EVAL_REPORT.md`](../eval/EVAL_REPORT.md):

| Dimension | [`laya_adk_model_router`](../laya_adk_model_router) (`laya-typed-decisions`) | [`decisions_adk_model_router`](.) (`gemini-decision-1`) |
| :--- | :--- | :--- |
| **Underlying Engine** | Local `laya-typed-decisions` (`421M` ModernBERT-large) | Cloud Google **Gemini Decision 1** (`v1beta`) |
| **Where It Runs** | On-device (CPU / Apple Silicon MPS / GPU) | Google Gemini Developer API (`POST /v1beta/decisions`) |
| **Context Window** | 1,024 tokens (text) | **262,144 tokens** (multimodal text, images, audio, video, PDF) |
| **Tier Classification Accuracy (`N=60`)** | **100.0% (60/60)** | **100.0% (60/60)** |
| **End-to-End Task Accuracy (`N=60`)** | **100.0% (60/60)** (+13.3% vs `All-Simple`) | **100.0% (60/60)** (+13.3% vs `All-Simple`) |
| **Avg Router Confidence (Min–Max)** | **0.71** (`0.56–0.83`) | **0.9801** (`0.9079–0.9991`) |
| **Avg Router Latency (p95)** | **102.7 ms** (p95: `151.2 ms`) | **606.9 ms** (p95: `1,144.4 ms`) |
| **Router Inference Cost (`N=60`)** | **$0.000000** (local compute, 0 output tokens) | **$0.002394** (`23,941` input tokens at `$0.10/1M`, 0 output tokens) |
| **Total 60-Task Multi-Agent Cost** | **$1.119405** (**-27.5%** vs `All-Complex`) | **$1.069982** (**-30.7%** vs `All-Complex`) |
| **Simple-Tier Tasks Cost (`N=30`)** | **$0.016566** (**-94.7%** vs `All-Complex`) | **$0.017419** (**-94.4%** vs `All-Complex`) |
| **Avg End-to-End Turn Latency** | **17,747.2 ms** (**-32.1%** vs `All-Complex`) | **17,742.1 ms** (**-32.1%** vs `All-Complex`) |

```bash
# Run the 60-sample evaluation for `with_decisions_router` standalone from model_router/:
python -m eval.run_eval --samples-per-dataset 10 --concurrency 5 --modes with_decisions_router
```
