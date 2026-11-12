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

## Customizing Routing Questions ([`questions.py`](questions.py))

Edit [`DECISIONS_COMPLEXITY_QUESTIONS`](questions.py) or pass a custom `questions` dictionary to `DecisionsModelRouterPlugin(questions=...)` using either `choice` or `noul` question types from the [Gemini Decisions API specification](https://ai.google.dev/gemini-api/docs/decisions-api#question-types):

```python
CUSTOM_QUESTIONS = {
    "complexity": {
        "type": "choice",
        "instructions": "How much reasoning depth is required to answer `user_message`?",
        "criteria": {
            "simple": "Greetings, routine FAQs, elementary lookups, or short helper snippets.",
            "complex": "Multi-step math proofs, deep architectural reasoning, or complex debugging.",
        },
    }
}
```

---

## Comparing `laya_adk_model_router` vs. `decisions_adk_model_router`

| Dimension | [`laya_adk_model_router`](../laya_adk_model_router) | [`decisions_adk_model_router`](.) |
| :--- | :--- | :--- |
| **Underlying Engine** | Local `laya-typed-decisions` (`convaiinnovations/laya`) | Cloud Google **Gemini Decision 1** (`gemini-decision-1`, `v1beta`) |
| **Where It Runs** | On-device (CPU / Apple Silicon MPS / GPU) | Google Gemini Developer API (`POST /v1beta/decisions`) |
| **Context Window** | Short text turns | Up to **262,144 tokens** (multimodal text, images, audio, video, PDF) |
| **Routing Cost** | $0.00 (local compute) | **$0.10 / 1M input tokens** (0 output tokens) |
| **Typical Overhead** | ~60–130 ms (local) | ~150–250 ms (network + single forward pass) |
| **Question Schema** | Typed `choice` / `noul` / `score` | Native `choice` / `noul` / `score` (up to 1,000 questions/call) |
