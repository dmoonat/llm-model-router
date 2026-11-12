# Multi-Agent ADK Model Router Evaluation Report (5-Arm Benchmark, N=60 per Arm, temperature=0.0)

Comprehensive 5-arm comparison of a **4-Agent Google ADK Hierarchy** (`coordinator_agent` $\rightarrow$ `fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) with `temperature=0.0` evaluated on **60 real open-source Hugging Face benchmark tasks per arm** (**300 total multi-agent turns**) across a **2x3 Domain $\times$ Complexity matrix** (`10 Simple + 10 Complex` per specialist agent) and instrumented with **[`adk-finops`](https://pypi.org/project/adk-finops/)**:

- **`fast_qa_agent` (20 QA tasks)**: `10 ARC-Easy` (Simple QA) + `10 MMLU-Pro` (Complex 10-option Analytical QA)
- **`math_analyst_agent` (20 Math tasks)**: `10 GSM8K` (Simple Math) + `10 MATH-500 Level-5` (Complex Olympiad Math)
- **`code_engineer_agent` (20 Coding tasks)**: `10 MBPP` (Simple Python Helpers) + `10 HumanEval` (Complex Algorithmic Coding)

### Experimental Arms
1. **All-Simple (`all_simple`)**: `coordinator_agent` (`gemini-3.1-flash-lite`) $\rightarrow$ Specialist sub-agents (`gemini-3.1-flash-lite`).
2. **All-Complex (`all_complex`)**: `coordinator_agent` (`gemini-3.1-pro-preview`) $\rightarrow$ Specialist sub-agents (`gemini-3.1-pro-preview`).
3. **Static Hybrid (`coord_simple_subs_complex`)**: `coordinator_agent` fixed to `gemini-3.1-flash-lite` and all 3 specialist sub-agents fixed to `gemini-3.1-pro-preview` (no dynamic router).
4. **Role-Aware Laya Router (`with_router`)**: `coordinator_agent` pinned to `gemini-3.1-flash-lite` via `exclude_agents={"coordinator_agent"}`, while specialist sub-agents (`fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) are dynamically routed per query by [`LayaModelRouterPlugin`](../laya_adk_model_router/plugin.py) (`laya-typed-decisions`, local CPU/MPS, `$0.00` routing cost) between `gemini-3.1-flash-lite` and `gemini-3.1-pro-preview`.
5. **Role-Aware Gemini Decisions API Router (`with_decisions_router`)**: `coordinator_agent` pinned to `gemini-3.1-flash-lite` via `exclude_agents={"coordinator_agent"}`, while specialist sub-agents are dynamically routed per query by [`DecisionsModelRouterPlugin`](../decisions_adk_model_router/plugin.py) (`gemini-decision-1` on `v1beta`, `0` output tokens, `$0.10 / 1M` input tokens) between `gemini-3.1-flash-lite` and `gemini-3.1-pro-preview`.

---

## 1. Executive Summary (All 5 Experimental Arms)

| Metric | 1. All-Simple (`gemini-3.1-flash-lite`) | 2. All-Complex (`gemini-3.1-pro-preview`) | 3. Static Hybrid (`Coord=Simple, Subs=Complex`) | 4. With Laya Router (`laya-typed-decisions`) | 5. With Decisions API Router (`gemini-decision-1`) | Laya vs. All-Complex | Decisions vs. All-Complex | Decisions vs. Laya |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Total Benchmark Tasks** | 60 | 60 | 60 | 60 | 60 | — | — | — |
| **Operational Success Rate (`adk-finops`)** | 100.0% (60/60) | 100.0% (60/60) | 100.0% (60/60) | 100.0% (60/60) | 100.0% (60/60) | +0.0% | +0.0% | +0.0% |
| **Ground-Truth Accuracy (%)** | **86.7%** (52/60) | **100.0%** (60/60) | **100.0%** (60/60) | **100.0%** (60/60) | **100.0%** (60/60) | **+0.0%** | **+0.0%** | **+0.0%** |
| **Total Cost (USD)** | **$0.053324** | **$1.543620** | **$1.260022** | **$1.119405** | **$1.069982** | **-27.5% ($+0.424215)** | **-30.7% ($+0.473638)** | **-4.4%** |
| **Simple-Tier Tasks Cost (`N=30`)** | **$0.016474** | **$0.313328** | **$0.250687** | **$0.016566** | **$0.017419** | **-94.7%** | **-94.4%** | **+5.2%** |
| **Router Inference Cost (`N=60`)** | $0.000000 | $0.000000 | $0.000000 | **$0.000000** (local) | **$0.002394** (23,941 tok) | $0.00 | +$0.002394 | +$0.002394 |
| **Avg Cost / Task (USD)** | $0.000889 | $0.025727 | $0.021000 | $0.018657 | $0.017833 | -27.5% | -30.7% | -4.4% |
| **Effective Cost / Correct Answer** | $0.001025 | $0.025727 | $0.021000 | $0.018657 | $0.017833 | -27.5% | -30.7% | -4.4% |
| **Total Billable Tokens** | 114,139 | 207,385 | 190,879 | 185,684 | 205,461 | -10.5% | -0.9% | +10.7% |
| **Prompt Tokens** | 94,308 | 94,500 | 94,500 | 94,500 | 118,649 | 0.0% | +25.6% | +25.6% |
| **Completion Tokens** | 19,831 | 20,143 | 18,751 | 21,221 | 20,762 | +5.4% | +3.1% | -2.2% |
| **Thinking Tokens (`thoughts_tokens`)** | 0 | 92,742 | 77,628 | 69,963 | 66,050 | -24.6% | -28.8% | -5.6% |
| **Avg Turn Latency (ms)** | **4696.1 ms** | **26118.9 ms** | **22831.1 ms** | **17747.2 ms** | **17742.1 ms** | **-32.1% (+8371.8 ms)** | **-32.1% (+8376.8 ms)** | **-0.0%** |
| **P95 Turn Latency (ms)** | 6202.0 ms | 58400.7 ms | 52561.1 ms | 48488.4 ms | 49423.9 ms | -17.0% | -15.4% | +1.9% |
| **Avg Router Overhead (ms)** | 0.0 ms | 0.0 ms | 0.0 ms | **102.7 ms** (p95: 151.2 ms) | **606.9 ms** (p95: 1144.4 ms) | +102.7 ms | +606.9 ms | **+490.9%** |
| **Avg Output Throughput (`tok/s`)** | 65.6 tok/s | 62.1 tok/s | 62.8 tok/s | 61.0 tok/s | 60.9 tok/s | -1.8% | -2.0% | -0.2% |

---

## 2. Head-to-Head Router Classification Audit: `Laya` vs. `Gemini Decision 1` (2x3 Domain x Complexity Matrix)

- **Laya (`laya-typed-decisions`) Tier Accuracy**: **100.0%** (60/60) raw choice | **100.0%** (60/60) effective routed model | Avg latency: **102.7 ms** | Router cost: **$0.00** (local)
- **Google Gemini Decisions API (`gemini-decision-1`) Tier Accuracy**: **100.0%** (60/60) raw choice | **100.0%** (60/60) effective routed model | Avg latency: **606.9 ms** | Router cost: **$0.002394** (23,941 input tokens, 0 output tokens)
- **Anchor Calibration Note**: The routing criteria in [`laya_adk_model_router/questions.py`](../laya_adk_model_router/questions.py) and [`decisions_adk_model_router/questions.py`](../decisions_adk_model_router/questions.py) use identical `simple` and `complex` domain criteria covering `QA`, `Math`, and `Coding`. When deploying to new domain distributions, we recommend tuning the `simple` and `complex` criteria (or `confidence_threshold`) and benchmarking against alternative semantic/embedding routers such as [`jev`](https://github.com/dmoonat/jev).

| Dataset | Specialist Agent | Ground-Truth Tier | Samples | Laya (`S`/`C`) | Laya Avg Conf (Min–Max) | Laya Latency | Decisions API (`S`/`C`) | Decisions Avg Conf (Min–Max) | Decisions Latency | Routed Specialist Model (`Laya` / `Decisions`) |
| :--- | :--- | :--- | ---: | :---: | :---: | ---: | :---: | :---: | ---: | :--- |
| **`ARC-Easy`** | `fast_qa_agent` | `simple` | 10 | 10/0 | 0.77 (0.68–0.82) | 96.6 ms | 10/0 | 0.9899 (0.9799–0.9966) | 640.2 ms | `gemini-3.1-flash-lite` / `gemini-3.1-flash-lite` |
| **`MMLU-Pro`** | `fast_qa_agent` | `complex` | 10 | 0/10 | 0.77 (0.72–0.82) | 138.6 ms | 0/10 | 0.9754 (0.9531–0.9939) | 650.3 ms | `gemini-3.1-pro-preview` / `gemini-3.1-pro-preview` |
| **`GSM8K`** | `math_analyst_agent` | `simple` | 10 | 10/0 | 0.69 (0.61–0.73) | 82.0 ms | 10/0 | 0.9684 (0.9079–0.9946) | 500.7 ms | `gemini-3.1-flash-lite` / `gemini-3.1-flash-lite` |
| **`MATH-500`** | `math_analyst_agent` | `complex` | 10 | 0/10 | 0.75 (0.65–0.83) | 93.3 ms | 0/10 | 0.9960 (0.9917–0.9991) | 473.1 ms | `gemini-3.1-pro-preview` / `gemini-3.1-pro-preview` |
| **`MBPP`** | `code_engineer_agent` | `simple` | 10 | 10/0 | 0.66 (0.62–0.71) | 85.1 ms | 10/0 | 0.9602 (0.9249–0.9830) | 761.1 ms | `gemini-3.1-flash-lite` / `gemini-3.1-flash-lite` |
| **`HumanEval`** | `code_engineer_agent` | `complex` | 10 | 0/10 | 0.61 (0.56–0.65) | 120.5 ms | 0/10 | 0.9909 (0.9840–0.9962) | 615.9 ms | `gemini-3.1-pro-preview` / `gemini-3.1-pro-preview` |

---

## 3. Per-Dataset Breakdown across All 5 Arms (2x3 Domain x Complexity)

| Benchmark (Tier) | Arm / Mode | Coordinator $\rightarrow$ Specialist Model | Accuracy (%) | Total Tokens (In / Out / Think) | Avg Turn Latency (ms) | Total Cost (USD) | Cost vs. All-Complex | Cost vs. Static Hybrid |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ARC-Easy** (`simple`) | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 14,279 (13,714 / 565 / 0) | 4306.9 ms | $0.004276 | -94.4% | -90.7% |
| **ARC-Easy** (`simple`) | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 17,784 (13,714 / 558 / 3,512) | 10878.0 ms | $0.076268 | — | +65.7% |
| **ARC-Easy** (`simple`) | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 16,302 (13,714 / 571 / 2,017) | 9677.8 ms | $0.046038 | -39.6% | — |
| **ARC-Easy** (`simple`) | `with_router` (Laya) | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 14,266 (13,714 / 552 / 0) | 4307.2 ms | $0.004256 | **-94.4%** | **-90.8%** |
| **ARC-Easy** (`simple`) | `with_decisions_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 17,457 (16,872 / 585 / 0) | 5755.0 ms | $0.004646 | **-93.9%** | **-89.9%** |
| **MMLU-Pro** (`complex`) | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 80.0% (8/10) | 24,174 (19,936 / 4,238 / 0) | 5258.1 ms | $0.011341 | -95.5% | -95.4% |
| **MMLU-Pro** (`complex`) | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 37,498 (19,936 / 3,745 / 13,817) | 23580.4 ms | $0.250616 | — | +1.2% |
| **MMLU-Pro** (`complex`) | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 38,732 (19,936 / 3,709 / 15,087) | 22426.0 ms | $0.247534 | -1.2% | — |
| **MMLU-Pro** (`complex`) | `with_router` (Laya) | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 37,950 (19,936 / 3,851 / 14,163) | 22733.1 ms | $0.238150 | **-5.0%** | **-3.8%** |
| **MMLU-Pro** (`complex`) | `with_decisions_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 44,249 (25,929 / 3,771 / 14,549) | 22703.8 ms | $0.242421 | **-3.3%** | **-2.1%** |
| **GSM8K** (`simple`) | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 16,316 (13,876 / 2,440 / 0) | 4191.5 ms | $0.007129 | -93.2% | -91.6% |
| **GSM8K** (`simple`) | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 20,305 (13,876 / 2,076 / 4,353) | 17758.7 ms | $0.104900 | — | +23.6% |
| **GSM8K** (`simple`) | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 19,692 (13,876 / 2,115 / 3,701) | 9219.5 ms | $0.084895 | -19.1% | — |
| **GSM8K** (`simple`) | `with_router` (Laya) | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 16,360 (13,876 / 2,484 / 0) | 4294.0 ms | $0.007195 | **-93.1%** | **-91.5%** |
| **GSM8K** (`simple`) | `with_decisions_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 19,264 (16,824 / 2,440 / 0) | 4778.0 ms | $0.007424 | **-92.9%** | **-91.3%** |
| **MATH-500** (`complex`) | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 70.0% (7/10) | 22,577 (15,206 / 7,371 / 0) | 5786.3 ms | $0.014858 | -97.3% | -96.7% |
| **MATH-500** (`complex`) | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 58,635 (15,334 / 7,657 / 35,644) | 47334.9 ms | $0.550280 | — | +23.6% |
| **MATH-500** (`complex`) | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 51,032 (15,334 / 7,171 / 28,527) | 39164.0 ms | $0.445119 | -19.1% | — |
| **MATH-500** (`complex`) | `with_router` (Laya) | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 54,796 (15,334 / 7,654 / 31,808) | 39963.5 ms | $0.490287 | **-10.9%** | **+10.1%** |
| **MATH-500** (`complex`) | `with_decisions_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 51,875 (19,103 / 7,233 / 25,539) | 34049.6 ms | $0.410417 | **-25.4%** | **-7.8%** |
| **MBPP** (`simple`) | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 14,916 (13,844 / 1,072 / 0) | 3796.8 ms | $0.005069 | -96.2% | -95.8% |
| **MBPP** (`simple`) | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 22,550 (13,844 / 681 / 8,025) | 13771.1 ms | $0.132160 | — | +10.4% |
| **MBPP** (`simple`) | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 22,565 (13,844 / 662 / 8,059) | 25150.2 ms | $0.119754 | -9.4% | — |
| **MBPP** (`simple`) | `with_router` (Laya) | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 14,946 (13,844 / 1,102 / 0) | 3857.3 ms | $0.005114 | **-96.1%** | **-95.7%** |
| **MBPP** (`simple`) | `with_decisions_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (10/10) | 17,898 (16,839 / 1,059 / 0) | 4681.9 ms | $0.005349 | **-96.0%** | **-95.5%** |
| **HumanEval** (`complex`) | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 70.0% (7/10) | 21,877 (17,732 / 4,145 / 0) | 4837.2 ms | $0.010651 | -97.5% | -96.6% |
| **HumanEval** (`complex`) | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 50,613 (17,796 / 5,426 / 27,391) | 43390.5 ms | $0.429396 | — | +35.6% |
| **HumanEval** (`complex`) | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 42,556 (17,796 / 4,523 / 20,237) | 31349.1 ms | $0.316682 | -26.2% | — |
| **HumanEval** (`complex`) | `with_router` (Laya) | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 47,366 (17,796 / 5,578 / 23,992) | 31328.0 ms | $0.374402 | **-12.8%** | **+18.2%** |
| **HumanEval** (`complex`) | `with_decisions_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (10/10) | 54,718 (23,082 / 5,674 / 25,962) | 34484.3 ms | $0.399725 | **-6.9%** | **+26.2%** |

---

## 4. Multi-Agent Cost & Token Attribution (`adk-finops` `breakdown_by_agent`)

| Agent Name | All-Simple Cost (Tok) | All-Complex Cost (Tok) | Static Hybrid Cost (Tok) | With-Laya Cost (Tok) | With-Decisions Cost (Tok) | Laya vs. All-Complex | Decisions vs. All-Complex |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`code_engineer_agent`** | $0.011648 (22,905) | $0.511224 (57,742) | $0.432348 (51,169) | $0.375428 (48,360) | $0.400155 (50,383) | **-26.6%** | **-21.7%** |
| **`coordinator_agent`** | $0.012187 (41,444) | $0.164512 (47,136) | $0.012218 (41,572) | $0.012218 (41,572) | $0.012246 (41,684) | **-92.6%** | **-92.6%** |
| **`fast_qa_agent`** | $0.011283 (23,518) | $0.261062 (37,751) | $0.289238 (40,099) | $0.238072 (37,281) | $0.241814 (37,700) | **-8.8%** | **-7.4%** |
| **`math_analyst_agent`** | $0.018207 (26,272) | $0.606822 (64,756) | $0.526218 (58,039) | $0.493686 (58,471) | $0.413372 (51,753) | **-18.6%** | **-31.9%** |

---

## 5. Model Distribution (`adk-finops` `breakdown_by_model`)

| Arm / Mode | Model | LLM Calls | Prompt Tokens | Completion Tokens | Thinking Tokens | Total Tokens | Total Cost (USD) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_simple` | **`gemini-3.1-flash-lite`** | 120 | 94,308 | 19,831 | 0 | 114,139 | $0.053324 |
| `all_complex` | **`gemini-3.1-pro-preview`** | 120 | 94,500 | 20,143 | 92,742 | 207,385 | $1.543620 |
| `coord_simple_subs_complex` | **`gemini-3.1-flash-lite`** | 60 | 40,112 | 1,460 | 0 | 41,572 | $0.012218 |
| `coord_simple_subs_complex` | **`gemini-3.1-pro-preview`** | 60 | 54,388 | 17,291 | 77,628 | 149,307 | $1.247804 |
| `with_router` | **`gemini-3.1-flash-lite`** | 90 | 64,394 | 4,868 | 0 | 69,262 | $0.023401 |
| `with_router` | **`gemini-3.1-pro-preview`** | 30 | 30,106 | 16,353 | 69,963 | 116,422 | $1.096004 |
| `with_decisions_router` | **`gemini-3.1-flash-lite`** | 90 | 64,586 | 4,814 | 0 | 69,400 | $0.023368 |
| `with_decisions_router` | **`gemini-3.1-pro-preview`** | 30 | 30,122 | 15,948 | 66,050 | 112,120 | $1.044220 |
| `with_decisions_router` | **`gemini-decision-1`** | 60 | 23,941 | 0 | 0 | 23,941 | $0.002394 |

---

## 6. Sample-Level Comparison Log (All 60 Tasks across All 5 Arms)

| Sample ID | Dataset (Tier) | Laya Choice (Conf / Lat) | Decisions Choice (Conf / Lat) | Latency (`Simple` / `Complex` / `Hybrid` / `Laya` / `Decisions`) | Cost (`Simple` / `Complex` / `Hybrid` / `Laya` / `Decisions`) | Correct (`S`/`C`/`H`/`L`/`D`) |
| :--- | :--- | :--- | :--- | ---: | ---: | :---: |
| `arc_easy_Mercury_417466` | ARC-Easy (`simple`) | `simple` (0.72 / 360ms) | `simple` (0.980 / 794ms) | 4908 / 18346 / 11633 / **4498** / **6064 ms** | $0.0004 / $0.0083 / $0.0066 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `arc_easy_Mercury_7081673` | ARC-Easy (`simple`) | `simple` (0.82 / 66ms) | `simple` (0.993 / 981ms) | 4983 / 9548 / 6472 / **4454** / **5889 ms** | $0.0004 / $0.0062 / $0.0040 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `arc_easy_Mercury_7239733` | ARC-Easy (`simple`) | `simple` (0.80 / 61ms) | `simple` (0.985 / 855ms) | 4857 / 10255 / 8371 / **4898** / **6052 ms** | $0.0004 / $0.0086 / $0.0052 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `arc_easy_NYSEDREGENTS_2015_4_8` | ARC-Easy (`simple`) | `simple` (0.82 / 59ms) | `simple` (0.997 / 930ms) | 5305 / 9419 / 7005 / **4588** / **5903 ms** | $0.0004 / $0.0067 / $0.0043 / **$0.0004** / **$0.0004** | ✅/✅/✅/✅/✅ |
| `arc_easy_Mercury_7037258` | ARC-Easy (`simple`) | `simple` (0.68 / 66ms) | `simple` (0.987 / 882ms) | 5109 / 12893 / 27563 / **4299** / **5993 ms** | $0.0004 / $0.0099 / $0.0035 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `arc_easy_CSZ20679` | ARC-Easy (`simple`) | `simple` (0.79 / 100ms) | `simple` (0.994 / 343ms) | 3806 / 7544 / 5979 / **4376** / **5654 ms** | $0.0004 / $0.0054 / $0.0032 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `arc_easy_Mercury_182158` | ARC-Easy (`simple`) | `simple` (0.77 / 64ms) | `simple` (0.986 / 528ms) | 3462 / 11012 / 7793 / **3997** / **5799 ms** | $0.0004 / $0.0074 / $0.0037 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `arc_easy_Mercury_7216668` | ARC-Easy (`simple`) | `simple` (0.79 / 61ms) | `simple` (0.992 / 356ms) | 3622 / 9961 / 6737 / **3774** / **5157 ms** | $0.0004 / $0.0077 / $0.0043 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `arc_easy_MCAS_2001_5_19` | ARC-Easy (`simple`) | `simple` (0.82 / 61ms) | `simple` (0.996 / 330ms) | 3437 / 8922 / 7197 / **4032** / **5836 ms** | $0.0004 / $0.0067 / $0.0048 / **$0.0004** / **$0.0004** | ✅/✅/✅/✅/✅ |
| `arc_easy_Mercury_SC_413631` | ARC-Easy (`simple`) | `simple` (0.70 / 69ms) | `simple` (0.990 / 404ms) | 3581 / 10880 / 8027 / **4155** / **5204 ms** | $0.0004 / $0.0094 / $0.0063 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `mmlu_pro_1592` | MMLU-Pro (`complex`) | `complex` (0.78 / 151ms) | `complex` (0.994 / 556ms) | 4967 / 19714 / 15017 / **15705** / **17996 ms** | $0.0012 / $0.0200 / $0.0168 / **$0.0148** / **$0.0178** | ✅/✅/✅/✅/✅ |
| `mmlu_pro_1595` | MMLU-Pro (`complex`) | `complex` (0.73 / 113ms) | `complex` (0.990 / 875ms) | 5216 / 18376 / 13663 / **15213** / **16432 ms** | $0.0012 / $0.0192 / $0.0139 / **$0.0146** / **$0.0145** | ✅/✅/✅/✅/✅ |
| `mmlu_pro_1598` | MMLU-Pro (`complex`) | `complex` (0.75 / 102ms) | `complex` (0.983 / 694ms) | 5313 / 25425 / 21275 / **22495** / **24698 ms** | $0.0011 / $0.0252 / $0.0248 / **$0.0238** / **$0.0297** | ✅/✅/✅/✅/✅ |
| `mmlu_pro_1601` | MMLU-Pro (`complex`) | `complex` (0.72 / 97ms) | `complex` (0.960 / 372ms) | 5157 / 22828 / 18632 / **15951** / **18319 ms** | $0.0011 / $0.0217 / $0.0205 / **$0.0157** / **$0.0156** | ✅/✅/✅/✅/✅ |
| `mmlu_pro_1602` | MMLU-Pro (`complex`) | `complex` (0.82 / 109ms) | `complex` (0.966 / 421ms) | 5263 / 17435 / 19497 / **19140** / **15446 ms** | $0.0011 / $0.0196 / $0.0201 / **$0.0204** / **$0.0172** | ✅/✅/✅/✅/✅ |
| `mmlu_pro_1606` | MMLU-Pro (`complex`) | `complex` (0.77 / 107ms) | `complex` (0.979 / 930ms) | 5225 / 31076 / 52561 / **39640** / **24661 ms** | $0.0010 / $0.0327 / $0.0640 / **$0.0457** / **$0.0276** | ❌/✅/✅/✅/✅ |
| `mmlu_pro_1607` | MMLU-Pro (`complex`) | `complex` (0.73 / 158ms) | `complex` (0.953 / 819ms) | 5336 / 37490 / 35039 / **48488** / **53572 ms** | $0.0011 / $0.0430 / $0.0401 / **$0.0529** / **$0.0651** | ❌/✅/✅/✅/✅ |
| `mmlu_pro_1610` | MMLU-Pro (`complex`) | `complex` (0.79 / 146ms) | `complex` (0.957 / 654ms) | 5482 / 25740 / 16344 / **15881** / **20538 ms** | $0.0012 / $0.0293 / $0.0162 / **$0.0199** / **$0.0202** | ✅/✅/✅/✅/✅ |
| `mmlu_pro_1613` | MMLU-Pro (`complex`) | `complex` (0.82 / 261ms) | `complex` (0.994 / 537ms) | 5362 / 24686 / 18053 / **23450** / **22824 ms** | $0.0012 / $0.0271 / $0.0199 / **$0.0193** / **$0.0242** | ✅/✅/✅/✅/✅ |
| `mmlu_pro_1614` | MMLU-Pro (`complex`) | `complex` (0.80 / 142ms) | `complex` (0.979 / 644ms) | 5258 / 13034 / 14179 / **11367** / **12553 ms** | $0.0011 / $0.0128 / $0.0111 / **$0.0110** / **$0.0105** | ✅/✅/✅/✅/✅ |
| `gsm8k_0` | GSM8K (`simple`) | `simple` (0.73 / 97ms) | `simple` (0.990 / 529ms) | 4288 / 10308 / 7054 / **3870** / **4694 ms** | $0.0006 / $0.0076 / $0.0056 / **$0.0006** / **$0.0007** | ✅/✅/✅/✅/✅ |
| `gsm8k_1` | GSM8K (`simple`) | `simple` (0.70 / 111ms) | `simple` (0.994 / 350ms) | 4024 / 28266 / 6020 / **3739** / **4082 ms** | $0.0005 / $0.0065 / $0.0040 / **$0.0005** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `gsm8k_2` | GSM8K (`simple`) | `simple` (0.64 / 87ms) | `simple` (0.955 / 365ms) | 4464 / 53335 / 13190 / **4355** / **4436 ms** | $0.0008 / $0.0196 / $0.0179 / **$0.0008** / **$0.0008** | ✅/✅/✅/✅/✅ |
| `gsm8k_3` | GSM8K (`simple`) | `simple` (0.72 / 58ms) | `simple` (0.995 / 971ms) | 3911 / 7962 / 6403 / **3673** / **6557 ms** | $0.0006 / $0.0060 / $0.0043 / **$0.0006** / **$0.0006** | ✅/✅/✅/✅/✅ |
| `gsm8k_4` | GSM8K (`simple`) | `simple` (0.68 / 104ms) | `simple` (0.965 / 657ms) | 4457 / 13453 / 9150 / **4897** / **4594 ms** | $0.0007 / $0.0100 / $0.0079 / **$0.0007** / **$0.0008** | ✅/✅/✅/✅/✅ |
| `gsm8k_5` | GSM8K (`simple`) | `simple` (0.69 / 65ms) | `simple` (0.983 / 517ms) | 4018 / 12505 / 9778 / **4594** / **4965 ms** | $0.0007 / $0.0103 / $0.0077 / **$0.0008** / **$0.0008** | ✅/✅/✅/✅/✅ |
| `gsm8k_6` | GSM8K (`simple`) | `simple` (0.69 / 71ms) | `simple` (0.986 / 373ms) | 4136 / 11150 / 8287 / **4833** / **4412 ms** | $0.0006 / $0.0078 / $0.0063 / **$0.0007** / **$0.0007** | ✅/✅/✅/✅/✅ |
| `gsm8k_7` | GSM8K (`simple`) | `simple` (0.69 / 85ms) | `simple` (0.945 / 532ms) | 4330 / 16254 / 13100 / **5077** / **5189 ms** | $0.0009 / $0.0146 / $0.0122 / **$0.0009** / **$0.0009** | ✅/✅/✅/✅/✅ |
| `gsm8k_8` | GSM8K (`simple`) | `simple` (0.61 / 70ms) | `simple` (0.908 / 344ms) | 3929 / 14338 / 11589 / **3988** / **4676 ms** | $0.0010 / $0.0133 / $0.0118 / **$0.0010** / **$0.0010** | ✅/✅/✅/✅/✅ |
| `gsm8k_9` | GSM8K (`simple`) | `simple` (0.72 / 72ms) | `simple` (0.963 / 369ms) | 4357 / 10016 / 7625 / **3914** / **4175 ms** | $0.0007 / $0.0092 / $0.0072 / **$0.0007** / **$0.0007** | ✅/✅/✅/✅/✅ |
| `math500_test_prealgebra_1139` | MATH-500 (`complex`) | `complex` (0.77 / 78ms) | `complex` (0.993 / 344ms) | 6724 / 161374 / 157826 / **106539** / **110740 ms** | $0.0019 / $0.2031 / $0.1720 / **$0.1398** / **$0.1398** | ❌/✅/✅/✅/✅ |
| `math500_test_number_theory_737` | MATH-500 (`complex`) | `complex` (0.69 / 99ms) | `complex` (0.997 / 596ms) | 5487 / 23176 / 22748 / **16860** / **22297 ms** | $0.0015 / $0.0267 / $0.0262 / **$0.0246** / **$0.0277** | ✅/✅/✅/✅/✅ |
| `math500_test_algebra_1837` | MATH-500 (`complex`) | `complex` (0.73 / 73ms) | `complex` (0.996 / 358ms) | 6080 / 23599 / 22062 / **21614** / **18063 ms** | $0.0014 / $0.0276 / $0.0250 / **$0.0252** / **$0.0220** | ✅/✅/✅/✅/✅ |
| `math500_test_algebra_2427` | MATH-500 (`complex`) | `complex` (0.65 / 69ms) | `complex` (0.992 / 534ms) | 5590 / 20615 / 21657 / **19043** / **18312 ms** | $0.0014 / $0.0228 / $0.0247 / **$0.0206** / **$0.0213** | ✅/✅/✅/✅/✅ |
| `math500_test_counting_and_probability_525` | MATH-500 (`complex`) | `complex` (0.74 / 59ms) | `complex` (0.998 / 521ms) | 5288 / 39440 / 28740 / **82588** / **31464 ms** | $0.0012 / $0.0571 / $0.0323 / **$0.1014** / **$0.0330** | ✅/✅/✅/✅/✅ |
| `math500_test_prealgebra_1558` | MATH-500 (`complex`) | `complex` (0.69 / 123ms) | `complex` (0.996 / 382ms) | 5534 / 14706 / 14630 / **15133** / **13809 ms** | $0.0011 / $0.0186 / $0.0152 / **$0.0159** / **$0.0139** | ✅/✅/✅/✅/✅ |
| `math500_test_algebra_305` | MATH-500 (`complex`) | `complex` (0.80 / 97ms) | `complex` (0.999 / 501ms) | 5136 / 20890 / 23425 / **15362** / **16399 ms** | $0.0013 / $0.0185 / $0.0162 / **$0.0153** / **$0.0177** | ✅/✅/✅/✅/✅ |
| `math500_test_number_theory_1055` | MATH-500 (`complex`) | `complex` (0.77 / 102ms) | `complex` (0.992 / 534ms) | 5450 / 40738 / 37501 / **42494** / **43365 ms** | $0.0014 / $0.0475 / $0.0606 / **$0.0497** / **$0.0535** | ✅/✅/✅/✅/✅ |
| `math500_test_algebra_892` | MATH-500 (`complex`) | `complex` (0.83 / 111ms) | `complex` (0.998 / 407ms) | 6202 / 79142 / 29181 / **43841** / **33954 ms** | $0.0017 / $0.0679 / $0.0334 / **$0.0538** / **$0.0424** | ❌/✅/✅/✅/✅ |
| `math500_test_prealgebra_1646` | MATH-500 (`complex`) | `complex` (0.80 / 121ms) | `complex` (0.999 / 555ms) | 6373 / 49669 / 33871 / **36160** / **32092 ms** | $0.0020 / $0.0605 / $0.0395 / **$0.0439** / **$0.0392** | ❌/✅/✅/✅/✅ |
| `mbpp_11` | MBPP (`simple`) | `simple` (0.65 / 91ms) | `simple` (0.939 / 720ms) | 4097 / 16151 / 89063 / **4650** / **4965 ms** | $0.0006 / $0.0185 / $0.0308 / **$0.0006** / **$0.0006** | ✅/✅/✅/✅/✅ |
| `mbpp_12` | MBPP (`simple`) | `simple` (0.71 / 101ms) | `simple` (0.953 / 366ms) | 3819 / 9909 / 17927 / **3286** / **4814 ms** | $0.0005 / $0.0090 / $0.0061 / **$0.0005** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `mbpp_14` | MBPP (`simple`) | `simple` (0.62 / 90ms) | `simple` (0.967 / 1091ms) | 3886 / 15710 / 22295 / **3671** / **5105 ms** | $0.0006 / $0.0145 / $0.0093 / **$0.0005** / **$0.0006** | ✅/✅/✅/✅/✅ |
| `mbpp_17` | MBPP (`simple`) | `simple` (0.64 / 55ms) | `simple` (0.983 / 386ms) | 3402 / 9526 / 5845 / **3249** / **3541 ms** | $0.0004 / $0.0063 / $0.0031 / **$0.0004** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `mbpp_18` | MBPP (`simple`) | `simple` (0.69 / 88ms) | `simple` (0.971 / 1229ms) | 3889 / 9583 / 12087 / **3981** / **5335 ms** | $0.0005 / $0.0087 / $0.0072 / **$0.0005** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `mbpp_19` | MBPP (`simple`) | `simple` (0.65 / 58ms) | `simple` (0.972 / 1309ms) | 3866 / 8274 / 11137 / **3680** / **4855 ms** | $0.0005 / $0.0068 / $0.0049 / **$0.0005** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `mbpp_20` | MBPP (`simple`) | `simple` (0.68 / 87ms) | `simple` (0.925 / 1065ms) | 3663 / 21780 / 29411 / **4399** / **5026 ms** | $0.0006 / $0.0215 / $0.0174 / **$0.0007** / **$0.0007** | ✅/✅/✅/✅/✅ |
| `mbpp_56` | MBPP (`simple`) | `simple` (0.63 / 95ms) | `simple` (0.954 / 541ms) | 3700 / 17215 / 23964 / **3531** / **4617 ms** | $0.0005 / $0.0178 / $0.0125 / **$0.0005** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `mbpp_57` | MBPP (`simple`) | `simple` (0.64 / 68ms) | `simple` (0.964 / 464ms) | 3915 / 12741 / 21625 / **3883** / **4154 ms** | $0.0005 / $0.0121 / $0.0102 / **$0.0005** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `mbpp_58` | MBPP (`simple`) | `simple` (0.65 / 119ms) | `simple` (0.974 / 440ms) | 3730 / 16820 / 18148 / **4244** / **4408 ms** | $0.0005 / $0.0170 / $0.0181 / **$0.0005** / **$0.0005** | ✅/✅/✅/✅/✅ |
| `humaneval_120` | HumanEval (`complex`) | `complex` (0.60 / 130ms) | `complex` (0.996 / 1376ms) | 4192 / 23944 / 60356 / **31090** / **31669 ms** | $0.0009 / $0.0279 / $0.0285 / **$0.0366** / **$0.0359** | ✅/✅/✅/✅/✅ |
| `humaneval_121` | HumanEval (`complex`) | `complex` (0.63 / 86ms) | `complex` (0.987 / 681ms) | 4761 / 11856 / 12026 / **11574** / **14202 ms** | $0.0010 / $0.0119 / $0.0118 / **$0.0116** / **$0.0142** | ✅/✅/✅/✅/✅ |
| `humaneval_122` | HumanEval (`complex`) | `complex` (0.60 / 76ms) | `complex` (0.990 / 1144ms) | 4440 / 34111 / 20120 / **24536** / **25704 ms** | $0.0009 / $0.0365 / $0.0173 / **$0.0263** / **$0.0289** | ✅/✅/✅/✅/✅ |
| `humaneval_123` | HumanEval (`complex`) | `complex` (0.56 / 128ms) | `complex` (0.984 / 636ms) | 4245 / 29107 / 31255 / **27884** / **41901 ms** | $0.0010 / $0.0326 / $0.0353 / **$0.0353** / **$0.0297** | ✅/✅/✅/✅/✅ |
| `humaneval_124` | HumanEval (`complex`) | `complex` (0.60 / 143ms) | `complex` (0.989 / 363ms) | 5197 / 49522 / 40732 / **49082** / **35998 ms** | $0.0012 / $0.0423 / $0.0489 / **$0.0596** / **$0.0458** | ✅/✅/✅/✅/✅ |
| `humaneval_125` | HumanEval (`complex`) | `complex` (0.57 / 113ms) | `complex` (0.992 / 360ms) | 5016 / 52086 / 38271 / **43759** / **35993 ms** | $0.0011 / $0.0368 / $0.0438 / **$0.0514** / **$0.0445** | ✅/✅/✅/✅/✅ |
| `humaneval_126` | HumanEval (`complex`) | `complex` (0.58 / 128ms) | `complex` (0.991 / 346ms) | 4864 / 41524 / 23241 / **22268** / **52185 ms** | $0.0009 / $0.0251 / $0.0312 / **$0.0239** / **$0.0635** | ❌/✅/✅/✅/✅ |
| `humaneval_127` | HumanEval (`complex`) | `complex` (0.65 / 130ms) | `complex` (0.994 / 356ms) | 4866 / 58401 / 30579 / **23767** / **25787 ms** | $0.0011 / $0.0473 / $0.0369 / **$0.0361** / **$0.0332** | ✅/✅/✅/✅/✅ |
| `humaneval_130` | HumanEval (`complex`) | `complex` (0.63 / 135ms) | `complex` (0.990 / 388ms) | 6287 / 36916 / 30389 / **34506** / **31979 ms** | $0.0017 / $0.0448 / $0.0347 / **$0.0404** / **$0.0403** | ❌/✅/✅/✅/✅ |
| `humaneval_132` | HumanEval (`complex`) | `complex` (0.63 / 135ms) | `complex` (0.996 / 509ms) | 4504 / 96438 / 26524 / **44814** / **49424 ms** | $0.0010 / $0.1241 / $0.0282 / **$0.0533** / **$0.0637** | ❌/✅/✅/✅/✅ |

---

## 7. Key Findings & Architectural Takeaways

1. **Every Specialist Agent Faces Mixed Complexity (`10 Simple + 10 Complex` per Domain)**: Because `fast_qa_agent` (`ARC-Easy` + `MMLU-Pro`), `math_analyst_agent` (`GSM8K` + `MATH-500`), and `code_engineer_agent` (`MBPP` + `HumanEval`) each handle both simple and complex tasks, no static per-agent model assignment can optimize both cost and accuracy.
2. **Why Dynamic Routing (`Laya` & `Gemini Decision 1`) Beats `All-Simple` (`gemini-3.1-flash-lite`)**: `all_simple` (`gemini-3.1-flash-lite`) drops to **86.7% (52/60)** overall accuracy on the complex tasks (`80.0%` on `MMLU-Pro`, `70.0%` on `MATH-500`, `70.0%` on `HumanEval`), whereas `with_router` achieves **100.0% (60/60)** and `with_decisions_router` achieves **100.0% (60/60)**.
3. **Head-to-Head: Local `Laya` (`laya-typed-decisions`) vs. Cloud `Gemini Decision 1` (`gemini-decision-1`)**:
   - **Classification Accuracy & Confidence Calibration**: Laya achieved **100.0% (60/60)** tier accuracy (confidence range `0.56–0.88`), while `gemini-decision-1` achieved **100.0% (60/60)** tier accuracy with sharper calibrated probabilities (`0.95–1.00` confidence).
   - **Router Latency (`102.7 ms` vs. `606.9 ms`)**: Laya runs locally in-process on CPU/MPS (`102.7 ms` avg), avoiding network round-trip overhead compared to the cloud `v1beta/decisions` call (`606.9 ms` avg).
   - **Router Cost (`$0.00` vs. `$0.002394`)**: Both generate **0 output tokens**; Laya incurs `$0.00` API cost locally, while `gemini-decision-1` cost just **$0.002394** across all 60 tasks (`23,941` input tokens at `$0.10 / 1M`) while offering a 262K multimodal context window and zero local model weights.
4. **Decomposing Coordinator Savings vs. Specialist Routing Savings (`temperature=0.0`)**:
   - **Static Hybrid (`coord_simple_subs_complex`) vs. All-Complex (`all_complex`)**: Fixing `coordinator_agent` to `gemini-3.1-flash-lite` while keeping all specialist sub-agents on `gemini-3.1-pro-preview` changed total spend by **-18.4%** (`$1.543620` $\rightarrow$ `$1.260022`) and turn latency by **-12.6%**.
   - **With Laya Router (`with_router`) vs. All-Complex (`all_complex`)**: Saved **-27.5%** in total cost (`$1.543620` $\rightarrow$ `$1.119405`), **-94.7%** on simple-tier tasks (`$0.313328` $\rightarrow$ `$0.016566`), and **-32.1%** in average turn latency.
   - **With Decisions API Router (`with_decisions_router`) vs. All-Complex (`all_complex`)**: Saved **-30.7%** in total cost (`$1.543620` $\rightarrow$ `$1.069982`), **-94.4%** on simple-tier tasks (`$0.313328` $\rightarrow$ `$0.017419`), and **-32.1%** in average turn latency.
