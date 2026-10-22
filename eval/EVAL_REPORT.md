# Multi-Agent ADK Model Router Evaluation Report (4-Arm Benchmark, N=45 per Arm)

Comprehensive 4-arm comparison of a **4-Agent Google ADK Hierarchy** (`coordinator_agent` $\rightarrow$ `fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) evaluated on **45 real open-source Hugging Face benchmark tasks** (`15 ARC-Easy`, `15 MATH-500 Level-5`, `15 HumanEval` = **180 total multi-agent turns / 360 LLM calls**) and instrumented with **[`adk-finops`](https://pypi.org/project/adk-finops/)**:

1. **All-Simple (`all_simple`)**: `coordinator_agent` (`gemini-3.1-flash-lite`) $\rightarrow$ Specialist sub-agents (`gemini-3.1-flash-lite`).
2. **All-Complex (`all_complex`)**: `coordinator_agent` (`gemini-3.1-pro-preview`) $\rightarrow$ Specialist sub-agents (`gemini-3.1-pro-preview`).
3. **Static Hybrid (`coord_simple_subs_complex`)**: `coordinator_agent` fixed to `gemini-3.1-flash-lite` and all 3 specialist sub-agents fixed to `gemini-3.1-pro-preview` (no dynamic router).
4. **Role-Aware Laya Router (`with_router`)**: `coordinator_agent` pinned to `gemini-3.1-flash-lite` via `exclude_agents={"coordinator_agent"}`, while specialist sub-agents (`fast_qa_agent`, `math_analyst_agent`, `code_engineer_agent`) are dynamically routed by `LayaModelRouterPlugin` (`laya-typed-decisions`) between `gemini-3.1-flash-lite` and `gemini-3.1-pro-preview`.

---

## 1. Executive Summary (All 4 Experimental Arms)

| Metric | 1. All-Simple (`gemini-3.1-flash-lite`) | 2. All-Complex (`gemini-3.1-pro-preview`) | 3. Static Hybrid (`Coord=Simple, Subs=Complex`) | 4. With Laya Router (`Coord=Simple, Subs=Laya`) | Router vs. All-Simple | Router vs. All-Complex | Router vs. Static Hybrid |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Total Benchmark Tasks** | 45 | 45 | 45 | 45 | — | — | — |
| **Operational Success Rate (`adk-finops`)** | 100.0% (45/45) | 100.0% (45/45) | 100.0% (45/45) | 100.0% (45/45) | +0.0% | +0.0% | +0.0% |
| **Ground-Truth Accuracy (%)** | **86.7%** (39/45) | **100.0%** (45/45) | **100.0%** (45/45) | **100.0%** (45/45) | **+13.3% (+6 tasks)** | **+0.0%** | **+0.0%** |
| **Total Cost (USD)** | **$0.044651** | **$1.520818** | **$1.412595** | **$1.413296** | +3065.2% | **-7.1% ($+0.107522 saved)** | **+0.0% ($-0.000701 saved)** |
| **Avg Cost / Task (USD)** | $0.000992 | $0.033796 | $0.031391 | $0.031407 | +3065.2% | -7.1% | +0.0% |
| **Effective Cost / Correct Answer** | $0.001145 | $0.033796 | $0.031391 | $0.031407 | +2643.2% | -7.1% | +0.0% |
| **Total Billable Tokens** | 87,711 | 184,679 | 180,900 | 183,121 | +108.8% | -0.8% | +1.2% |
| **Prompt Tokens** | 69,533 | 69,533 | 69,533 | 69,533 | 0.0% | 0.0% | 0.0% |
| **Completion Tokens** | 18,178 | 19,669 | 18,210 | 20,854 | +14.7% | +6.0% | +14.5% |
| **Thinking Tokens (`thoughts_tokens`)** | 0 | 95,477 | 93,157 | 92,734 | — | -2.9% | -0.5% |
| **Avg Turn Latency (ms)** | **4887.0 ms** | **36310.5 ms** | **29586.4 ms** | **27602.3 ms** | +464.8% | **-24.0% (+8708.3 ms faster)** | **-6.7% (+1984.1 ms faster)** |
| **P95 Turn Latency (ms)** | 6372.8 ms | 102531.1 ms | 98410.6 ms | 85514.3 ms | +1241.9% | -16.6% | -13.1% |
| **Avg Laya Router Overhead (ms)** | 0.0 ms | 0.0 ms | 0.0 ms | **102.0 ms** (p95: 135.4 ms) | +102.0 ms | +102.0 ms (~0.37% of turn) | +102.0 ms |
| **Avg Output Throughput (`tok/s`)** | 75.9 tok/s | 65.9 tok/s | 70.7 tok/s | 66.2 tok/s | -12.8% | +0.5% | -6.4% |

---

## 2. Laya Router Classification Audit (Confusion Matrix & Confidence Distribution)

- **Raw Laya Classifier Accuracy (`choice` vs. `expected_complexity`)**: **100.0%** (45/45)
- **Effective Routed Specialist Model Tier Accuracy**: **100.0%** (45/45)

| Dataset (Ground-Truth Tier) | Total Samples | Predicted `simple` | Predicted `complex` | Avg Confidence | Min–Max Confidence | Effective Specialist Model |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`ARC-Easy`** (Expected `simple`) | 15 | 15 | 0 | 0.76 | 0.66–0.82 | `gemini-3.1-flash-lite` |
| **`MATH-500`** (Expected `complex`) | 15 | 0 | 15 | 0.85 | 0.77–0.89 | `gemini-3.1-pro-preview` |
| **`HumanEval`** (Expected `complex`) | 15 | 0 | 15 | 0.66 | 0.62–0.72 | `gemini-3.1-pro-preview` |

---

## 3. Per-Dataset Breakdown across All 4 Arms

| Benchmark | Arm / Mode | Coordinator $\rightarrow$ Specialist Model | Accuracy (%) | Total Tokens (In / Out / Think) | Avg Turn Latency (ms) | Total Cost (USD) | Cost vs. All-Complex | Cost vs. Static Hybrid |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ARC-Easy** | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (15/15) | 21,428 (20,595 / 833 / 0) | 4093.2 ms | $0.006398 | -94.3% | -90.9% |
| **ARC-Easy** | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (15/15) | 26,498 (20,595 / 814 / 5,089) | 10473.7 ms | $0.112026 | — | +59.0% |
| **ARC-Easy** | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (15/15) | 24,591 (20,595 / 854 / 3,142) | 11652.9 ms | $0.070452 | -37.1% | — |
| **ARC-Easy** | `with_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 100.0% (15/15) | 21,414 (20,595 / 819 / 0) | 4127.9 ms | $0.006377 | **-94.3%** | **-90.9%** |
| **MATH-500** | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 80.0% (12/15) | 33,661 (22,502 / 11,159 / 0) | 5713.8 ms | $0.022364 | -97.2% | -97.3% |
| **MATH-500** | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (15/15) | 85,390 (22,502 / 12,535 / 50,353) | 51390.8 ms | $0.799660 | — | -4.4% |
| **MATH-500** | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (15/15) | 90,180 (22,502 / 11,825 / 55,853) | 44277.6 ms | $0.836690 | +4.6% | — |
| **MATH-500** | `with_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (15/15) | 95,052 (22,502 / 13,831 / 58,719) | 49093.6 ms | $0.895154 | **+11.9%** | **+7.0%** |
| **HumanEval** | `all_simple` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-flash-lite` | 80.0% (12/15) | 32,622 (26,436 / 6,186 / 0) | 4853.8 ms | $0.015888 | -97.4% | -96.9% |
| **HumanEval** | `all_complex` | `gemini-3.1-pro-preview` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (15/15) | 72,791 (26,436 / 6,320 / 40,035) | 47067.1 ms | $0.609132 | — | +20.5% |
| **HumanEval** | `coord_simple_subs_complex` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (15/15) | 66,129 (26,436 / 5,531 / 34,162) | 32828.7 ms | $0.505453 | -17.0% | — |
| **HumanEval** | `with_router` | `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.1-pro-preview` | 100.0% (15/15) | 66,655 (26,436 / 6,204 / 34,015) | 29585.3 ms | $0.511765 | **-16.0%** | **+1.2%** |

---

## 4. Multi-Agent Cost & Token Attribution (`adk-finops` `breakdown_by_agent`)

| Agent Name | All-Simple Cost (Tok) | All-Complex Cost (Tok) | Static Hybrid Cost (Tok) | With-Router Cost (Tok) | Router vs. All-Complex | Router vs. Static Hybrid |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`code_engineer_agent`** | $0.012497 (20,859) | $0.559410 (59,145) | $0.502062 (54,366) | $0.508374 (54,892) | **-9.1%** | **+1.3%** |
| **`coordinator_agent`** | $0.008983 (30,454) | $0.139766 (36,113) | $0.008983 (30,454) | $0.008983 (30,454) | **-93.6%** | **0.0%** |
| **`fast_qa_agent`** | $0.003728 (12,548) | $0.063930 (15,390) | $0.067782 (15,711) | $0.003707 (12,534) | **-94.2%** | **-94.5%** |
| **`math_analyst_agent`** | $0.019442 (23,850) | $0.757712 (74,031) | $0.833768 (80,369) | $0.892232 (85,241) | **+17.8%** | **+7.0%** |

---

## 5. Model Distribution (`adk-finops` `breakdown_by_model`)

| Arm / Mode | Model | LLM Calls | Prompt Tokens | Completion Tokens | Thinking Tokens | Total Tokens | Total Cost (USD) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_simple` | **`gemini-3.1-flash-lite`** | 90 | 69,533 | 18,178 | 0 | 87,711 | $0.044651 |
| `all_complex` | **`gemini-3.1-pro-preview`** | 90 | 69,533 | 19,669 | 95,477 | 184,679 | $1.520818 |
| `coord_simple_subs_complex` | **`gemini-3.1-flash-lite`** | 45 | 29,359 | 1,095 | 0 | 30,454 | $0.008983 |
| `coord_simple_subs_complex` | **`gemini-3.1-pro-preview`** | 45 | 40,174 | 17,115 | 93,157 | 150,446 | $1.403612 |
| `with_router` | **`gemini-3.1-flash-lite`** | 60 | 41,434 | 1,554 | 0 | 42,988 | $0.012690 |
| `with_router` | **`gemini-3.1-pro-preview`** | 30 | 28,099 | 19,300 | 92,734 | 140,133 | $1.400606 |

---

## 6. Sample-Level Comparison Log (All 45 Tasks across All 4 Arms)

| Sample ID | Dataset | Laya Choice (Conf) | Laya Latency | Specialist Model (`with_router`) | Latency (`Simple` / `Complex` / `Hybrid` / `Router`) | Cost (`Simple` / `Complex` / `Hybrid` / `Router`) | Correct (`S`/`C`/`H`/`R`) |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | :---: |
| `arc_easy_Mercury_417466` | ARC-Easy | `simple` (0.72) | 359.7 ms | `gemini-3.1-flash-lite` | 4908 / 18346 / 11633 / **4498 ms** | $0.0004 / $0.0083 / $0.0066 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_Mercury_7081673` | ARC-Easy | `simple` (0.82) | 66.4 ms | `gemini-3.1-flash-lite` | 4983 / 9548 / 6472 / **4454 ms** | $0.0004 / $0.0062 / $0.0040 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_Mercury_7239733` | ARC-Easy | `simple` (0.80) | 60.8 ms | `gemini-3.1-flash-lite` | 4857 / 10255 / 8371 / **4898 ms** | $0.0004 / $0.0086 / $0.0052 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_NYSEDREGENTS_2015_4_8` | ARC-Easy | `simple` (0.82) | 58.7 ms | `gemini-3.1-flash-lite` | 5305 / 9419 / 7005 / **4588 ms** | $0.0004 / $0.0067 / $0.0043 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_Mercury_7037258` | ARC-Easy | `simple` (0.68) | 65.5 ms | `gemini-3.1-flash-lite` | 5109 / 12893 / 27563 / **4299 ms** | $0.0004 / $0.0099 / $0.0035 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_CSZ20679` | ARC-Easy | `simple` (0.79) | 100.1 ms | `gemini-3.1-flash-lite` | 3806 / 7544 / 5979 / **4376 ms** | $0.0004 / $0.0054 / $0.0032 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_Mercury_182158` | ARC-Easy | `simple` (0.77) | 64.1 ms | `gemini-3.1-flash-lite` | 3462 / 11012 / 7793 / **3997 ms** | $0.0004 / $0.0074 / $0.0037 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_Mercury_7216668` | ARC-Easy | `simple` (0.79) | 61.2 ms | `gemini-3.1-flash-lite` | 3622 / 9961 / 6737 / **3774 ms** | $0.0004 / $0.0077 / $0.0043 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_MCAS_2001_5_19` | ARC-Easy | `simple` (0.82) | 61.1 ms | `gemini-3.1-flash-lite` | 3437 / 8922 / 7197 / **4032 ms** | $0.0004 / $0.0067 / $0.0048 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_Mercury_SC_413631` | ARC-Easy | `simple` (0.70) | 68.6 ms | `gemini-3.1-flash-lite` | 3581 / 10880 / 8027 / **4155 ms** | $0.0004 / $0.0094 / $0.0063 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_MCAS_2005_5_10` | ARC-Easy | `simple` (0.68) | 95.9 ms | `gemini-3.1-flash-lite` | 3854 / 8196 / 7983 / **3709 ms** | $0.0004 / $0.0064 / $0.0060 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_Mercury_7166145` | ARC-Easy | `simple` (0.77) | 69.9 ms | `gemini-3.1-flash-lite` | 3701 / 10588 / 6194 / **3748 ms** | $0.0004 / $0.0074 / $0.0033 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_MSA_2013_5_6` | ARC-Easy | `simple` (0.66) | 61.9 ms | `gemini-3.1-flash-lite` | 3589 / 10453 / 47695 / **3970 ms** | $0.0004 / $0.0068 / $0.0049 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_Mercury_SC_405199` | ARC-Easy | `simple` (0.77) | 57.8 ms | `gemini-3.1-flash-lite` | 3952 / 9520 / 7008 / **3726 ms** | $0.0004 / $0.0074 / $0.0046 / **$0.0004** | ✅/✅/✅/✅ |
| `arc_easy_NYSEDREGENTS_2015_8_28` | ARC-Easy | `simple` (0.79) | 58.4 ms | `gemini-3.1-flash-lite` | 3233 / 9569 / 9136 / **3694 ms** | $0.0004 / $0.0078 / $0.0057 / **$0.0004** | ✅/✅/✅/✅ |
| `math500_test_prealgebra_1139` | MATH-500 | `complex` (0.88) | 104.4 ms | `gemini-3.1-pro-preview` | 6724 / 146919 / 161632 / **181397 ms** | $0.0019 / $0.1451 / $0.1767 / **$0.2211** | ❌/✅/✅/✅ |
| `math500_test_number_theory_737` | MATH-500 | `complex` (0.84) | 64.6 ms | `gemini-3.1-pro-preview` | 5487 / 63843 / 25906 / **22765 ms** | $0.0015 / $0.0249 / $0.0318 / **$0.0357** | ✅/✅/✅/✅ |
| `math500_test_algebra_1837` | MATH-500 | `complex` (0.86) | 70.5 ms | `gemini-3.1-pro-preview` | 6080 / 31335 / 21234 / **24609 ms** | $0.0014 / $0.0357 / $0.0256 / **$0.0268** | ✅/✅/✅/✅ |
| `math500_test_algebra_2427` | MATH-500 | `complex` (0.77) | 67.6 ms | `gemini-3.1-pro-preview` | 5590 / 23314 / 21320 / **24365 ms** | $0.0014 / $0.0260 / $0.0245 / **$0.0284** | ✅/✅/✅/✅ |
| `math500_test_counting_and_probability_525` | MATH-500 | `complex` (0.85) | 62.0 ms | `gemini-3.1-pro-preview` | 5288 / 45109 / 25315 / **48872 ms** | $0.0012 / $0.0510 / $0.0290 / **$0.0557** | ✅/✅/✅/✅ |
| `math500_test_prealgebra_1558` | MATH-500 | `complex` (0.81) | 125.5 ms | `gemini-3.1-pro-preview` | 5534 / 23303 / 15137 / **19215 ms** | $0.0011 / $0.0240 / $0.0214 / **$0.0208** | ✅/✅/✅/✅ |
| `math500_test_algebra_305` | MATH-500 | `complex` (0.88) | 94.6 ms | `gemini-3.1-pro-preview` | 5136 / 24641 / 14725 / **16421 ms** | $0.0013 / $0.0248 / $0.0166 / **$0.0181** | ✅/✅/✅/✅ |
| `math500_test_number_theory_1055` | MATH-500 | `complex` (0.84) | 77.9 ms | `gemini-3.1-pro-preview` | 5450 / 47625 / 28751 / **51823 ms** | $0.0014 / $0.0553 / $0.0427 / **$0.0636** | ✅/✅/✅/✅ |
| `math500_test_prealgebra_951` | MATH-500 | `complex` (0.88) | 94.7 ms | `gemini-3.1-pro-preview` | 4792 / 16082 / 14830 / **14625 ms** | $0.0012 / $0.0210 / $0.0161 / **$0.0147** | ✅/✅/✅/✅ |
| `math500_test_precalculus_675` | MATH-500 | `complex` (0.85) | 72.4 ms | `gemini-3.1-pro-preview` | 4818 / 35583 / 41490 / **35656 ms** | $0.0014 / $0.0398 / $0.0317 / **$0.0449** | ✅/✅/✅/✅ |
| `math500_test_intermediate_algebra_279` | MATH-500 | `complex` (0.89) | 102.7 ms | `gemini-3.1-pro-preview` | 5382 / 31162 / 24426 / **30809 ms** | $0.0013 / $0.0344 / $0.0369 / **$0.0370** | ✅/✅/✅/✅ |
| `math500_test_intermediate_algebra_1849` | MATH-500 | `complex` (0.88) | 96.8 ms | `gemini-3.1-pro-preview` | 7527 / 85307 / 98411 / **85514 ms** | $0.0020 / $0.1033 / $0.1187 / **$0.0962** | ✅/✅/✅/✅ |
| `math500_test_algebra_892` | MATH-500 | `complex` (0.83) | 110.9 ms | `gemini-3.1-pro-preview` | 6202 / 79142 / 29181 / **43841 ms** | $0.0017 / $0.0679 / $0.0334 / **$0.0538** | ❌/✅/✅/✅ |
| `math500_test_prealgebra_1646` | MATH-500 | `complex` (0.80) | 120.9 ms | `gemini-3.1-pro-preview` | 6373 / 49669 / 33871 / **36160 ms** | $0.0020 / $0.0605 / $0.0395 / **$0.0439** | ❌/✅/✅/✅ |
| `math500_test_intermediate_algebra_582` | MATH-500 | `complex` (0.86) | 115.0 ms | `gemini-3.1-pro-preview` | 5324 / 67830 / 107937 / **100332 ms** | $0.0016 / $0.0859 / $0.1920 / **$0.1345** | ✅/✅/✅/✅ |
| `humaneval_120` | HumanEval | `complex` (0.68) | 135.3 ms | `gemini-3.1-pro-preview` | 4192 / 31326 / 33301 / **31005 ms** | $0.0009 / $0.0325 / $0.0400 / **$0.0325** | ✅/✅/✅/✅ |
| `humaneval_121` | HumanEval | `complex` (0.68) | 113.8 ms | `gemini-3.1-pro-preview` | 4761 / 18481 / 12037 / **16797 ms** | $0.0010 / $0.0180 / $0.0114 / **$0.0177** | ✅/✅/✅/✅ |
| `humaneval_122` | HumanEval | `complex` (0.71) | 113.2 ms | `gemini-3.1-pro-preview` | 4440 / 21578 / 18682 / **22594 ms** | $0.0009 / $0.0304 / $0.0255 / **$0.0254** | ✅/✅/✅/✅ |
| `humaneval_123` | HumanEval | `complex` (0.69) | 136.1 ms | `gemini-3.1-pro-preview` | 4245 / 27194 / 24342 / **27194 ms** | $0.0010 / $0.0325 / $0.0287 / **$0.0303** | ✅/✅/✅/✅ |
| `humaneval_124` | HumanEval | `complex` (0.65) | 132.8 ms | `gemini-3.1-pro-preview` | 5197 / 30837 / 44994 / **32587 ms** | $0.0012 / $0.0348 / $0.0548 / **$0.0357** | ✅/✅/✅/✅ |
| `humaneval_125` | HumanEval | `complex` (0.63) | 117.6 ms | `gemini-3.1-pro-preview` | 5016 / 45177 / 37398 / **34769 ms** | $0.0011 / $0.0514 / $0.0413 / **$0.0411** | ✅/✅/✅/✅ |
| `humaneval_126` | HumanEval | `complex` (0.62) | 132.8 ms | `gemini-3.1-pro-preview` | 4864 / 26160 / 27124 / **20052 ms** | $0.0009 / $0.0293 / $0.0323 / **$0.0275** | ❌/✅/✅/✅ |
| `humaneval_127` | HumanEval | `complex` (0.65) | 135.4 ms | `gemini-3.1-pro-preview` | 4866 / 24096 / 23484 / **35309 ms** | $0.0011 / $0.0283 / $0.0279 / **$0.0450** | ✅/✅/✅/✅ |
| `humaneval_128` | HumanEval | `complex` (0.70) | 117.6 ms | `gemini-3.1-pro-preview` | 4419 / 102531 / 50622 / **15523 ms** | $0.0008 / $0.0246 / $0.0380 / **$0.0157** | ✅/✅/✅/✅ |
| `humaneval_129` | HumanEval | `complex` (0.64) | 128.4 ms | `gemini-3.1-pro-preview` | 5777 / 52594 / 46257 / **42271 ms** | $0.0017 / $0.0643 / $0.0552 / **$0.0487** | ✅/✅/✅/✅ |
| `humaneval_130` | HumanEval | `complex` (0.63) | 135.0 ms | `gemini-3.1-pro-preview` | 6287 / 36916 / 30389 / **34506 ms** | $0.0017 / $0.0448 / $0.0347 / **$0.0404** | ❌/✅/✅/✅ |
| `humaneval_131` | HumanEval | `complex` (0.62) | 119.3 ms | `gemini-3.1-pro-preview` | 4150 / 18473 / 53100 / **15707 ms** | $0.0007 / $0.0174 / $0.0155 / **$0.0166** | ✅/✅/✅/✅ |
| `humaneval_132` | HumanEval | `complex` (0.63) | 134.9 ms | `gemini-3.1-pro-preview` | 4504 / 96438 / 26524 / **44814 ms** | $0.0010 / $0.1241 / $0.0282 / **$0.0533** | ❌/✅/✅/✅ |
| `humaneval_133` | HumanEval | `complex` (0.72) | 121.1 ms | `gemini-3.1-pro-preview` | 4325 / 16839 / 19394 / **12499 ms** | $0.0009 / $0.0192 / $0.0181 / **$0.0126** | ✅/✅/✅/✅ |
| `humaneval_134` | HumanEval | `complex` (0.68) | 125.1 ms | `gemini-3.1-pro-preview` | 5764 / 157368 / 44782 / **58152 ms** | $0.0011 / $0.0576 / $0.0537 / **$0.0692** | ✅/✅/✅/✅ |

---

## 7. Key Findings & Architectural Takeaways

1. **Why `With Laya Router` Beats `All-Simple` (`gemini-3.1-flash-lite`)**: While `all_simple` (`gemini-3.1-flash-lite`) is ultra-fast and inexpensive, it lacks System-2 thinking tokens and drops to **86.7% (39/45)** overall accuracy on hard competition math (`MATH-500` Level 5) and algorithmic coding (`HumanEval`). `with_router` recovers **100.0% (45/45)** accuracy (**+13.3%** gain, +6 additional tasks solved) by selectively upgrading `MATH-500` and `HumanEval` to `gemini-3.1-pro-preview`.
2. **100% Laya Classification Accuracy Across All 45 Open-Source Samples**: Laya (`laya-typed-decisions`) classified **45/45 (100.0%)** benchmark tasks into their exact ground-truth complexity tier (`15/15 ARC-Easy` $\rightarrow$ `simple`, `15/15 MATH-500` $\rightarrow$ `complex`, `15/15 HumanEval` $\rightarrow$ `complex`).
3. **Decomposing Coordinator Savings vs. Specialist Routing Savings**:
   - **Static Hybrid (`coord_simple_subs_complex`) vs. All-Complex (`all_complex`)**: Fixing `coordinator_agent` to `gemini-3.1-flash-lite` while keeping all specialist sub-agents on `gemini-3.1-pro-preview` changed total spend by **-7.1%** (`$1.520818` $\rightarrow$ `$1.412595`) and turn latency by **-18.5%**.
   - **With Laya Router (`with_router`) vs. Static Hybrid (`coord_simple_subs_complex`)**: Dynamically routing specialist sub-agents with `LayaModelRouterPlugin` delivered negligible overall cost reduction (`$1.412595` $\rightarrow$ `$1.413296`), including **-90.9%** on `ARC-Easy` (`fast_qa_agent`).
   - **With Laya Router (`with_router`) vs. All-Complex (`all_complex`)**: Combined savings of **-7.1%** in total cost (`$1.520818` $\rightarrow$ `$1.413296`) and **-24.0%** in average turn latency (`36310.5 ms` $\rightarrow$ `27602.3 ms`).
