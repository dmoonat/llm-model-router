"""Multi-Agent Google ADK application for evaluating Laya Model Router with adk-finops.

Architecture:
  coordinator_agent (Root Orchestrator)
  ├── fast_qa_agent (Factual & Science Multiple-Choice Specialist)
  ├── math_analyst_agent (Multi-Step Mathematical Reasoning Specialist)
  └── code_engineer_agent (Python Software Engineering & Algorithm Specialist)

4 Experimental Arms:
  1. `all_simple`: coordinator=SIMPLE_MODEL, sub-agents=SIMPLE_MODEL (no router)
  2. `all_complex`: coordinator=COMPLEX_MODEL, sub-agents=COMPLEX_MODEL (no router)
  3. `coord_simple_subs_complex`: coordinator=SIMPLE_MODEL, sub-agents=COMPLEX_MODEL (no router)
  4. `with_router`: coordinator=SIMPLE_MODEL (`exclude_agents={"coordinator_agent"}`),
     sub-agents dynamically routed by `LayaModelRouterPlugin` between SIMPLE_MODEL and COMPLEX_MODEL.
"""

from __future__ import annotations

from adk_finops import FinOpsCostPlugin
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from laya_adk_model_router.plugin import (
    COMPLEX_MODEL,
    SIMPLE_MODEL,
    LayaModelRouterPlugin,
)


def build_multi_agent_hierarchy(
    coordinator_model: str = COMPLEX_MODEL,
    specialist_model: str = COMPLEX_MODEL,
    temperature: float = 0.0,
) -> Agent:
    """Creates a fresh 4-agent ADK hierarchy (Coordinator + 3 Specialist Sub-Agents) with deterministic temperature."""
    gen_config = types.GenerateContentConfig(temperature=temperature)

    fast_qa_agent = Agent(
        name="fast_qa_agent",
        model=specialist_model,
        generate_content_config=gen_config,
        description=(
            "Specialist sub-agent for factual lookups, science/legal/analytical reasoning, "
            "and multiple-choice questions (`Options:`)."
        ),
        instruction=(
            "You are an accurate factual, scientific, and analytical multiple-choice QA specialist.\n"
            "- Carefully evaluate the question and all provided options.\n"
            "- Always end your response with `FINAL_OPTION: <label>` (for example, `FINAL_OPTION: A`)."
        ),
    )

    math_analyst_agent = Agent(
        name="math_analyst_agent",
        model=specialist_model,
        generate_content_config=gen_config,
        description=(
            "Specialist sub-agent for arithmetic word problems, competition mathematics, "
            "algebraic reasoning, and quantitative analysis (`FINAL_ANSWER:`)."
        ),
        instruction=(
            "You are a mathematical reasoning specialist.\n"
            "- Solve the mathematical problem step-by-step and verify intermediate arithmetic.\n"
            "- Always conclude your response with `FINAL_ANSWER: <number>` "
            "(using only the final numeric value, e.g. `FINAL_ANSWER: 42`)."
        ),
    )

    code_engineer_agent = Agent(
        name="code_engineer_agent",
        model=specialist_model,
        generate_content_config=gen_config,
        description=(
            "Specialist sub-agent for Python code generation, utility functions, "
            "and algorithm design (`Required function signature:`)."
        ),
        instruction=(
            "You are a senior Python software engineer.\n"
            "- Implement the requested Python function using the exact function signature specified.\n"
            "- Enclose the complete, self-contained Python implementation inside a "
            "```python ... ``` code block (including any needed standard library imports)."
        ),
    )

    coordinator_agent = Agent(
        name="coordinator_agent",
        model=coordinator_model,
        generate_content_config=gen_config,
        description="Root orchestrator that delegates user requests to the right specialist sub-agent.",
        instruction=(
            "You are the Root Coordinator for a team of specialist agents.\n"
            "Do NOT answer the user's question directly. Instead, immediately delegate "
            "to the appropriate specialist sub-agent using `transfer_to_agent`:\n"
            "1. Transfer to `fast_qa_agent` for multiple-choice questions (`Options:` / `FINAL_OPTION:`).\n"
            "2. Transfer to `math_analyst_agent` for math problems (`FINAL_ANSWER:`).\n"
            "3. Transfer to `code_engineer_agent` for Python programming tasks (`Required function signature:`)."
        ),
        sub_agents=[fast_qa_agent, math_analyst_agent, code_engineer_agent],
    )
    return coordinator_agent


def build_eval_app(
    *,
    mode: str = "with_router",
    use_router: bool | None = None,
    finops_plugin: FinOpsCostPlugin,
    confidence_threshold: float = 0.6,
    simple_model: str = SIMPLE_MODEL,
    complex_model: str = COMPLEX_MODEL,
    exclude_agents: set[str] | None = None,
) -> App:
    """Constructs the multi-agent ADK App for one of the 4 evaluation arms.

    Modes:
      - `all_simple`: Coordinator (`simple_model`) -> Sub-agents (`simple_model`).
      - `all_complex`: Coordinator (`complex_model`) -> Sub-agents (`complex_model`).
      - `coord_simple_subs_complex`: Coordinator (`simple_model`) -> Sub-agents (`complex_model`) (no router).
      - `with_router`: Coordinator (`simple_model`, excluded from upgrade via `exclude_agents={"coordinator_agent"}`)
        -> Sub-agents dynamically routed by `LayaModelRouterPlugin` between `simple_model` and `complex_model`.
    """
    if use_router is not None:
        mode = "with_router" if use_router else "all_complex"

    if mode == "all_simple":
        coordinator_model = simple_model
        specialist_model = simple_model
    elif mode == "all_complex":
        coordinator_model = complex_model
        specialist_model = complex_model
    elif mode == "coord_simple_subs_complex":
        coordinator_model = simple_model
        specialist_model = complex_model
    elif mode == "with_router":
        coordinator_model = simple_model
        specialist_model = complex_model
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")

    root_coordinator = build_multi_agent_hierarchy(
        coordinator_model=coordinator_model,
        specialist_model=specialist_model,
    )

    plugins: list[BasePlugin] = []
    if mode == "with_router":
        active_excluded = (
            exclude_agents if exclude_agents is not None else {"coordinator_agent"}
        )
        plugins.append(
            LayaModelRouterPlugin(
                confidence_threshold=confidence_threshold,
                simple_model=simple_model,
                complex_model=complex_model,
                exclude_agents=active_excluded,
            )
        )
    plugins.append(finops_plugin)

    app_name = f"multi_agent_{mode}"
    return App(
        name=app_name,
        root_agent=root_coordinator,
        plugins=plugins,
    )
