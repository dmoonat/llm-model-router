import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from laya import Router

from .questions import LAYA_COMPLEXITY_QUESTIONS

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

SIMPLE_MODEL = os.getenv("SIMPLE_MODEL", "gemini-3.1-flash-lite")
COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", "gemini-3.1-pro-preview")

logger = logging.getLogger(__name__)

# Preload the checkpoint at startup so the first request pays no load latency
laya_engine = Router(default="laya-typed-decisions").preload(["laya-typed-decisions"])
_laya_predict_lock = asyncio.Lock()


# 1. Standalone ADK before_model_callback using Laya
async def laya_router(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    *,
    router: Router | Any | None = None,
    model: str | None = None,
    confidence_threshold: float = 0.7,
    simple_model: str = SIMPLE_MODEL,
    complex_model: str = COMPLEX_MODEL,
    exclude_agents: set[str] | list[str] | tuple[str, ...] | None = None,
) -> LlmResponse | None:
    """Standalone ADK `before_model_callback` using `laya.Router` (`convaiinnovations/laya`).

    Args:
        callback_context: ADK callback context.
        llm_request: Outgoing ADK LLM request whose `.model` will be dynamically set.
        router: Optional preloaded `laya.Router` instance.
        model: Optional Laya checkpoint name (`"laya"`, `"laya-multilingual"`, `"laya-typed-decisions"`).
        confidence_threshold: Minimum confidence required to stay on `simple_model`.
        simple_model: Fast/cost-efficient model ID for simple queries.
        complex_model: Frontier reasoning model ID for complex or low-confidence queries.
        exclude_agents: Optional collection of agent names (e.g. `{"coordinator_agent"}`) that
            should always remain pinned to `simple_model` (such as lightweight routing/triage
            agents), while still computing and storing the turn's `routed_model` in session
            state for specialist sub-agents.
    """
    invocation_id = getattr(callback_context, "invocation_id", None)
    agent_name = getattr(callback_context, "agent_name", "") or ""
    excluded_set = set(exclude_agents) if exclude_agents else set()
    is_excluded_agent = agent_name in excluded_set

    # If this user turn (invocation_id) was already routed (e.g. coordinator -> sub-agent delegation),
    # apply simple_model for excluded triage agents or the turn's routed_model for specialist agents.
    if (
        invocation_id
        and callback_context.state.get("last_routed_invocation_id") == invocation_id
        and (routed_model := callback_context.state.get("routed_model"))
    ):
        llm_request.model = simple_model if is_excluded_agent else routed_model
        return None

    if not llm_request.contents or llm_request.contents[-1].role != "user":
        if is_excluded_agent:
            llm_request.model = simple_model
        elif routed_model := callback_context.state.get("routed_model"):
            llm_request.model = routed_model
        return None

    user_q = " ".join(
        p.text
        for p in (llm_request.contents[-1].parts or [])
        if getattr(p, "text", None) and not p.text.lstrip().startswith("For context:")
    ).strip()

    if not user_q:
        if is_excluded_agent:
            llm_request.model = simple_model
        elif routed_model := callback_context.state.get("routed_model"):
            llm_request.model = routed_model
        return None

    if (
        callback_context.state.get("last_routed_user_q") == user_q
        and (routed_model := callback_context.state.get("routed_model"))
    ):
        llm_request.model = simple_model if is_excluded_agent else routed_model
        return None

    recent_turns: list[str] = []
    for content in llm_request.contents[-5:-1]:
        turn_text = " ".join(
            p.text
            for p in (content.parts or [])
            if getattr(p, "text", None) and not p.text.lstrip().startswith("For context:")
        ).strip()
        if turn_text:
            recent_turns.append(f"{content.role}: {turn_text}")

    eval_state: dict[str, Any] = {"user_message": user_q}
    if recent_turns:
        eval_state["recent_context"] = "\n".join(recent_turns)

    active_router = router or laya_engine
    predict_kwargs: dict[str, Any] = {}
    target_model = model if model is not None else getattr(active_router, "default", None)
    if target_model is not None:
        predict_kwargs["model"] = target_model

    try:
        async with _laya_predict_lock:
            t0 = time.perf_counter()
            res = await asyncio.to_thread(
                active_router.predict,
                eval_state,
                LAYA_COMPLEXITY_QUESTIONS,
                **predict_kwargs,
            )
            laya_latency_ms = (time.perf_counter() - t0) * 1000.0
        ans = res["answers"]["complexity"]
        choice = str(ans["choice"])
        confidence = float(ans.get("answer_confidence", ans.get("confidence", 0.0)))
        checkpoint = str((res.get("routing") or {}).get("model", "english"))
    except Exception as exc:
        logger.warning(
            "[LayaRouter] Classifier evaluation failed (%s); falling back to existing/default model.",
            exc,
        )
        if is_excluded_agent:
            llm_request.model = simple_model
        elif routed_model := callback_context.state.get("routed_model"):
            llm_request.model = routed_model
        return None

    is_complex = choice == "complex" or confidence < confidence_threshold
    routed_model = complex_model if is_complex else simple_model

    if is_complex:
        logger.info(
            "[LayaRouter] Upgraded turn to %s (choice=%s, conf=%.2f, laya_model=%s, latency=%.1fms%s)",
            complex_model,
            choice,
            confidence,
            checkpoint,
            laya_latency_ms,
            f", agent={agent_name} pinned to {simple_model}" if is_excluded_agent else "",
        )
    else:
        logger.info(
            "[LayaRouter] Routed turn to %s (choice=%s, conf=%.2f, laya_model=%s, latency=%.1fms)",
            simple_model,
            choice,
            confidence,
            checkpoint,
            laya_latency_ms,
        )

    llm_request.model = simple_model if is_excluded_agent else routed_model

    callback_context.state["routed_model"] = routed_model
    callback_context.state["last_routed_user_q"] = user_q
    if invocation_id:
        callback_context.state["last_routed_invocation_id"] = invocation_id
    callback_context.state["laya_routing"] = {
        "routed_model": routed_model,
        "choice": choice,
        "confidence": round(confidence, 4),
        "checkpoint": checkpoint,
        "laya_latency_ms": round(laya_latency_ms, 2),
        "excluded_agents": sorted(excluded_set),
    }
    return None


# 2. ADK Plugin wrapper around Laya Router
class LayaModelRouterPlugin(BasePlugin):
    """ADK Plugin wrapper around `laya.Router` (`convaiinnovations/laya`)."""

    def __init__(
        self,
        router: Router | Any | None = None,
        model: str | None = None,
        confidence_threshold: float = 0.7,
        simple_model: str = SIMPLE_MODEL,
        complex_model: str = COMPLEX_MODEL,
        exclude_agents: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(name="laya_model_router_plugin")
        self._router = router or laya_engine
        self._model = model
        self._confidence_threshold = confidence_threshold
        self._simple_model = simple_model
        self._complex_model = complex_model
        self._exclude_agents = set(exclude_agents) if exclude_agents else set()

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        return await laya_router(
            callback_context=callback_context,
            llm_request=llm_request,
            router=self._router,
            model=self._model,
            confidence_threshold=self._confidence_threshold,
            simple_model=self._simple_model,
            complex_model=self._complex_model,
            exclude_agents=self._exclude_agents,
        )
