import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from laya import Router

from .questions import LAYA_COMPLEXITY_QUESTIONS

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

logger = logging.getLogger(__name__)

# Preload the English checkpoint at startup so the first request pays no load latency
laya_engine = Router(default="laya-typed-decisions").preload(["laya-typed-decisions"])


# 1. Standalone ADK before_model_callback using Laya
async def laya_router(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    *,
    router: Router | Any | None = None,
    model: str | None = None,
    confidence_threshold: float = 0.7,
) -> LlmResponse | None:
    """Standalone ADK `before_model_callback` using `laya.Router` (`convaiinnovations/laya`)."""
    if not llm_request.contents or llm_request.contents[-1].role != "user":
        if routed_model := callback_context.state.get("routed_model"):
            llm_request.model = routed_model
        return None

    user_q = " ".join(
        p.text
        for p in (llm_request.contents[-1].parts or [])
        if getattr(p, "text", None)
    ).strip()

    if not user_q:
        if routed_model := callback_context.state.get("routed_model"):
            llm_request.model = routed_model
        return None

    recent_turns: list[str] = []
    for content in llm_request.contents[-5:-1]:
        turn_text = " ".join(
            p.text for p in (content.parts or []) if getattr(p, "text", None)
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
        res = await asyncio.to_thread(
            active_router.predict,
            eval_state,
            LAYA_COMPLEXITY_QUESTIONS,
            **predict_kwargs,
        )
        ans = res["answers"]["complexity"]
        choice = str(ans["choice"])
        confidence = float(ans.get("answer_confidence", ans.get("confidence", 0.0)))
        checkpoint = str((res.get("routing") or {}).get("model", "english"))
    except Exception as exc:
        logger.warning(
            "[LayaRouter] Classifier evaluation failed (%s); falling back to existing/default model.",
            exc,
        )
        if routed_model := callback_context.state.get("routed_model"):
            llm_request.model = routed_model
        return None

    is_complex = choice == "complex" or confidence < confidence_threshold
    if is_complex:
        logger.info(
            "[LayaRouter] Upgraded to gemini-3.1-pro-preview (choice=%s, conf=%.2f, laya_model=%s)",
            choice,
            confidence,
            checkpoint,
        )
        llm_request.model = "gemini-3.1-pro-preview"
    else:
        logger.info(
            "[LayaRouter] Routed to gemini-3.6-flash (choice=%s, conf=%.2f, laya_model=%s)",
            choice,
            confidence,
            checkpoint,
        )
        llm_request.model = "gemini-3.6-flash"

    callback_context.state["routed_model"] = llm_request.model
    return None


# 3. ADK Plugin wrapper around Laya Router
class LayaModelRouterPlugin(BasePlugin):
    """ADK Plugin wrapper around `laya.Router` (`convaiinnovations/laya`)."""

    def __init__(
        self,
        router: Router | Any | None = None,
        model: str | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        super().__init__(name="laya_model_router_plugin")
        self._router = router or laya_engine
        self._model = model
        self._confidence_threshold = confidence_threshold

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        return await laya_router(
            callback_context=callback_context,
            llm_request=llm_request,
            router=self._router,
            model=self._model,
            confidence_threshold=self._confidence_threshold,
        )


# 4. Attach to the Laya Routed ADK App
root_agent = Agent(
    name="root_agent",
    model="gemini-3.1-pro-preview",  # Default fallback
    instruction="You are an AI assistant.",
)

app = App(
    name="laya_adk_model_router",
    root_agent=root_agent,
    plugins=[LayaModelRouterPlugin()],
)
