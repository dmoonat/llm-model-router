import asyncio
import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from google import genai
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from .questions import DECISIONS_COMPLEXITY_QUESTIONS

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

SIMPLE_MODEL = os.getenv("SIMPLE_MODEL", "gemini-3.1-flash-lite")
COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", "gemini-3.1-pro-preview")
DECISION_MODEL = os.getenv("DECISION_MODEL", "gemini-decision-1")
DECISIONS_REST_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"

# Gemini Decision 1 pricing: $0.10 per 1M input tokens, $0.00 output tokens (always 0 output tokens)
DECISION_INPUT_PRICE_PER_1M = 0.10

logger = logging.getLogger(__name__)


def _resolve_decisions_api_key(explicit_api_key: str | None = None) -> str | None:
    """Resolves the Gemini Developer API key used for `gemini-decision-1` (`v1beta`)."""
    return (
        explicit_api_key
        or os.getenv("DECISIONS_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


class DecisionsClient:
    """Client for the Google Gemini Decisions API (`gemini-decision-1`, `v1beta`).

    Uses `google.genai.Client(http_options={"api_version": "v1beta"}).decisions.create(...)`
    when `google-genai >= 3.2` is installed, and automatically falls back to the
    `POST https://generativelanguage.googleapis.com/v1beta/decisions` REST endpoint
    when an earlier `google-genai` version is installed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DECISION_MODEL,
        endpoint: str = DECISIONS_REST_ENDPOINT,
        timeout_sec: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self.default_model = model
        self._endpoint = endpoint
        self._timeout_sec = timeout_sec
        self._sdk_client: Any | None = None
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def _get_api_key(self) -> str:
        key = _resolve_decisions_api_key(self._api_key)
        if not key:
            raise RuntimeError(
                "Missing API key for Gemini Decisions API (`gemini-decision-1`). "
                "Set `GEMINI_API_KEY` (or `DECISIONS_API_KEY` / `GOOGLE_API_KEY`) in your environment or `.env` file."
            )
        return key

    def _get_sdk_decisions_resource(self, api_key: str) -> Any | None:
        """Returns `client.decisions` if `google-genai >= 3.2` is installed, else `None`."""
        if self._sdk_client is None:
            try:
                self._sdk_client = genai.Client(
                    api_key=api_key,
                    vertexai=False,
                    http_options={"api_version": "v1beta"},
                )
            except Exception:
                return None
        return getattr(self._sdk_client, "decisions", None)

    def predict(
        self,
        input_data: str | dict[str, Any] | list[dict[str, Any]],
        questions: dict[str, dict[str, Any]] | None = None,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Evaluates typed questions against `input_data` via `gemini-decision-1`.

        Returns a normalized dictionary matching the Decisions API response structure:
        ```python
        {
            "answers": {
                "complexity": {
                    "type": "choice",
                    "choice": "simple" | "complex",
                    "confidence": float,
                    "probabilities": {"simple": float, "complex": float},
                }
            },
            "usage": {
                "total_input_tokens": int,
                "total_output_tokens": 0,
                "total_tokens": int,
                "decision_cost_usd": float,
            },
            "routing": {"model": "gemini-decision-1"},
        }
        ```
        """
        api_key = self._get_api_key()
        target_model = model or self.default_model
        active_questions = questions or DECISIONS_COMPLEXITY_QUESTIONS

        if isinstance(input_data, dict):
            serialized_input: str | list[dict[str, Any]] = json.dumps(input_data, ensure_ascii=False)
        else:
            serialized_input = input_data

        decisions_res = self._get_sdk_decisions_resource(api_key)
        if decisions_res is not None and hasattr(decisions_res, "create"):
            sdk_result = decisions_res.create(
                model=target_model,
                input=serialized_input,
                questions=active_questions,
            )
            return self._normalize_sdk_response(sdk_result, target_model)

        return self._call_rest_endpoint(
            api_key=api_key,
            model=target_model,
            serialized_input=serialized_input,
            questions=active_questions,
        )

    def _call_rest_endpoint(
        self,
        *,
        api_key: str,
        model: str,
        serialized_input: str | list[dict[str, Any]],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": model,
                "input": serialized_input,
                "questions": questions,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
                "User-Agent": "decisions-adk-model-router/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_sec, context=self._ssl_ctx) as resp:
                raw_body = resp.read().decode("utf-8")
                data = json.loads(raw_body)
        except urllib.error.HTTPError as http_err:
            err_body = ""
            try:
                err_body = http_err.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"Gemini Decisions API HTTP {http_err.code}: {err_body or http_err.reason}"
            ) from http_err

        return self._normalize_dict_response(data, model)

    @staticmethod
    def _normalize_sdk_response(sdk_result: Any, model: str) -> dict[str, Any]:
        if isinstance(sdk_result, dict):
            return DecisionsClient._normalize_dict_response(sdk_result, model)

        answers_out: dict[str, dict[str, Any]] = {}
        raw_answers = getattr(sdk_result, "answers", None) or {}
        for q_key, ans_obj in raw_answers.items():
            if isinstance(ans_obj, dict):
                answers_out[q_key] = dict(ans_obj)
            else:
                ans_dict: dict[str, Any] = {
                    "type": getattr(ans_obj, "type", "choice"),
                    "confidence": float(getattr(ans_obj, "confidence", 0.0) or 0.0),
                }
                if getattr(ans_obj, "choice", None) is not None:
                    ans_dict["choice"] = str(ans_obj.choice)
                if getattr(ans_obj, "probabilities", None) is not None:
                    ans_dict["probabilities"] = dict(ans_obj.probabilities)
                if getattr(ans_obj, "noul", None) is not None:
                    ans_dict["noul"] = float(ans_obj.noul)
                if getattr(ans_obj, "score", None) is not None:
                    ans_dict["score"] = float(ans_obj.score)
                if getattr(ans_obj, "legend", None) is not None:
                    ans_dict["legend"] = dict(ans_obj.legend)
                answers_out[q_key] = ans_dict

        raw_usage = getattr(sdk_result, "usage", None)
        if isinstance(raw_usage, dict):
            in_tok = int(raw_usage.get("total_input_tokens", 0) or 0)
            out_tok = int(raw_usage.get("total_output_tokens", 0) or 0)
            tot_tok = int(raw_usage.get("total_tokens", in_tok + out_tok) or 0)
        else:
            in_tok = int(getattr(raw_usage, "total_input_tokens", 0) or 0)
            out_tok = int(getattr(raw_usage, "total_output_tokens", 0) or 0)
            tot_tok = int(getattr(raw_usage, "total_tokens", in_tok + out_tok) or 0)

        cost_usd = (in_tok / 1_000_000.0) * DECISION_INPUT_PRICE_PER_1M
        return {
            "answers": answers_out,
            "usage": {
                "total_input_tokens": in_tok,
                "total_output_tokens": out_tok,
                "total_tokens": tot_tok,
                "decision_cost_usd": round(cost_usd, 8),
            },
            "routing": {"model": model},
        }

    @staticmethod
    def _normalize_dict_response(data: dict[str, Any], model: str) -> dict[str, Any]:
        answers_out: dict[str, dict[str, Any]] = {}
        for q_key, ans_val in (data.get("answers") or {}).items():
            if isinstance(ans_val, dict):
                answers_out[q_key] = dict(ans_val)
        usage_raw = data.get("usage") or {}
        in_tok = int(usage_raw.get("total_input_tokens", 0) or 0)
        out_tok = int(usage_raw.get("total_output_tokens", 0) or 0)
        tot_tok = int(usage_raw.get("total_tokens", in_tok + out_tok) or 0)
        cost_usd = (in_tok / 1_000_000.0) * DECISION_INPUT_PRICE_PER_1M
        return {
            "answers": answers_out,
            "usage": {
                "total_input_tokens": in_tok,
                "total_output_tokens": out_tok,
                "total_tokens": tot_tok,
                "decision_cost_usd": round(cost_usd, 8),
            },
            "routing": {"model": model},
        }


decisions_client = DecisionsClient()


# 1. Standalone ADK before_model_callback using Google Gemini Decisions API (`gemini-decision-1`)
async def decisions_router(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    *,
    client: DecisionsClient | Any | None = None,
    api_key: str | None = None,
    model: str = DECISION_MODEL,
    questions: dict[str, dict[str, Any]] | None = None,
    confidence_threshold: float = 0.7,
    simple_model: str = SIMPLE_MODEL,
    complex_model: str = COMPLEX_MODEL,
    exclude_agents: set[str] | list[str] | tuple[str, ...] | None = None,
) -> LlmResponse | None:
    """Standalone ADK `before_model_callback` using Google's Gemini Decisions API (`gemini-decision-1`).

    Args:
        callback_context: ADK callback context.
        llm_request: Outgoing ADK LLM request whose `.model` will be dynamically set.
        client: Optional `DecisionsClient` instance.
        api_key: Optional explicit `GEMINI_API_KEY` for the Decisions API (`v1beta`).
        model: Decisions API model ID (default: `"gemini-decision-1"`).
        questions: Optional custom Decisions API question map (defaults to `DECISIONS_COMPLEXITY_QUESTIONS`).
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

    active_client = client or (DecisionsClient(api_key=api_key, model=model) if api_key else decisions_client)
    active_questions = questions or DECISIONS_COMPLEXITY_QUESTIONS

    try:
        t0 = time.perf_counter()
        res = await asyncio.to_thread(
            active_client.predict,
            eval_state,
            active_questions,
            model=model,
        )
        decisions_latency_ms = (time.perf_counter() - t0) * 1000.0
        ans = res["answers"]["complexity"]
        if ans.get("type") == "noul" or "noul" in ans:
            noul_prob = float(ans.get("noul", 0.5))
            choice = "complex" if noul_prob >= 0.5 else "simple"
            confidence = float(ans.get("confidence", max(noul_prob, 1.0 - noul_prob)))
            probabilities = {"complex": round(noul_prob, 4), "simple": round(1.0 - noul_prob, 4)}
        else:
            choice = str(ans["choice"])
            confidence = float(ans.get("confidence", ans.get("answer_confidence", 0.0)))
            probabilities = dict(ans.get("probabilities") or {})
        usage = dict(res.get("usage") or {})
        input_tokens = int(usage.get("total_input_tokens", 0) or 0)
        decision_cost_usd = float(usage.get("decision_cost_usd", 0.0) or 0.0)
    except Exception as exc:
        logger.warning(
            "[DecisionsRouter] Gemini Decisions API evaluation failed (%s); falling back to existing/default model.",
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
            "[DecisionsRouter] Upgraded turn to %s (choice=%s, conf=%.2f, decision_model=%s, in_tok=%d, cost=$%.6f, latency=%.1fms%s)",
            complex_model,
            choice,
            confidence,
            model,
            input_tokens,
            decision_cost_usd,
            decisions_latency_ms,
            f", agent={agent_name} pinned to {simple_model}" if is_excluded_agent else "",
        )
    else:
        logger.info(
            "[DecisionsRouter] Routed turn to %s (choice=%s, conf=%.2f, decision_model=%s, in_tok=%d, cost=$%.6f, latency=%.1fms)",
            simple_model,
            choice,
            confidence,
            model,
            input_tokens,
            decision_cost_usd,
            decisions_latency_ms,
        )

    llm_request.model = simple_model if is_excluded_agent else routed_model

    callback_context.state["routed_model"] = routed_model
    callback_context.state["last_routed_user_q"] = user_q
    if invocation_id:
        callback_context.state["last_routed_invocation_id"] = invocation_id

    routing_metadata = {
        "routed_model": routed_model,
        "choice": choice,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "decision_model": model,
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "decision_cost_usd": round(decision_cost_usd, 8),
        "decisions_latency_ms": round(decisions_latency_ms, 2),
        "excluded_agents": sorted(excluded_set),
    }
    callback_context.state["decisions_routing"] = routing_metadata
    return None


# 2. ADK Plugin wrapper around Google Gemini Decisions API
class DecisionsModelRouterPlugin(BasePlugin):
    """ADK Plugin wrapper around Google's Gemini Decisions API (`gemini-decision-1`)."""

    def __init__(
        self,
        client: DecisionsClient | Any | None = None,
        api_key: str | None = None,
        model: str = DECISION_MODEL,
        questions: dict[str, dict[str, Any]] | None = None,
        confidence_threshold: float = 0.7,
        simple_model: str = SIMPLE_MODEL,
        complex_model: str = COMPLEX_MODEL,
        exclude_agents: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(name="decisions_model_router_plugin")
        self._client = client or (DecisionsClient(api_key=api_key, model=model) if api_key else decisions_client)
        self._api_key = api_key
        self._model = model
        self._questions = questions
        self._confidence_threshold = confidence_threshold
        self._simple_model = simple_model
        self._complex_model = complex_model
        self._exclude_agents = set(exclude_agents) if exclude_agents else set()

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        return await decisions_router(
            callback_context=callback_context,
            llm_request=llm_request,
            client=self._client,
            api_key=self._api_key,
            model=self._model,
            questions=self._questions,
            confidence_threshold=self._confidence_threshold,
            simple_model=self._simple_model,
            complex_model=self._complex_model,
            exclude_agents=self._exclude_agents,
        )
