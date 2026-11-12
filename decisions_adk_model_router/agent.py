from google.adk.agents import Agent
from google.adk.apps import App

from .plugin import (
    COMPLEX_MODEL,
    DECISION_MODEL,
    SIMPLE_MODEL,
    DecisionsClient,
    DecisionsModelRouterPlugin,
    decisions_client,
    decisions_router,
)
from .questions import DECISIONS_COMPLEXITY_QUESTIONS

# Define the ADK Agent and attach DecisionsModelRouterPlugin to the App
root_agent = Agent(
    name="root_agent",
    model=COMPLEX_MODEL,  # Default fallback
    instruction="You are an AI assistant.",
)

app = App(
    name="decisions_adk_model_router",
    root_agent=root_agent,
    plugins=[DecisionsModelRouterPlugin()],
)

__all__ = [
    "COMPLEX_MODEL",
    "DECISIONS_COMPLEXITY_QUESTIONS",
    "DECISION_MODEL",
    "DecisionsClient",
    "DecisionsModelRouterPlugin",
    "SIMPLE_MODEL",
    "app",
    "decisions_client",
    "decisions_router",
    "root_agent",
]
