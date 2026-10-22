from google.adk.agents import Agent
from google.adk.apps import App

from .plugin import (
    COMPLEX_MODEL,
    SIMPLE_MODEL,
    LayaModelRouterPlugin,
    laya_engine,
    laya_router,
)
from .questions import LAYA_COMPLEXITY_QUESTIONS

# Define the ADK Agent and attach LayaModelRouterPlugin to the App
root_agent = Agent(
    name="root_agent",
    model=COMPLEX_MODEL,  # Default fallback
    instruction="You are an AI assistant.",
)

app = App(
    name="laya_adk_model_router",
    root_agent=root_agent,
    plugins=[LayaModelRouterPlugin()],
)

__all__ = [
    "COMPLEX_MODEL",
    "LAYA_COMPLEXITY_QUESTIONS",
    "LayaModelRouterPlugin",
    "SIMPLE_MODEL",
    "app",
    "laya_engine",
    "laya_router",
    "root_agent",
]
