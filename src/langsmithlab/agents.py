"""Agents from the 6-middleware notebook: summarization and human-in-the-loop."""

import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware
from langchain_core.tools import tool

DEFAULT_MODEL = "openai:gpt-4o-mini"

# Tiny thresholds so summarization fires within a short demo.
# Fractions are relative to the model's context window (gpt-4o-mini: 128k tokens).
SUMMARIZATION_LIMITS = {
    "messages": {"trigger": ("messages", 10), "keep": ("messages", 4)},
    "tokens": {"trigger": ("tokens", 550), "keep": ("tokens", 200)},
    "fraction": {"trigger": ("fraction", 0.005), "keep": ("fraction", 0.002)},
}


@tool
def search_hotels(city: str) -> str:
    """Search hotels in a city."""
    return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night, business center
    3. Budget Stay - 3 star, $75/night, free wifi"""


@tool
def read_email(email_id: str) -> str:
    """Read an email by its ID."""
    return f"Email content for ID: {email_id}"


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient with a subject and body."""
    return f"Email sent to {recipient} with subject '{subject}'"


def _model(model):
    return model or os.getenv("AGENT_MODEL", DEFAULT_MODEL)


def build_summarization_agent(strategy="messages", model=None, checkpointer=None):
    """Agent whose history is summarized once the strategy's trigger is reached."""
    model = _model(model)
    return create_agent(
        model,
        tools=[] if strategy == "messages" else [search_hotels],
        checkpointer=checkpointer,
        middleware=[SummarizationMiddleware(model=model, **SUMMARIZATION_LIMITS[strategy])],
        name=f"summarization_{strategy}",
    )


def build_email_agent(model=None, checkpointer=None):
    """Agent that pauses before send_email for an approve / edit / reject decision."""
    return create_agent(
        _model(model),
        tools=[read_email, send_email],
        system_prompt="You are an email assistant. If a send is rejected, do not retry.",
        checkpointer=checkpointer,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                    "read_email": False,
                }
            )
        ],
        name="email_assistant",
    )
