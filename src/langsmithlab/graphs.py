"""Graphs served by `langgraph dev` (LangSmith Studio) via langgraph.json.

No checkpointer here: the Agent Server provides persistence and rejects custom ones.
"""

from langsmithlab.agents import build_email_agent, build_summarization_agent

summarization_messages = build_summarization_agent("messages")
summarization_tokens = build_summarization_agent("tokens")
summarization_fraction = build_summarization_agent("fraction")
email_assistant = build_email_agent()
