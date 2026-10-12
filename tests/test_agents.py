"""Offline tests: a fake chat model stands in for OpenAI, so no API keys are needed."""

import itertools

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from langsmithlab import demo


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def send_email_then_reply():
    call = {"name": "send_email", "id": "call-1",
            "args": {"recipient": "wrong@email.com", "subject": "Test", "body": "Hello"}}
    return FakeToolModel(messages=iter([AIMessage("", tool_calls=[call]), AIMessage("Done.")]))


def test_summarization_replaces_history_with_summary():
    model = FakeToolModel(messages=(AIMessage(f"answer {i}") for i in itertools.count()))
    messages = demo.run_summarization("messages", model=model)
    assert messages[0].additional_kwargs.get("lc_source") == "summarization"
    assert len(messages) < 2 * len(demo.QUESTIONS["messages"])


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("approve", "Email sent to wrong@email.com"),
        ("edit", "Email sent to correct@email.com"),
        ("reject", "User rejected the tool call"),
    ],
)
def test_hitl_decisions(decision, expected):
    messages = demo.run_hitl(decision, model=send_email_then_reply())
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert expected in tool_message.content
    assert (tool_message.status == "error") == (decision == "reject")


def test_graphs_load_for_langgraph_dev():
    from langsmithlab import graphs

    for name in ["summarization_messages", "summarization_tokens",
                 "summarization_fraction", "email_assistant"]:
        assert getattr(graphs, name).checkpointer is None
