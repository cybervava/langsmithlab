"""Run the notebook demos locally. Traces go to LangSmith when LANGSMITH_TRACING=true.

    uv run langsmithlab summarize --strategy tokens
    uv run langsmithlab hitl --decision edit
"""

import argparse
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langsmithlab.agents import SUMMARIZATION_LIMITS, build_email_agent, build_summarization_agent

CITIES = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]
QUESTIONS = {
    "messages": ["What is 2+2?", "What is 10*5?", "What is 100/4?",
                 "What is 15-7?", "What is 3*3?", "What is 4*4?"],
    "tokens": [f"Find hotels in {city}" for city in CITIES],
    "fraction": [f"Hotels in {city}" for city in CITIES],
}
EMAIL_REQUEST = "Send email to wrong@email.com with subject 'Test' and body 'Hello'"
EDITED_EMAIL = {
    "recipient": "correct@email.com",
    "subject": "Corrected Subject",
    "body": "This was edited by human before sending",
}


def make_config(run_name, tags):
    # thread_id in metadata groups the turns in LangSmith's Threads view.
    thread_id = f"{run_name}-{uuid.uuid4().hex[:8]}"
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": run_name,
        "tags": tags,
        "metadata": {"thread_id": thread_id},
    }


def run_summarization(strategy, model=None):
    agent = build_summarization_agent(strategy, model=model, checkpointer=InMemorySaver())
    config = make_config(f"summarization-{strategy}", ["summarization", strategy])
    for question in QUESTIONS[strategy]:
        messages = agent.invoke({"messages": [HumanMessage(question)]}, config)["messages"]
        summarized = any(m.additional_kwargs.get("lc_source") == "summarization" for m in messages)
        note = "  <- history summarized" if summarized else ""
        print(f"{question:<28} {len(messages):>2} messages in state{note}")
    return messages


def make_decision(kind, action):
    if kind == "edit":
        return {"type": "edit", "edited_action": {"name": action["name"], "args": EDITED_EMAIL}}
    return {"type": kind}


def run_hitl(decision, model=None):
    agent = build_email_agent(model=model, checkpointer=InMemorySaver())
    config = make_config(f"email-hitl-{decision}", ["hitl", decision])
    result = agent.invoke({"messages": [HumanMessage(EMAIL_REQUEST)]}, config)
    while "__interrupt__" in result:
        actions = result["__interrupt__"][0].value["action_requests"]
        for action in actions:
            print(f"Paused for approval: {action['name']}({action['args']}) -> {decision}")
        decisions = [make_decision(decision, action) for action in actions]
        result = agent.invoke(Command(resume={"decisions": decisions}), config)
    print(f"Agent: {result['messages'][-1].content}")
    return result["messages"]


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(prog="langsmithlab")
    sub = parser.add_subparsers(dest="command", required=True)
    summarize = sub.add_parser("summarize", help="SummarizationMiddleware demo")
    summarize.add_argument("--strategy", choices=list(SUMMARIZATION_LIMITS), default="messages")
    hitl = sub.add_parser("hitl", help="HumanInTheLoopMiddleware demo")
    hitl.add_argument("--decision", choices=["approve", "edit", "reject"], default="approve")
    args = parser.parse_args(argv)

    try:
        if args.command == "summarize":
            run_summarization(args.strategy)
        else:
            run_hitl(args.decision)
    finally:
        wait_for_all_tracers()  # flush traces before the process exits


if __name__ == "__main__":
    main()
