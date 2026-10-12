# langsmithlab

The agents from `notebooks/6-middleware.ipynb` (LangChain v1 middleware), packaged
so they can be traced in **LangSmith** and run interactively in **LangSmith Studio**.

- **SummarizationMiddleware**: compresses chat history when a `messages`, `tokens`, or `fraction` trigger is hit
- **HumanInTheLoopMiddleware**: pauses before `send_email` so a human can approve, edit, or reject the call

## Setup

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY and LANGSMITH_API_KEY
```

## 1. Trace the demos in LangSmith

```bash
uv run langsmithlab summarize --strategy messages   # or tokens / fraction
uv run langsmithlab hitl --decision edit            # or approve / reject
```

The runs show up in the `LANGSMITH_PROJECT` project (default `langsmithlab`). Each one is
tagged with its demo name, and the turns of a conversation are grouped under one thread
in the **Threads** tab. Open a summarization trace to see the extra
`SummarizationMiddleware.before_model` step and the summary model call.

## 2. Run the agents in LangSmith Studio

```bash
uv run langgraph dev
```

This starts a local Agent Server and opens Studio. Four graphs are available (see
`langgraph.json`): `summarization_messages`, `summarization_tokens`,
`summarization_fraction`, and `email_assistant`. With `email_assistant`, ask it to send
an email. The run pauses at the interrupt, and you can approve, edit, or reject it from
the Studio UI.

The same `langgraph.json` works for deploying to LangSmith Deployments.

## Tests

```bash
uv run pytest
```

The tests use a fake chat model, so they need no API keys and make no network calls.

## Layout

```
src/langsmithlab/agents.py   tools + agent builders
src/langsmithlab/graphs.py   graphs exported for langgraph.json
src/langsmithlab/demo.py     CLI demos (langsmithlab summarize | hitl)
tests/test_agents.py         offline tests
notebooks/                   original notebook
```
