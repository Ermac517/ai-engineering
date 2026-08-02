# AI Engineering

A personal learning repo for an AI Engineering course, organized as Jupyter notebooks in three parts:

| Part | Focus |
| --- | --- |
| [AI Engineering Part 1](AI%20Engineering%20Part%201/) | LLM fundamentals, prompting, tool calling, RAG, Gradio |
| [AI Engineering Part 2](AI%20Engineering%20Part%202/) | Agents (hand-rolled and OpenAI Agents SDK) + evals |
| [AI Engineering Part 3 MCP](AI%20Engineering%20Part%203%20MCP/) | Model Context Protocol servers and agents |

There is no package, no test suite, no linter, and no build step — the deliverable of each lesson is a
runnable notebook. Commits follow `Week N Day M - Topic`.

## Getting started

Two virtualenvs live at the repo root (both gitignored, so create them yourself on a fresh clone):

- `ai_env` — **Python 3.14.3, the current default.** Jupyter kernel display name `ai_env (3.14.3.final.0)`.
- `ai_env_3_12` — Python 3.12.10, kept because three notebooks still pin a 3.12 kernel:
  `rag.ipynb`, `digital-twin.ipynb`, `digital-twin-arch4-full-RAG.ipynb`.

```bash
source ai_env/bin/activate
jupyter lab                 # or: jupyter notebook
```

Both envs have `openai`, `openai-agents`, `mcp`, `chromadb`, `gradio`, `litellm`, `ddgs`,
`trafilatura`, `plotly`, and `scikit-learn` installed. There is no root `requirements.txt` —
notebooks install extras inline (`!pip install ddgs trafilatura -q`).

### Secrets

Create a `.env` at the repo root (gitignored):

```bash
OPENAI_API_KEY=sk-...
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
```

Every notebook's first cell calls `load_dotenv()` and raises if `OPENAI_API_KEY` is missing.

### Extra requirements for Part 3

Part 3 launches MCP servers as subprocesses, so these binaries must be on `PATH`:

- `npx` / `node` — for `@modelcontextprotocol/server-filesystem`
- `uvx` — for `mcp-server-fetch`

## Notebook conventions

Cells are grouped under `### Step N:` markdown headings. Step 1 is always imports + `load_dotenv()` +
`client = OpenAI(api_key=OPENAI_API_KEY)`. Most notebooks define `MODEL = "gpt-4.1-mini"`; evals use
`JUDGE_MODEL = "gpt-4.1"`.

## Part 1 — LLM, RAG, and Gradio fundamentals

| Notebook | What it covers |
| --- | --- |
| [hello-world.ipynb](AI%20Engineering%20Part%201/hello-world.ipynb) | First Chat Completions call |
| [system-vs-user-prompt.ipynb](AI%20Engineering%20Part%201/system-vs-user-prompt.ipynb) | Role of each message type |
| [conversation-history.ipynb](AI%20Engineering%20Part%201/conversation-history.ipynb) | Multi-turn state |
| [caching.ipynb](AI%20Engineering%20Part%201/caching.ipynb) | Prompt caching |
| [ai-wars.ipynb](AI%20Engineering%20Part%201/ai-wars.ipynb) | Model-vs-model comparison |
| [gradio.ipynb](AI%20Engineering%20Part%201/gradio.ipynb) | Chat UIs with Gradio |
| [tool-calling.ipynb](AI%20Engineering%20Part%201/tool-calling.ipynb) | Hand-rolled tool calling |
| [rag.ipynb](AI%20Engineering%20Part%201/rag.ipynb) | Chunking → embeddings → ChromaDB retrieval |
| [my-post-generator.ipynb](AI%20Engineering%20Part%201/my-post-generator.ipynb) | Applied writing assistant |
| [extreme-coding-challenge.ipynb](AI%20Engineering%20Part%201/extreme-coding-challenge.ipynb) | Line-by-line walkthrough exercise |

### RAG pipeline

`rag.ipynb` and the digital-twin line share one pipeline: `split_text_into_chunks()` (character
chunking that walks the cut backward to a paragraph → newline → sentence → space boundary, only past
the halfway point) → `client.embeddings.create(model="text-embedding-3-small")` → ChromaDB. The
collection is emptied before re-adding (`collection.delete(collection.get()["ids"])`) so cells are
idempotent. Retrieval concatenates the top-3 documents and appends them to the **system** message as
`"\n\nContext:\n" + context`.

The persistent Chroma dirs (`chroma_db/`, `chroma_db3/`, `chroma_db_twin/`) are gitignored generated
artifacts — rebuild them by re-running the notebook.

### The digital twin

`digital-twin-arch1` → `arch4` are progressive versions of the same app:

1. [arch1 — dynamic context injection](AI%20Engineering%20Part%201/digital-twin-arch1-dynamic-context.ipynb)
2. [arch2 — basic tool calling](AI%20Engineering%20Part%201/digital-twin-arch2-basic-tool-calling.ipynb)
3. [arch3 — proper tool calling](AI%20Engineering%20Part%201/digital-twin-arch3-proper-tool-calling.ipynb)
4. [arch4 — full RAG](AI%20Engineering%20Part%201/digital-twin-arch4-full-RAG.ipynb)

[digital-twin.ipynb](AI%20Engineering%20Part%201/digital-twin.ipynb) is the consolidated final version.
The numbered `archX` snapshots are frozen teaching steps — change the consolidated notebook and
[digital-twin/app.py](AI%20Engineering%20Part%201/digital-twin/app.py) instead.

[digital-twin/](AI%20Engineering%20Part%201/digital-twin/) is the Hugging Face Space deployment bundle
(`app.py`, `requirements.txt`, `mcruz.jpeg` avatar). `app.py` is a flattened copy of the notebook with
two deliberate differences: it reads secrets via `os.getenv` without `load_dotenv` (Space secrets), and
it uses an **in-memory** `chromadb.Client()` instead of `PersistentClient`, so the index is rebuilt on
every Space boot. Keep the two in sync when changing prompts, tools, or documents.
[my-first-space/](AI%20Engineering%20Part%201/my-first-space/) is the trivial Gradio Space used to learn
deployment.

## Part 2 — Agents

| Notebook | What it covers |
| --- | --- |
| [research-agent.ipynb](AI%20Engineering%20Part%202/research-agent.ipynb) | Hand-rolled research agent |
| [research-agent-sdk.ipynb](AI%20Engineering%20Part%202/research-agent-sdk.ipynb) | Same agent, OpenAI Agents SDK |
| [research-agent-evals.ipynb](AI%20Engineering%20Part%202/research-agent-evals.ipynb) | LLM-as-judge evals |
| [dice-roll-agent.ipynb](AI%20Engineering%20Part%202/dice-roll-agent.ipynb) | Minimal SDK agent |
| [article-writer.ipynb](AI%20Engineering%20Part%202/article-writer.ipynb) | Orchestrator + handoffs |

### Two parallel agent stacks

The repo deliberately builds the same capabilities twice — once by hand, once with the SDK.

**Hand-rolled tool calling** (Part 1 `tool-calling.ipynb`, `digital-twin*`, Part 2 `research-agent.ipynb`,
`research-agent-evals.ipynb`, and `digital-twin/app.py`):

- `tools = []`, then one JSON-schema dict per function appended as `{"type": "function", "function": {...}}`.
- `handle_tool_call(tool_calls)` dispatches on `function_name` via if/elif and returns a list of
  `{"role": "tool", "content": ..., "tool_call_id": ...}` messages.
- The agentic loop is `while message.tool_calls:` — append the assistant message, extend with tool
  results, re-call the API — always with an iteration/message cap to prevent runaway loops.

**OpenAI Agents SDK** (Part 2 `research-agent-sdk.ipynb`, `article-writer.ipynb`, all of Part 3):
`from agents import Agent, Runner, function_tool, handoff, trace`, then
`await Runner.run(agent, input=..., max_turns=N)`. `article-writer.ipynb` is the orchestrator/handoff
example — research and image agents exposed as tools, three writer personas as handoff targets.

### Evals

`research-agent-evals.ipynb` is the LLM-as-judge pattern: a binary TRUE/FALSE judge prompt scored with
`temperature=0, max_tokens=1`, run across a fixed `TOPICS` list, with agent stdout suppressed via
`contextlib.redirect_stdout`. "Evals Run 1" and "Evals Run 2" differ only in the agent's system prompt —
the notebook exists to compare failure rates between prompt revisions, so both runs are kept.

## Part 3 — MCP

| Notebook | What it covers |
| --- | --- |
| [mcp-fetch-intro.ipynb](AI%20Engineering%20Part%203%20MCP/mcp-fetch-intro.ipynb) | First MCP server (`mcp-server-fetch`) |
| [mcp-filesystem.ipynb](AI%20Engineering%20Part%203%20MCP/mcp-filesystem.ipynb) | Filesystem server, sandboxing, and multiple servers on one agent |

Servers are launched as
`async with MCPServerStdio(name=..., params=..., client_session_timeout_seconds=60) as server:` and
passed to `Agent(..., mcp_servers=[...])` — the agent must be constructed *inside* the context manager.
`SANDBOX_DIR = os.path.abspath("secret_project_007")`, since the filesystem server needs an absolute
allowed directory. The sandbox dirs (`secret_project_006/`, `secret_project_007/`) are gitignored and
are created plus seeded with sample files by the notebook itself.

`mcp-filesystem.ipynb` keeps a running `# Errors to debug:` comment block above successive versions of
`SUMMARIZER_AGENT_PROMPT`, each version fixing one observed agent failure — the iteration is the lesson,
so that history is preserved.

## Repo layout

```
AI Engineering Part 1/      LLM/RAG/Gradio notebooks + digital-twin & my-first-space Spaces
AI Engineering Part 2/      Agent notebooks + evals
AI Engineering Part 3 MCP/  MCP notebooks (+ gitignored sandbox dirs)
CLAUDE.md                   Guidance for Claude Code
.env                        Secrets (gitignored)
ai_env/, ai_env_3_12/       Virtualenvs (gitignored)
```
