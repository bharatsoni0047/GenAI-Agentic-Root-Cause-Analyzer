# RCA Agent — AI Root-Cause Analysis (LangGraph + RAG)

Send an error message to a REST API, get back a structured root-cause analysis —
grounded in your past incident history via retrieval-augmented generation.

```json
POST /rca  {"error": "Database connection timeout on user service"}

→ {
    "root_cause": "Connection pool exhausted under peak load...",
    "impact": "User data requests fail application-wide...",
    "fix": "Increase pool size, add query timeouts...",
    "confidence": 0.85
  }
```

## How it works

```
POST /rca ──▶ retrieve ──────────▶ analyze ─────────────▶ RCAReport (JSON)
              Chroma vector        LLM with structured
              search over past     output (Pydantic-
              incidents            validated)
```

| Component | Choice | Swap via |
|---|---|---|
| Orchestration | LangGraph 1.x | — |
| LLM | Groq `openai/gpt-oss-120b` | `RCA_LLM_MODEL` (e.g. `openai:gpt-4o`) |
| Embeddings | `all-MiniLM-L6-v2` (local, free) | `RCA_EMBEDDING_MODEL` |
| Vector DB | Chroma (embedded) | `RCA_CHROMA_DIR` |
| API | FastAPI | — |

Structured output forces the LLM to return a valid `RCAReport` on every call —
no parsing, no retry loops.

## Quickstart

```bash
git clone https://github.com/bharatsoni0047/GenAI-Agentic-Root-Cause-Analyzer.git
cd GenAI-Agentic-Root-Cause-Analyzer

python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env          # then add your GROQ_API_KEY (free: console.groq.com)

python -m rca_agent.ingest      # one-time: index past incidents into Chroma
uvicorn rca_agent.app:app
```

Open http://127.0.0.1:8000/docs for the interactive API.

## Project structure

```
data/error_logs.txt    past incidents (the RAG knowledge base)
rca_agent/
  config.py            all settings — env-driven, one place
  schemas.py           API contract + structured report model
  ingest.py            chunk + embed + index incidents
  graph.py             LangGraph pipeline: retrieve → analyze
  app.py               FastAPI service (/rca, /health)
```

## Extending

Add more incidents to `data/error_logs.txt` (or point `RCA_LOGS_PATH` at your own
log export) and re-run `python -m rca_agent.ingest` — analysis quality scales with
the incident history you feed it.
