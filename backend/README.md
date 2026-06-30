# Backend — Simba API

FastAPI service for **Simba** (RAG over `kb.duckdb`): retrieval, chat completion (Ollama / Azure OpenAI), health, model list, and index rebuild.

## Run

From the **repository root** (recommended):

```bash
python backend/run.py
```

- API: [http://localhost:8000](http://localhost:8000)  
- UI (static): [http://localhost:8000/ui/](http://localhost:8000/ui/)  
- OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

`run.py` changes cwd to the project root, loads `.env`, and starts uvicorn.

## Layout

| Path | Role |
|------|------|
| `run.py` | Entry point |
| `app/main.py` | Routes, CORS, `/ui` mount, `POST /api/reindex` |
| `app/rag_service.py` | DuckDB retrieval, embeddings, prompts (**Simba**), Ollama chat |
| `app/openai_llm.py` | Azure / OpenAI chat completions |
| `app/eval_metrics.py` | Metrics used by `scripts/eval_rag.py` |
| `requirements.txt` | FastAPI, uvicorn, duckdb, ollama, openai, … |

## Install

```bash
pip install -r backend/requirements.txt
pip install -r ../requirements.txt   # indexing deps if you run reindex from API
```

## Configuration

See root [README.md](../README.md) for `.env`, `RAG_EMBED_PROVIDER`, `RAG_OLLAMA_EMBED_MODEL`, Azure OpenAI, and reindex options.

## Main endpoints

- `GET /api/health` — KB, chunk count, Ollama/Azure status  
- `GET /api/models` — UI model dropdown  
- `POST /api/ask`, `POST /api/ask/stream` — Q&A with sources  
- `POST /api/reindex` — rebuild `kb.duckdb` (local folder or `.zip` path)  
- `POST /api/models/ensure` — pull Ollama model from UI **Refresh**
