# RAG POC – Backend API

FastAPI REST API that exposes the RAG pipeline (retrieve from DuckDB + generate with Ollama).

## Run

From **project root** (rag-poc):

```bash
pip install -r backend/requirements.txt
python backend/run.py
```

Or with uvicorn directly:

```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- OpenAPI docs: http://localhost:8000/docs  

## Endpoints

- **GET /api/health** – Health check
- **POST /api/ask** – Ask a question; body: `{ "question": "...", "k": 5, "model": "llama3.1:8b" }`

The API uses `kb.duckdb` and the embedder/Ollama stack from the existing RAG scripts. Build the index first with `build_index_duckdb_local_incremental.py`.
