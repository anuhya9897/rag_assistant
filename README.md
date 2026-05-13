# RAG POC

Retrieval-augmented generation over a local knowledge base (DuckDB + sentence-transformers + Ollama).

## Project structure

```
rag-poc/
├── backend/                 # REST API (FastAPI)
│   ├── app/
│   │   ├── main.py         # API routes
│   │   └── rag_service.py  # RAG logic (retrieve + Ollama)
│   ├── run.py              # Run API server
│   └── requirements.txt
├── frontend/               # Web UI
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── build_index_duckdb_local_incremental.py  # Index KB into DuckDB
├── ask.py                  # CLI: ask questions
├── rag_ollama_duckdb.py    # Experimental RAG CLI
├── kb.duckdb               # Created after indexing
└── data/Knowledgebase/     # Your documents
```

## Quick start

### 1. Index the knowledge base

```bash
python build_index_duckdb_local_incremental.py
```

### 2. Start the API

From the project root:

```bash
pip install -r backend/requirements.txt
python backend/run.py
```

API runs at **http://localhost:8000**. Docs: http://localhost:8000/docs

### 3. Open the UI

- **Option A:** Open `frontend/index.html` in a browser (file://). Ensure the API is running so CORS works.
- **Option B:** Serve the frontend, e.g. `npx serve frontend` then open http://localhost:3000

The UI sends requests to `http://localhost:8000` by default. Ask a question and view the answer and sources.

## CLI (no API)

### Ask questions

```bash
python ask.py "your question here" --k 5
```

### Experimental RAG

```bash
python rag_ollama_duckdb.py "your question here"
```

## API endpoints

| Method | Path         | Description                    |
|--------|--------------|--------------------------------|
| GET    | /api/health  | Health check                   |
| POST   | /api/ask     | Ask a question (JSON body)     |

**POST /api/ask** body:

```json
{
  "question": "How do I configure X?",
  "k": 5,
  "model": "llama3.1:8b"
}
```

Response: `{ "answer": "...", "sources": [...] }`
