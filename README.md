# Simba — Knowledge Base Assistant (RAG POC)

**Simba** is a retrieval-augmented generation assistant for RedMane: documents are chunked, embedded, and stored in **`kb.duckdb`** (DuckDB). Questions are embedded, similar chunks are retrieved, and an **Ollama** or **OpenAI/Azure GPT** chat model generates answers with cited sources.

Default stack: **DuckDB** + **Ollama** embeddings (query + index must use the same provider/dimensions) + **Ollama** or **OpenAI/Azure GPT** for answers (e.g. `llama3.1:8b` or `gpt-4o-mini`). Optional **sentence-transformers** embeddings are gated by environment variables (see below).

The web UI at **`/ui/`** uses RedMane branding; the assistant identifies itself as **Simba** in prompts and health/chat responses.

## Prerequisites

- **Python** 3.10+ recommended  
- **[Ollama](https://ollama.com/)** installed and running (`ollama serve` / app running)  
- At least one **chat** model pulled, e.g. `ollama pull llama3.1:8b`  
- For indexing with the default provider, an **embedding** model: e.g. `ollama pull nomic-embed-text` (or rely on auto-selection if another embed-capable model is already installed — see `backend/app/rag_service.py`)  
- For **`.docx`** in the knowledge base: **`docx2txt`** (included in root `requirements.txt`)

## Project structure

```
rag_assistant/
├── backend/                    # FastAPI REST API
│   ├── app/
│   │   ├── main.py             # Routes, CORS, timing middleware, /ui static mount
│   │   ├── rag_service.py      # Retrieve, prompt, Ollama/OpenAI generate / embed
│   │   ├── openai_llm.py       # Azure/OpenAI chat completions
│   │   └── eval_metrics.py     # Retrieval / answer metrics (scripts/eval_rag.py)
│   ├── run.py                  # uvicorn entry (chdir to project root, loads .env)
│   └── requirements.txt        # Minimal API deps
├── web/                        # Web UI (served by API at /ui/)
│   ├── index.html              # Simba + RedMane header, reindex form
│   ├── styles.css              # RedMane PPT-inspired theme
│   ├── app.js
│   ├── Redmane.png             # Header logo (add to repo as needed)
│   └── Simba.png               # Title image (add to repo as needed)
├── scripts/
│   ├── benchmark_ask.py        # API timing benchmark
│   ├── benchmark_embeddings.py # MiniLM vs mpnet embed timing (local ST cache)
│   └── eval_rag.py             # Retrieval (and optional LLM) eval
├── eval/
│   └── example_gold.json       # Example gold set for eval_rag.py
├── build_index_duckdb_local_incremental.py   # Build / update kb.duckdb
├── kb_source.py                # Resolve KB: local folder/zip, http_zip, http_manifest
├── kb.duckdb                   # Created after indexing (often gitignored)
├── kb_staging/                 # Temp extract for zips (gitignored)
├── requirements.txt            # LlamaIndex, indexing, docx2txt, etc.
├── data/Knowledgebase/         # Default source documents (see data/Knowledgebase/README.md)
├── frontend/                   # Legacy standalone UI (see frontend/README.md)
└── PLAN.md                     # Delivery / roadmap checklist
```

**Folder READMEs:** [backend](backend/README.md) · [web](web/README.md) · [scripts](scripts/README.md) · [eval](eval/README.md) · [frontend](frontend/README.md) · [data/Knowledgebase](data/Knowledgebase/README.md)

## Install dependencies

From the **repository root** (`rag_assistant/`), preferably in a venv:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

`requirements.txt` pulls in **LlamaIndex**, **`docx2txt`** (for Word `.docx`), and packages needed for **`build_index_duckdb_local_incremental.py`**. `backend/requirements.txt` lists the FastAPI stack; installing both avoids missing packages when running the API and scripts together.

## 1. Index the knowledge base

### Default: `data/Knowledgebase/`

Ensure `data/Knowledgebase/` exists and contains documents. Then from the repo root:

```bash
python build_index_duckdb_local_incremental.py
```

This creates or updates **`kb.duckdb`** at the project root. Embedding provider follows **`RAG_EMBED_PROVIDER`** (defaults to **`ollama`** — same as the API). Rebuild the index if you change embed model or dimensions.

### Indexed file types

The indexer reads (recursively):

`.md`, `.markdown`, `.txt`, `.html`, `.htm`, `.pdf`, `.docx`

Chunking defaults: **800** tokens per chunk, **100** overlap (`build_index_duckdb_local_incremental.py`).

### Local folder or `.zip` (`RAG_KB_DIR`)

**`RAG_KB_SOURCE=local`** (default) accepts a folder **or** a **`.zip`** file. Zip archives are extracted under **`kb_staging/`** before indexing (removed after a successful run unless **`RAG_KB_STAGING_KEEP=1`**).

Paths may be absolute or relative to the project root. Surrounding quotes in paths are stripped.

```powershell
# Folder
$env:RAG_KB_DIR="data\Knowledgebase"
python build_index_duckdb_local_incremental.py

# Zip on disk (e.g. exported notices)
$env:RAG_KB_DIR="data\Notices.zip"
python build_index_duckdb_local_incremental.py
```

### Indexing from a server (HTTP)

Set **`RAG_KB_SOURCE`** before running the indexer (requires **`httpx`**, in root **`requirements.txt`**):

| Mode | Purpose | Required env |
|------|---------|----------------|
| **`local`** (default) | Folder or **`.zip`** on disk | Optional **`RAG_KB_DIR`** |
| **`http_zip`** | `GET` a **`.zip`**, extract, then index | **`RAG_KB_HTTP_URL`**, optional **`RAG_KB_STAGING_DIR`** |
| **`http_manifest`** | `GET` JSON manifest, then each file URL | **`RAG_KB_MANIFEST_URL`** — `{"files":[{"url":"...","path":"docs/a.md"}, ...]}` |

Shared HTTP options: **`RAG_KB_HTTP_BEARER`**, **`RAG_KB_HTTP_HEADERS`** (JSON object), **`RAG_KB_HTTP_TIMEOUT`** (default 300s), **`RAG_KB_HTTP_INSECURE=1`** (dev only), **`RAG_KB_STAGING_KEEP=1`**.

```powershell
$env:RAG_KB_SOURCE="http_zip"
$env:RAG_KB_HTTP_URL="https://internal.example.com/knowledge/export.zip"
python build_index_duckdb_local_incremental.py
```

The running API serves **`kb.duckdb`** only. Refresh the index by re-running the build, setting env vars above, or using **Rebuild knowledge index** in the web UI (**`POST /api/reindex`**).

## 2. Start the API (this also serves the UI)

From the repo root:

```bash
python backend/run.py
```

- **API base:** [http://localhost:8000](http://localhost:8000) (redirects to `/ui/`)  
- **Web UI:** [http://localhost:8000/ui/](http://localhost:8000/ui/)  
- **OpenAPI docs:** [http://localhost:8000/docs](http://localhost:8000/docs)  
- **Health:** [http://localhost:8000/api/health](http://localhost:8000/api/health) — KB presence, chunk count, Ollama/embedder status  

Optional environment variables:

- **`RAG_WARMUP_EMBEDDER=1`** — load the embedder at startup (otherwise first request may be slower).  
- **`RAG_UVICORN_RELOAD=1`** — auto-reload on code changes.  
- **`RAG_EMBED_PROVIDER`** — `ollama` (default) or `st` (sentence-transformers only if **`RAG_ALLOW_ST_EMBEDDINGS=1`**).  
- **`RAG_OLLAMA_EMBED_MODEL`** — pin Ollama embedding model name (e.g. `nomic-embed-text`).  
- **`RAG_UI_REINDEX=0`** — disable **Rebuild knowledge index** in the UI / **`POST /api/reindex`**.  
- **`RAG_REINDEX_TIMEOUT_SEC`** — max seconds for one index subprocess (default **3600**, clamped 60–86400).

### OpenAI / GPT models (chat only)

Embeddings and indexing stay on **Ollama** (or ST) by default; GPT is used only to **generate answers** after retrieval.

**Standard OpenAI:**

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:RAG_OPENAI_MODELS="gpt-4o-mini,gpt-4o"
```

**Azure OpenAI** (deployment names in `RAG_OPENAI_MODELS`):

```powershell
$env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY="..."
$env:AZURE_OPENAI_API_VERSION="2024-08-01-preview"
$env:RAG_OPENAI_MODELS="gpt-5.4"
```

**Local `.env` (recommended):** copy `.env.example` to `.env` in the repo root. `python backend/run.py` loads it automatically (`.env` is gitignored).

Install the API extra: `pip install -r backend/requirements.txt`. **`GET /api/models`** returns **Ollama** and **OpenAI / GPT** groups in the UI.

## 3. Use the Web UI

1. Start the server as in step 2.  
2. Open **http://localhost:8000/ui/** in a browser.

The UI lets you:

- Ask a question, set **k** (chunks), optional **min similarity**, and **model** via a dropdown (**Ollama** and **OpenAI / GPT** from **`GET /api/models`**). **Refresh** reloads the list or pulls the selected Ollama model.  
- Toggle **Stream answer** (`/api/ask/stream` SSE) vs **`/api/ask`**.  
- See **retrieved sources** (scores, paths, chunk text) next to the answer.  
- Open **Rebuild knowledge index** to run **`POST /api/reindex`**: enter a **local folder or `.zip` path** on the API server (empty = `data/Knowledgebase`). This can take a long time; keep the tab open until it finishes.

Cross-origin UI example: `http://localhost:3000/ui/?api=http://localhost:8000`

## API summary

| Method | Path | Description |
|--------|------|----------------|
| GET | `/` | Redirect to `/ui/` |
| GET | `/api/health` | Health and KB/embedder/Ollama snapshot |
| GET | `/api/models` | Ollama + OpenAI/Azure models for the UI |
| POST | `/api/ask` | JSON question → full answer + sources |
| POST | `/api/ask/stream` | SSE: `sources`, `token`, `done`, `error` |
| POST | `/api/reindex` | Rebuild **`kb.duckdb`** via subprocess indexer |
| POST | `/api/models/ensure` | Pull/install selected Ollama model |

**POST /api/ask** body (JSON):

```json
{
  "question": "How do I configure X?",
  "k": 5,
  "model": "llama3.1:8b",
  "similarity_min": null
}
```

**POST /api/reindex** body (JSON):

```json
{
  "source": "local",
  "path": "data/Notices.zip"
}
```

`path` is optional; when omitted, indexing uses **`data/Knowledgebase`**. `path` may be a **folder** or **`.zip`** (absolute or relative to project root). Only one index build runs at a time (`409` if busy). Disable with **`RAG_UI_REINDEX=0`** on untrusted networks.

HTTP zip/manifest indexing remains available via **CLI env vars** (`RAG_KB_SOURCE=http_zip` / `http_manifest`); the UI reindex form is **local path only**.

## Scripts (benchmark & eval)

From repo root:

```bash
python scripts/benchmark_ask.py --url http://localhost:8000
python scripts/benchmark_embeddings.py
python scripts/eval_rag.py --eval eval/example_gold.json -k 5
```

See each script’s **`--help`**. For ad-hoc questions, use the **web UI** or **`POST /api/ask`** (`/docs`).

## Troubleshooting

- **`503` / “Knowledge base not found”** — run the indexer; confirm **`kb.duckdb`** exists at the project root.  
- **Ollama errors** — ensure the daemon is running; for embeddings, `ollama pull nomic-embed-text` (or set **`RAG_OLLAMA_EMBED_MODEL`**). If you see `501` / embeddings not supported, restart Ollama with embeddings enabled.  
- **`.docx` index fails** — `pip install docx2txt` (or reinstall `requirements.txt`).  
- **Zip index: “No files found”** — ensure the zip contains supported extensions; extraction uses **`kb_staging/`** (not a hidden path).  
- **Reindex path errors** — do not wrap paths in quotes; use a **folder** or **`.zip`**, not a file of another type.  
- **Embedder / Hugging Face** — keep **`RAG_EMBED_PROVIDER=ollama`** unless you intentionally enable **`st`** with **`RAG_ALLOW_ST_EMBEDDINGS=1`**.

## Roadmap

Higher-level phases and enterprise direction are tracked in **`PLAN.md`**.
