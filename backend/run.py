"""
Run the RAG API. From project root (repository root):
  python backend/run.py
  OR: python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import sys

# Ensure project root is on path when running backend/run.py from repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# Load .env from project root (Azure OpenAI, RAG_OPENAI_MODELS, etc.)
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

# Default to Ollama embeddings (rag_service also setdefaults on import).
os.environ.setdefault("RAG_EMBED_PROVIDER", "ollama")

if __name__ == "__main__":
    import uvicorn

    # If RAG_OLLAMA_EMBED_MODEL points at a model that is not installed, drop it for auto-select.
    try:
        from backend.app import rag_service as _rag

        _pinned = os.environ.get("RAG_OLLAMA_EMBED_MODEL", "").strip()
        if _pinned and not _rag.ollama_embed_model_is_pulled(_pinned):
            os.environ.pop("RAG_OLLAMA_EMBED_MODEL", None)
    except Exception:
        pass

    _reload = os.environ.get("RAG_UVICORN_RELOAD", "").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=_reload,
    )
