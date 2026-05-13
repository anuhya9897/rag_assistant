"""
RAG service: retrieval + prompt building + Ollama generation.
Uses project-root kb.duckdb so API can be run from any cwd.

Embedding (env):
  RAG_EMBED_PROVIDER     – "ollama" (default) or "st". "st" is only used if RAG_ALLOW_ST_EMBEDDINGS=1
    (avoids accidental global env st= → Hugging Face / SSL failures).
  RAG_ALLOW_ST_EMBEDDINGS – set to 1 to allow sentence-transformers when RAG_EMBED_PROVIDER=st.
  RAG_OLLAMA_EMBED_MODEL – Ollama model for /api/embed. If unset/empty, auto-pick an installed
    model (prefers nomic-embed-text, then other embed models, then DEFAULT_LLM if present).
  RAG_EMBED_DIM          – optional override for Ollama vector length.
  RAG_EMBED_MODEL_PATH   – for provider "st": local folder (no HF download).
  RAG_EMBED_LOCAL_FILES_ONLY – for "st": use only HF cache (needs full cache).
"""
import os
import logging
import time
import duckdb
import ollama
from ollama import ResponseError
import math
from typing import Optional

# Project root (parent of backend/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = _PROJECT_ROOT
DB_PATH = os.path.join(_PROJECT_ROOT, "kb.duckdb")
TABLE = "kb_chunks_local"
# Hub id (must match indexing; see build_index_duckdb_local_incremental.py). Override with RAG_EMBED_MODEL_PATH.
EMBED_MODEL = "all-MiniLM-L6-v2"
# Vector length for all-MiniLM-L6-v2 (must match stored embeddings and SQL FLOAT[n] casts).
EMBED_DIM = 384
DEFAULT_LLM = "llama3.1:8b"

# So `python -m uvicorn ...` also defaults to Ollama unless the user explicitly set RAG_EMBED_PROVIDER.
os.environ.setdefault("RAG_EMBED_PROVIDER", "ollama")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

log = logging.getLogger(__name__)

_embedder = None
_ollama_embed_dim_cached: Optional[int] = None
_unknown_embed_provider_logged = False
_st_gated_logged = False


class EmbedderLoadError(RuntimeError):
    """Raised when the sentence-transformers model cannot be loaded (e.g. SSL or missing cache)."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def embed_provider() -> str:
    """
    Embedding provider.
      - "ollama" (default): Ollama /api/embed — no Hugging Face.
      - "st": sentence-transformers only if RAG_ALLOW_ST_EMBEDDINGS=1 (MiniLM / HF or local cache).
    """
    global _unknown_embed_provider_logged, _st_gated_logged
    v = (os.environ.get("RAG_EMBED_PROVIDER", "ollama") or "ollama").strip().lower()
    if v not in ("st", "ollama"):
        if not _unknown_embed_provider_logged:
            log.warning("Unknown RAG_EMBED_PROVIDER=%r; using ollama", v)
            _unknown_embed_provider_logged = True
        return "ollama"
    if v == "st":
        if not _env_truthy("RAG_ALLOW_ST_EMBEDDINGS"):
            if not _st_gated_logged:
                log.warning(
                    "RAG_EMBED_PROVIDER=st ignored unless RAG_ALLOW_ST_EMBEDDINGS=1 (prevents accidental "
                    "Hugging Face use). Using ollama embeddings. To force MiniLM/HF: set both vars, then restart."
                )
                _st_gated_logged = True
            return "ollama"
        return "st"
    return "ollama"


def _ollama_list_model_names() -> list[str]:
    try:
        r = ollama.list()
    except Exception:
        return []
    if isinstance(r, dict):
        models = r.get("models") or []
    else:
        models = getattr(r, "models", None) or []
    out: list[str] = []
    for m in models:
        if isinstance(m, dict):
            n = m.get("model") or m.get("name")
        else:
            n = getattr(m, "model", None)
        if n:
            out.append(str(n))
    return out


def _ollama_embed_model() -> str:
    """
    Which Ollama model to call for /api/embed.

    If RAG_OLLAMA_EMBED_MODEL is set to a non-empty string, that value is used exactly.

    If unset or empty, pick the best installed candidate so devs who only have a chat model
    (e.g. llama3.1:8b) are not stuck on 404 for nomic-embed-text. Rebuild kb.duckdb after any
    embed model / dimension change.
    """
    raw = os.environ.get("RAG_OLLAMA_EMBED_MODEL")
    if raw is not None and raw.strip():
        return raw.strip()

    names = _ollama_list_model_names()
    if not names:
        return "nomic-embed-text"

    name_by_base: dict[str, str] = {}
    for n in names:
        base = n.split(":", 1)[0]
        name_by_base.setdefault(base, n)

    preference = (
        "nomic-embed-text",
        "mxbai-embed-large",
        "snowflake-arctic-embed",
        "bge-m3",
        "embeddinggemma",
    )
    for pref in preference:
        base = pref.split(":", 1)[0]
        if base in name_by_base:
            picked = name_by_base[base]
            log.info("Ollama embed model (auto): %s", picked)
            return picked

    llm_base = DEFAULT_LLM.split(":", 1)[0]
    if llm_base in name_by_base:
        picked = name_by_base[llm_base]
        log.warning(
            "No dedicated embedding model in Ollama; using %r for /api/embed. "
            "Recommended: ollama pull nomic-embed-text. Rebuild kb.duckdb if you change embed model.",
            picked,
        )
        return picked

    picked = names[0]
    log.warning(
        "No dedicated embedding model in Ollama; using first installed model %r for /api/embed. "
        "Set RAG_OLLAMA_EMBED_MODEL to override.",
        picked,
    )
    return picked


def ollama_embed_model_is_pulled(model: str | None = None) -> bool:
    """True if an Ollama model matching the embed model name (base tag) is installed."""
    wanted = (model or _ollama_embed_model()).strip()
    if not wanted:
        return False
    base = wanted.split(":", 1)[0]
    for n in _ollama_list_model_names():
        if n == wanted:
            return True
        if n.split(":", 1)[0] == base:
            return True
    return False


def _ollama_embed_missing_message(exc: Exception, model: str) -> str:
    return (
        f"Ollama embedding model {model!r} is not installed (or name does not match `ollama list`). "
        f"In a terminal run: ollama pull {model.split(':', 1)[0]}\n"
        "If you cannot pull new models but already have a model like llama3.1:8b, unset "
        "RAG_OLLAMA_EMBED_MODEL (or set it empty) so the API auto-picks an installed model, "
        "then rebuild kb.duckdb with the same settings.\n"
        f"Then restart this API. Original error: {type(exc).__name__}: {exc}"
    )


def _l2_normalize(vec: list[float]) -> list[float]:
    denom = math.sqrt(sum((x * x) for x in vec)) or 0.0
    if denom == 0.0:
        return vec
    return [x / denom for x in vec]


def db_embedding_dim() -> Optional[int]:
    if not os.path.isfile(DB_PATH):
        return None
    try:
        con = duckdb.connect(DB_PATH)
        try:
            row = con.execute(
                f"SELECT len(embedding) FROM {TABLE} WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
            if not row:
                return None
            dim = int(row[0])
            return dim if dim > 0 else None
        finally:
            con.close()
    except Exception:
        return None


def _probe_ollama_embedding_dim() -> int:
    """One short embed call to learn vector size for the configured Ollama embed model."""
    model = _ollama_embed_model()
    try:
        resp = ollama.embed(model=model, input=" ")
    except ResponseError as e:
        if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
            raise EmbedderLoadError(_ollama_embed_missing_message(e, model)) from e
        raise
    vec = (resp.get("embeddings") or [None])[0]
    if not vec:
        raise EmbedderLoadError(
            f"Ollama returned an empty embedding for model {model!r}. Try: ollama pull {model.split(':', 1)[0]}"
        )
    return len(_l2_normalize([float(x) for x in vec]))


def embed_dim() -> int:
    """
    Reported embedding dimension for /api/health (Ollama: probe once; ST: fixed 384).
    Retrieval always uses the actual query vector length and checks it against DuckDB rows.
    """
    global _ollama_embed_dim_cached
    if embed_provider() != "ollama":
        return EMBED_DIM
    env_dim = os.environ.get("RAG_EMBED_DIM", "").strip()
    if env_dim:
        try:
            return int(env_dim)
        except ValueError:
            pass
    if _ollama_embed_dim_cached is not None:
        return _ollama_embed_dim_cached
    try:
        _ollama_embed_dim_cached = _probe_ollama_embedding_dim()
        return _ollama_embed_dim_cached
    except Exception:
        # Before first successful probe (e.g. Ollama down), prefer DB hint or common nomic size.
        db_dim = db_embedding_dim()
        return db_dim if db_dim is not None else 768


def embedder_model_id() -> str:
    """Resolved model path or Hub id for health / logging."""
    if embed_provider() == "ollama":
        return _ollama_embed_model()
    path = os.environ.get("RAG_EMBED_MODEL_PATH", "").strip()
    return path if path else EMBED_MODEL


def is_embedder_loaded() -> bool:
    if embed_provider() == "ollama":
        return True
    return _embedder is not None


def _get_embedder():
    global _embedder
    if embed_provider() == "ollama":
        # No local model object to load; Ollama is checked at call time.
        return None
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        model_id = embedder_model_id()
        st_kwargs: dict = {}
        if _env_truthy("RAG_EMBED_LOCAL_FILES_ONLY"):
            st_kwargs["local_files_only"] = True
        try:
            _embedder = SentenceTransformer(model_id, **st_kwargs)
        except Exception as e:
            hint = (
                "sentence-transformers (RAG_EMBED_PROVIDER=st with RAG_ALLOW_ST_EMBEDDINGS=1) loads via "
                "Hugging Face unless RAG_EMBED_MODEL_PATH is local or RAG_EMBED_LOCAL_FILES_ONLY=1 with a full cache. "
                "Corporate SSL: REQUESTS_CA_BUNDLE / SSL_CERT_FILE.\n"
                "To use Ollama only: unset RAG_ALLOW_ST_EMBEDDINGS, set RAG_EMBED_PROVIDER=ollama (or remove st), restart. "
                "Rebuild kb.duckdb if embedding vector length changes. "
            )
            log.exception("Failed to load embedding model %r", model_id)
            raise EmbedderLoadError(f"{hint}Underlying error: {type(e).__name__}: {e}") from e
    return _embedder


def warmup_embedder():
    """Load the embedding model once (e.g. at API startup)."""
    if embed_provider() == "ollama":
        # Ensure Ollama is reachable and embedding model exists.
        try:
            ollama.list()
        except Exception as e:
            raise EmbedderLoadError(f"Ollama not reachable: {type(e).__name__}: {e}") from e
        return
    _get_embedder()


def get_health_snapshot():
    """Lightweight checks for /api/health: KB file, row count, Ollama."""
    out = {
        "kb_path": DB_PATH,
        "kb_present": os.path.isfile(DB_PATH),
        "chunk_count": None,
        "embed_dim": embed_dim(),
        "ollama_ok": False,
        "ollama_error": None,
        "embedder": {
            "loaded": is_embedder_loaded(),
            "model": embedder_model_id(),
            "local_files_only": _env_truthy("RAG_EMBED_LOCAL_FILES_ONLY"),
            "provider": embed_provider(),
            "kb_embedding_dim": db_embedding_dim(),
            "ollama_embed_model_pulled": ollama_embed_model_is_pulled()
            if embed_provider() == "ollama"
            else None,
        },
    }
    if out["kb_present"]:
        try:
            con = duckdb.connect(DB_PATH)
            try:
                row = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()
                out["chunk_count"] = int(row[0]) if row else 0
            finally:
                con.close()
        except Exception as e:
            out["kb_error"] = str(e)
    try:
        ollama.list()
        out["ollama_ok"] = True
    except Exception as e:
        out["ollama_error"] = str(e)
    return out


def retrieve(question: str, k: int, similarity_min: float | None = None):
    """
    Vector search using DuckDB array_cosine_similarity on fixed-size FLOAT[dim].
    Optional similarity_min filters weak matches (cosine in [-1, 1] for normalized vectors).
    """
    kb_dim = db_embedding_dim()
    if embed_provider() == "ollama":
        om = _ollama_embed_model()
        try:
            resp = ollama.embed(model=om, input=question)
            # EmbedResponse has "embeddings": list[list[float]]
            q_emb = (resp.get("embeddings") or [None])[0]
            if not q_emb:
                raise RuntimeError("Empty embedding returned from Ollama")
            q_emb = _l2_normalize([float(x) for x in q_emb])
        except EmbedderLoadError:
            raise
        except ResponseError as e:
            if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
                raise EmbedderLoadError(_ollama_embed_missing_message(e, om)) from e
            raise EmbedderLoadError(f"Ollama embed failed: {type(e).__name__}: {e}") from e
        except Exception as e:
            raise EmbedderLoadError(f"Ollama embed failed: {type(e).__name__}: {e}") from e
        dim = len(q_emb)
        global _ollama_embed_dim_cached
        if _ollama_embed_dim_cached is None:
            _ollama_embed_dim_cached = dim
        if kb_dim is not None and kb_dim != dim:
            raise EmbedderLoadError(
                f"Knowledge base was indexed with vector length {kb_dim}, but Ollama model "
                f"{_ollama_embed_model()!r} returns length {dim}. Rebuild the index so both use the same embedder.\n"
                "From the rag-poc folder (PowerShell):\n"
                "  python build_index_duckdb_local_incremental.py\n"
                "The index script uses the same Ollama embed settings as the API and will clear/re-embed "
                "when it detects a dimension mismatch.\n"
                "Or keep the 384-dim index and use MiniLM (HF/local): "
                "$env:RAG_EMBED_PROVIDER='st'; $env:RAG_ALLOW_ST_EMBEDDINGS='1'; python backend\\run.py"
            )
    else:
        embedder = _get_embedder()
        q_emb = embedder.encode([question], normalize_embeddings=True)[0].tolist()
        dim = len(q_emb)
        if kb_dim is not None and kb_dim != dim:
            raise EmbedderLoadError(
                f"Knowledge base vector length is {kb_dim} but the loaded embedder returns {dim}. "
                "Rebuild the index or fix RAG_EMBED_MODEL_PATH / RAG_EMBED_PROVIDER."
            )
        if dim != EMBED_DIM:
            raise ValueError(f"Query embedding length {len(q_emb)} != {EMBED_DIM} (provider=st)")

    con = duckdb.connect(DB_PATH)
    # Column is FLOAT[]; DuckDB requires matching fixed array type for array_cosine_similarity.
    base = f"""
    WITH scored AS (
      SELECT
        source,
        chunk_id,
        text,
        array_cosine_similarity(embedding::FLOAT[{dim}], ?::FLOAT[{dim}]) AS similarity
      FROM {TABLE}
      WHERE embedding IS NOT NULL
        AND len(embedding) = {dim}
    )
    SELECT source, chunk_id, text, similarity AS score
    FROM scored
    """
    if similarity_min is not None:
        sql = base + " WHERE similarity >= ? ORDER BY similarity DESC LIMIT ?;"
        params = [q_emb, similarity_min, k]
    else:
        sql = base + " ORDER BY similarity DESC LIMIT ?;"
        params = [q_emb, k]
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def build_prompt(question: str, retrieved):
    blocks = []
    for idx, (source, chunk_id, text, score) in enumerate(retrieved, 1):
        blocks.append(
            f"[Source {idx}] score={score:.4f}\n"
            f"path: {source}\n"
            f"chunk_id: {chunk_id}\n"
            f"content:\n{text}\n"
        )
    system = (
        "The documents below were found by the search engine: exact or closest match to the user's question.\n\n"
        "Your role: understand the user's intent, then generate the best possible answer.\n"
        "Use only the provided documents. Cite with [Source 1], [Source 2], etc. where appropriate.\n"
        "If the documents do not contain enough to answer, say so clearly (e.g. 'I don't know based on the documents.').\n"
        "Be clear, accurate, and direct. No disclaimers or meta-commentary."
    )
    user = f"Question: {question}\n\nDocuments (exact or closest match):\n\n" + "\n\n".join(blocks)
    return system, user


def ask(question: str, k: int = 5, model: str = DEFAULT_LLM, similarity_min: float | None = None):
    """Retrieve chunks, build prompt, call Ollama; returns answer and source list."""
    t0 = time.perf_counter()
    retrieved = retrieve(question, k, similarity_min=similarity_min)
    t_retrieve = time.perf_counter()
    sources = [
        {"source": source, "chunk_id": chunk_id, "text": text, "score": round(score, 4)}
        for (source, chunk_id, text, score) in retrieved
    ]
    system, user = build_prompt(question, retrieved)
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    t_done = time.perf_counter()
    log.info(
        "rag_ask timings_ms retrieve=%.1f llm=%.1f total=%.1f k=%s model=%s",
        (t_retrieve - t0) * 1000,
        (t_done - t_retrieve) * 1000,
        (t_done - t0) * 1000,
        k,
        model,
    )
    answer = resp["message"]["content"]
    return {"answer": answer, "sources": sources}


def ask_stream(question: str, k: int = 5, model: str = DEFAULT_LLM, similarity_min: float | None = None):
    """
    Same as ask() but streams LLM tokens. Yields: ("sources", sources_list) once,
    then ("token", text) for each chunk, then ("done", full_answer).
    """
    t0 = time.perf_counter()
    retrieved = retrieve(question, k, similarity_min=similarity_min)
    t_retrieve = time.perf_counter()
    sources = [
        {"source": source, "chunk_id": chunk_id, "text": text, "score": round(score, 4)}
        for (source, chunk_id, text, score) in retrieved
    ]
    yield ("sources", sources)
    system, user = build_prompt(question, retrieved)
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
    )
    full = []
    first_token = True
    for chunk in stream:
        part = (chunk.get("message") or {}).get("content") or ""
        if part:
            if first_token:
                log.info(
                    "rag_ask_stream timings_ms retrieve=%.1f ttft=%.1f k=%s model=%s",
                    (t_retrieve - t0) * 1000,
                    (time.perf_counter() - t_retrieve) * 1000,
                    k,
                    model,
                )
                first_token = False
            full.append(part)
            yield ("token", part)
    t_done = time.perf_counter()
    log.info(
        "rag_ask_stream timings_ms stream_total=%.1f wall=%.1f k=%s model=%s",
        (t_done - t_retrieve) * 1000,
        (t_done - t0) * 1000,
        k,
        model,
    )
    yield ("done", "".join(full))
