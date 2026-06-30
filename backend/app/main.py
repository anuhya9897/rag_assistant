"""
RAG POC REST API: ask questions against the knowledge base.
"""
import json
import logging
import os

try:
    from dotenv import load_dotenv

    _env_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(_env_root, ".env"))
except ImportError:
    pass
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field, model_validator

from ollama import ResponseError

from kb_source import normalize_kb_path

from .openai_llm import is_openai_chat_model
from .rag_service import (
    DEFAULT_LLM,
    PROJECT_ROOT,
    EmbedderLoadError,
    ModelInstallError,
    OpenAIChatError,
    ask,
    ask_stream,
    get_health_snapshot,
    ensure_ollama_chat_model,
    list_all_llm_models,
    warmup_embedder,
    _is_ollama_model_not_found,
    _ollama_model_is_installed_exact,
)

log = logging.getLogger("rag.api")

_WEB_DIR = os.path.join(PROJECT_ROOT, "web")


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=16000,
        description="Question to ask the knowledge base",
    )
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    model: str = Field(default=DEFAULT_LLM, description="Ollama model name")
    similarity_min: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="If set, only return chunks with cosine similarity >= this value.",
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class ModelEnsureRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256, description="Ollama model tag to install")


class ModelEnsureResponse(BaseModel):
    ok: bool
    model: str
    installed: bool


class ReindexRequest(BaseModel):
    """Rebuild kb.duckdb from a local knowledge-base folder or .zip file."""

    source: Literal["local"] = Field(
        default="local",
        description="Local folder or .zip on the API server",
    )
    path: Optional[str] = Field(
        default=None,
        max_length=8000,
        description="KB folder or .zip path (absolute or relative to project root). Empty = data/Knowledgebase",
    )

class ReindexResponse(BaseModel):
    ok: bool
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""


_reindex_lock = threading.Lock()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Do not load the embedder here by default: Hugging Face / SSL failures would block the UI.
    if _env_truthy("RAG_WARMUP_EMBEDDER"):
        log.info("Warming up embedder (RAG_WARMUP_EMBEDDER=1)…")
        try:
            warmup_embedder()
            log.info("Embedder ready.")
        except EmbedderLoadError as e:
            log.warning("Embedder warmup failed; API will still start: %s", e)
    else:
        log.info("Embedder will load on first question (omit HF at startup).")
    yield


app = FastAPI(
    title="RAG POC API",
    description="Ask questions against the indexed knowledge base (DuckDB + Ollama).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    log.info("%s %s %.1fms", request.method, request.url.path, ms)
    return response


@app.get("/")
def root():
    return RedirectResponse(url="/ui/", status_code=302)


@app.get("/api/health")
def health():
    snap = get_health_snapshot()
    status = "ok"
    if not snap.get("kb_present"):
        status = "degraded"
    if not snap.get("ollama_ok"):
        status = "degraded"
    emb = snap.get("embedder") or {}
    if emb.get("provider") == "ollama" and emb.get("ollama_embed_model_pulled") is False:
        status = "degraded"
    return {"status": status, **snap}


def _reindex_url_host_allowed(url: str) -> bool:
    """If RAG_REINDEX_URL_ALLOWLIST is set (comma-separated hostnames), URL host must match one entry."""
    allow = os.environ.get("RAG_REINDEX_URL_ALLOWLIST", "").strip()
    if not allow:
        return True
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    for entry in allow.split(","):
        suffix = entry.strip().lower()
        if not suffix:
            continue
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


def _validate_reindex_url(url: str) -> str:
    u = url.strip()
    p = urlparse(u)
    if p.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="url must use http or https")
    if not p.hostname:
        raise HTTPException(status_code=400, detail="url must include a hostname")
    if not _reindex_url_host_allowed(u):
        raise HTTPException(
            status_code=400,
            detail="URL host is not allowed by RAG_REINDEX_URL_ALLOWLIST server configuration",
        )
    return u


@app.post("/api/reindex", response_model=ReindexResponse)
def post_reindex(body: ReindexRequest):
    """
    Run build_index_duckdb_local_incremental.py in a subprocess with RAG_KB_* env from this request.
    Disable with RAG_UI_REINDEX=0. Long-running; ensure reverse-proxy timeouts are sufficient.
    """
    if os.environ.get("RAG_UI_REINDEX", "1").strip().lower() in ("0", "false", "no", "off"):
        raise HTTPException(status_code=403, detail="Reindex from UI is disabled (RAG_UI_REINDEX=0).")

    acquired = _reindex_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=409, detail="An index build is already running.")

    try:
        kb_path = (body.path or "").strip().strip('"').strip("'")
        if kb_path:
            resolved = normalize_kb_path(kb_path)
            is_zip = os.path.isfile(resolved) and resolved.lower().endswith(".zip")
            if not os.path.isdir(resolved) and not is_zip:
                raise HTTPException(
                    status_code=400,
                    detail=f"Knowledge base folder or .zip not found: {resolved}",
                )
        script = os.path.join(PROJECT_ROOT, "build_index_duckdb_local_incremental.py")
        if not os.path.isfile(script):
            raise HTTPException(status_code=500, detail="Indexer script not found on server.")

        env = os.environ.copy()
        for k in (
            "RAG_KB_SOURCE",
            "RAG_KB_DIR",
            "RAG_KB_HTTP_URL",
            "RAG_KB_MANIFEST_URL",
            "RAG_KB_HTTP_BEARER",
            "RAG_KB_STAGING_DIR",
            "RAG_KB_HTTP_HEADERS",
            "RAG_KB_HTTP_TIMEOUT",
            "RAG_KB_HTTP_INSECURE",
            "RAG_KB_STAGING_KEEP",
        ):
            env.pop(k, None)

        env["RAG_KB_SOURCE"] = "local"
        if kb_path:
            env["RAG_KB_DIR"] = kb_path

        timeout_raw = os.environ.get("RAG_REINDEX_TIMEOUT_SEC", "").strip()
        timeout_sec = float(timeout_raw) if timeout_raw else 3600.0
        timeout_sec = max(60.0, min(timeout_sec, 86400.0))

        log.info("Starting KB reindex subprocess source=%s", body.source)
        proc = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out_t = (proc.stdout or "")[-6000:]
        err_t = (proc.stderr or "")[-6000:]
        if proc.returncode != 0:
            log.warning("Reindex failed rc=%s stderr=%s", proc.returncode, err_t[:500])
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Index build failed",
                    "returncode": proc.returncode,
                    "stdout_tail": out_t,
                    "stderr_tail": err_t,
                },
            )
        log.info("KB reindex finished OK")
        return ReindexResponse(ok=True, returncode=0, stdout_tail=out_t, stderr_tail=err_t)
    except subprocess.TimeoutExpired as e:
        log.error("Reindex subprocess timed out after %s s", timeout_sec)
        so = e.stdout if e.stdout is not None else getattr(e, "output", None) or ""
        se = e.stderr if e.stderr is not None else ""
        if isinstance(so, bytes):
            so = so.decode(errors="replace")
        if isinstance(se, bytes):
            se = se.decode(errors="replace")
        out_t = str(so)[-4000:]
        err_t = str(se)[-4000:]
        raise HTTPException(
            status_code=504,
            detail={
                "message": f"Index build timed out after {int(timeout_sec)}s",
                "stdout_tail": out_t,
                "stderr_tail": err_t,
            },
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        log.exception("reindex failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        _reindex_lock.release()


@app.get("/api/models")
def list_models():
    """Ollama + OpenAI/Azure GPT models for UI dropdown."""
    snap = get_health_snapshot()
    ollama_ok = bool(snap.get("ollama_ok"))
    openai_ok = bool(snap.get("openai_ok"))
    try:
        bundled = list_all_llm_models()
    except Exception:
        log.exception("list_all_llm_models failed")
        bundled = {"models": [DEFAULT_LLM], "providers": {"ollama": [DEFAULT_LLM], "openai": []}}
    models = bundled.get("models") or [DEFAULT_LLM]
    if not models:
        models = [DEFAULT_LLM]
    return {
        "models": models,
        "providers": bundled.get("providers") or {"ollama": models, "openai": []},
        "default": DEFAULT_LLM,
        "ollama_ok": ollama_ok,
        "openai_ok": openai_ok,
    }


@app.post("/api/models/ensure", response_model=ModelEnsureResponse)
def post_models_ensure(body: ModelEnsureRequest):
    """Pull an Ollama model if missing (used when UI Refresh is clicked with a custom model tag)."""
    tag = body.model.strip()
    if is_openai_chat_model(tag):
        return ModelEnsureResponse(ok=True, model=tag, installed=True)
    already = _ollama_model_is_installed_exact(tag)
    try:
        ensure_ollama_chat_model(tag)
    except ModelInstallError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ResponseError as e:
        if _is_ollama_model_not_found(e):
            raise HTTPException(status_code=503, detail="model installation failed") from e
        raise HTTPException(status_code=500, detail="Internal error") from e
    return ModelEnsureResponse(ok=True, model=tag, installed=already)


@app.post("/api/ask", response_model=AskResponse)
def post_ask(body: AskRequest):
    try:
        result = ask(
            question=body.question,
            k=body.k,
            model=body.model,
            similarity_min=body.similarity_min,
        )
        return AskResponse(**result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not found. Run index build first.",
        )
    except EmbedderLoadError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ModelInstallError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except OpenAIChatError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ResponseError as e:
        if _is_ollama_model_not_found(e):
            raise HTTPException(status_code=503, detail="model installation failed") from e
        log.exception("ask failed (ollama)")
        raise HTTPException(status_code=500, detail="Internal error") from e
    except Exception as e:
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail="Internal error") from e


def _sse_stream(body: AskRequest):
    """Generator yielding SSE lines for streaming ask."""
    try:
        for kind, payload in ask_stream(
            question=body.question,
            k=body.k,
            model=body.model,
            similarity_min=body.similarity_min,
        ):
            if kind == "sources":
                yield f"data: {json.dumps({'event': 'sources', 'sources': payload})}\n\n"
            elif kind == "token":
                yield f"data: {json.dumps({'event': 'token', 'content': payload})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'event': 'done', 'answer': payload})}\n\n"
    except FileNotFoundError:
        yield f"data: {json.dumps({'event': 'error', 'detail': 'Knowledge base not found.'})}\n\n"
    except EmbedderLoadError as e:
        yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"
    except ModelInstallError as e:
        yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"
    except OpenAIChatError as e:
        yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"
    except ResponseError as e:
        if _is_ollama_model_not_found(e):
            yield f"data: {json.dumps({'event': 'error', 'detail': 'model installation failed'})}\n\n"
        else:
            log.exception("ask_stream failed (ollama)")
            yield f"data: {json.dumps({'event': 'error', 'detail': 'Internal error'})}\n\n"
    except Exception as e:
        log.exception("ask_stream failed")
        yield f"data: {json.dumps({'event': 'error', 'detail': 'Internal error'})}\n\n"


@app.post("/api/ask/stream")
def post_ask_stream(body: AskRequest):
    """Stream answer tokens as Server-Sent Events for faster perceived response."""
    return StreamingResponse(
        _sse_stream(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if os.path.isdir(_WEB_DIR):
    app.mount("/ui", StaticFiles(directory=_WEB_DIR, html=True), name="ui")
