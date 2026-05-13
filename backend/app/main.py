"""
RAG POC REST API: ask questions against the knowledge base.
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from pydantic import BaseModel, Field

from .rag_service import (
    DEFAULT_LLM,
    PROJECT_ROOT,
    EmbedderLoadError,
    ask,
    ask_stream,
    get_health_snapshot,
    warmup_embedder,
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
