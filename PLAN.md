# AB KB Assistant — delivery plan

Actionable phases for the RAG POC (`rag-poc/`). Check items off as you go.

## Phase 1 — Speed and observability

- [x] Request timing logs on the API (middleware).
- [x] Phase timing inside RAG (retrieve vs LLM) in server logs.
- [x] Embedder warmup on API startup.
- [x] Health endpoint: KB file present, chunk count, Ollama reachable.
- [x] One-shot benchmark script: `python scripts/benchmark_ask.py` (optional `--url`).
- [ ] Record baseline numbers (embed / retrieve / first token / total) for your machine.
- [ ] If corpus grows: profile DuckDB search; consider ANN / vector index options in DuckDB docs.
- [x] Eval script + example gold: `python scripts/eval_rag.py --eval eval/example_gold.json -k 5` (retrieval P/R@k); add `--with-llm` for answer token P/R/F1 when `gold_answer` is set.
- [ ] Grow eval set (10–20 questions + expected sources); compare 2–3 chunk sizes/overlaps.

## Phase 2 — UI (current sprint)

- [x] Minimal web UI at `/ui` (chat, streaming, sources).
- [ ] Polish: loading states, error toasts, configurable API base URL for deployments.
- [ ] Optional: trim full source text in API for large payloads (flag on request).

## Phase 3 — Ingestion pipeline

- [ ] Finish scraper output contract (manifest or folder layout).
- [ ] Scheduled job: scrape → detect new/changed → chunk → embed → update `kb.duckdb`.
- [ ] Incremental indexing where possible; atomic swap or versioned DB during rebuild.

## Phase 4 — Teams / Power Platform

- [ ] Move repo to shared org (mCase delivery) under agreed name.
- [ ] Bot or connector calling the same REST API (no duplicate RAG logic).
- [ ] Document OpenAPI / auth for reverse proxy or API keys.

## Phase 5 — Retrieval quality (when speed is acceptable)

- [ ] Re-rank top-N retrieved chunks with a cross-encoder (optional).
- [ ] Hybrid search (BM25 / FTS + vectors) if keyword recall matters.
- [ ] Low similarity threshold: refuse or clarify instead of hallucinating.

---

**Run locally:** from `rag-poc/`, `python backend/run.py` — API at `http://localhost:8000`, UI at `http://localhost:8000/ui/`.
