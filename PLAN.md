# Simba — delivery plan

Actionable phases for the RAG knowledge-base assistant (`rag_assistant/`). Check items off as you go.

**Product name:** **Simba** (RedMane-themed web UI; assistant name in `rag_service.ASSISTANT_NAME`).

**Strategic direction:** evolve from a simple local RAG prototype toward **enterprise AI experimentation**: model benchmarking, optional **internal/private AI infrastructure**, **scalable knowledge retrieval** (including live enterprise sources), and **cross-team collaboration** (e.g. introductions to newer AI initiatives, possible overlap with groups such as “Simpo”).

---

## Phase 1 — Speed and observability

- [x] Request timing logs on the API (middleware).
- [x] Phase timing inside RAG (retrieve vs LLM) in server logs.
- [x] Embedder warmup on API startup (optional via `RAG_WARMUP_EMBEDDER=1`).
- [x] Health endpoint: KB file present, chunk count, Ollama/Azure status, embedder metadata.
- [x] Benchmark script: `python scripts/benchmark_ask.py` (optional `--url`).
- [x] Embedding benchmark script: `python scripts/benchmark_embeddings.py` (local ST models).
- [ ] Record baseline numbers (embed / retrieve / first token / total) for your machine.
- [ ] If corpus grows: profile DuckDB search; consider ANN / vector index options in DuckDB docs.
- [x] Eval script + example gold: `python scripts/eval_rag.py --eval eval/example_gold.json -k 5`; `--with-llm` for answer metrics when `gold_answer` is set.
- [ ] Grow eval set (10–20 questions + expected sources); compare 2–3 chunk sizes/overlaps.
- [ ] When multiple LLM backends exist: run the **same prompts** against each model for comparable benchmarks.

## Phase 2 — UI (current sprint)

- [x] Web UI at `/ui` (chat, streaming, sources).
- [x] **RedMane branding:** header logo, Simba title, PPT-inspired theme (red/teal/white).
- [x] **Model selection:** dropdown to switch Ollama vs OpenAI/Azure GPT on the same KB (A/B testing).
- [x] **Rebuild index from UI:** local **folder or `.zip` path** (`POST /api/reindex`); empty path → `data/Knowledgebase`.
- [x] Assistant identity: **Simba** in system prompt (name questions answered correctly).
- [ ] Polish: loading states, clearer reindex progress, configurable API base URL for deployments.
- [ ] Optional: trim full source text in API for large payloads (flag on request).
- [ ] Surface in UI which embed model / KB path was last indexed (from health or settings).

## Phase 3 — Model experimentation & infrastructure

**Goal:** compare tradeoffs (**model complexity**, **infrastructure cost**, **latency**, **answer quality**) and find a practical **“sweet spot”** model for enterprise-style use cases.

- [ ] **Larger open-source models:** try Llama (and similar) **beyond the current ~8B** where hardware allows.
- [ ] **Capacity check:** document limits (VRAM, RAM, quantisation) for local vs server inference.
- [ ] **Multiple backends in scope:**
  - Local **Ollama**-hosted models.
  - **Larger Llama** models (hosted or org endpoints).
  - **Frontier models** via **Azure OpenAI** / OpenAI-compatible APIs.
  - **Other hosted models** (e.g. **Gemini**) where policy allows.
- [x] **OpenAI-compatible** chat path (Azure OpenAI via `.env` + `openai_llm.py`).
- [ ] **Evaluate consistently:** accuracy, speed, cost, subjective quality on identical retrieval context.
- [ ] Summarise findings: recommended default model + when to escalate.

## Phase 4 — Ingestion pipeline & enterprise sources

**Goal:** move **beyond static exports only** toward **dynamic, system-connected** knowledge where approved.

- [x] **Local `.zip` indexing:** `RAG_KB_DIR` or UI path → extract to `kb_staging/` → index (`.docx` via `docx2txt`).
- [x] **HTTP zip / manifest** indexing via `kb_source.py` (CLI / env; not in UI reindex form).
- [x] Incremental index updates (`build_index_duckdb_local_incremental.py` — stable chunk ids, embed only new/changed).
- [ ] **`.doc` / `.xlsx`** and other enterprise formats (optional readers or pre-convert step).
- [ ] Scheduled job: scrape or export → detect new/changed → chunk → embed → update `kb.duckdb`.
- [ ] Atomic swap or versioned DB during full rebuilds at scale.
- [ ] **Enterprise integration (exploratory):** pull documents from internal platforms (e.g. **Curam / mCase**) under agreed security patterns.
- [ ] Continue **expanding content**; keep **test prompt sets** stable to compare across models and ingestion changes.

## Phase 5 — Retrieval quality (when speed is acceptable)

- [ ] Re-rank top-N retrieved chunks with a cross-encoder (optional).
- [ ] Hybrid search (BM25 / FTS + vectors) if keyword recall matters.
- [ ] Low similarity threshold: refuse or clarify instead of hallucinating.

## Phase 6 — Teams / Power Platform & collaboration

- [ ] Move repo to shared org (mCase delivery) under agreed name.
- [ ] Bot or connector calling the same REST API (no duplicate RAG logic).
- [ ] Document OpenAPI / auth for reverse proxy or API keys.
- [ ] **Follow-up from leadership:** introductions to teams on **newer AI initiatives**; explore collaboration (e.g. **Simpo**-related work, **client-facing AI**, more datasets/content).

---

## Long-term — Secure internal AI knowledge chat

Aspirational track: **index internal enterprise documents/systems**, **chatbot-style Q&A** over them, **open-source models on internal infrastructure** where feasible — aiming for **lower recurring API cost** than always-on public LLM APIs and **stronger security/privacy** for sensitive data. Use **Azure AI infrastructure** when external or hybrid hosting is required.

---

**Run locally:** from the repository root, activate `.venv`, then `python backend/run.py` — API at `http://localhost:8000`, UI at `http://localhost:8000/ui/`.

**Index locally:** `python build_index_duckdb_local_incremental.py` or UI **Rebuild knowledge index** with a folder/zip path (e.g. `data/Notices.zip`).
