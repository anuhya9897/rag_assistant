import os
import sys
import hashlib
import math
import shutil
import duckdb
from tqdm import tqdm

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# Project root (folder containing this script)
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("RAG_EMBED_PROVIDER", "ollama")

import ollama
from ollama import ResponseError

from backend.app.rag_service import _ollama_embed_model, embed_provider
from kb_source import prepare_kb_dir

DB_PATH = "kb.duckdb"
TABLE = "kb_chunks_local"

INCLUDE_EXTS = [".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def l2_normalize(vec: list[float]) -> list[float]:
    denom = math.sqrt(sum((x * x) for x in vec)) or 0.0
    if denom == 0.0:
        return vec
    return [x / denom for x in vec]


def norm_path(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def normalize_text(t: str) -> str:
    return " ".join(t.split())


def stable_chunk_id(source_path: str, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(chunk_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def main():
    prov = embed_provider()
    print(f"RAG_EMBED_PROVIDER effective: {prov}")

    staging_remove: str | None = None
    try:
        kb_dir, staging_remove = prepare_kb_dir()
        print(f"Knowledge base directory: {kb_dir}")

        if prov == "ollama":
            print(f"Ollama embed model: {_ollama_embed_model()!r}")

        con = duckdb.connect(DB_PATH)
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
          chunk_id VARCHAR PRIMARY KEY,
          source   VARCHAR,
          text     VARCHAR,
          embedding FLOAT[]
        );
        """)

        con.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_source ON {TABLE}(source);")

        if not os.path.isdir(kb_dir):
            raise RuntimeError(
                f"Knowledge base directory does not exist: {kb_dir}. Aborting to avoid DB changes."
            )

        reader = SimpleDirectoryReader(
            input_dir=kb_dir,
            recursive=True,
            required_exts=INCLUDE_EXTS,
            exclude_hidden=False,
        )
        documents = reader.load_data()

        if len(documents) == 0:
            raise RuntimeError(
                "No documents loaded from knowledge base directory. Aborting to avoid deleting DB contents. "
                "Check path, RAG_KB_SOURCE / remote URLs, and INCLUDE_EXTS."
            )

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(documents)

        current_rows = []
        current_sources = set()
        for n in nodes:
            md = n.metadata or {}
            src = md.get("file_path") or md.get("file_name") or ""
            if not src:
                src = "UNKNOWN_SOURCE"
            src = norm_path(src) if src != "UNKNOWN_SOURCE" else src
            cid = stable_chunk_id(src, normalize_text(n.text))
            current_rows.append((cid, src, n.text))
            current_sources.add(src)

        print(f"Loaded {len(documents)} documents -> {len(nodes)} chunks")
        print(f"Current sources: {len(current_sources)}")
        print(f"Current chunks (stable ids): {len(current_rows)}")

        if current_sources:
            con.execute(
                f"DELETE FROM {TABLE} WHERE source NOT IN (SELECT * FROM UNNEST(?::VARCHAR[]))",
                [list(current_sources)],
            )
        else:
            con.execute(f"DELETE FROM {TABLE}")

        current_ids = [cid for (cid, _src, _text) in current_rows]
        if current_ids and current_sources:
            con.execute(
                f"""
                DELETE FROM {TABLE}
                WHERE source IN (SELECT * FROM UNNEST(?::VARCHAR[]))
                  AND chunk_id NOT IN (SELECT * FROM UNNEST(?::VARCHAR[]))
                """,
                [list(current_sources), current_ids],
            )

        if current_ids:
            existing_ids = set(
                r[0]
                for r in con.execute(
                    f"SELECT chunk_id FROM {TABLE} WHERE chunk_id IN (SELECT * FROM UNNEST(?::VARCHAR[]))",
                    [current_ids],
                ).fetchall()
            )
        else:
            existing_ids = set()

        to_add = [(cid, src, text) for (cid, src, text) in current_rows if cid not in existing_ids]
        print(f"Existing chunks kept: {len(existing_ids)}")
        print(f"New chunks to embed+insert: {len(to_add)}")

        # Ollama: if DB already has another embedding length, clear table and re-embed everything (no 384+4096 mix).
        if prov == "ollama" and (to_add or existing_ids):
            om = _ollama_embed_model()
            try:
                r = ollama.embed(model=om, input=" ")
                vec = (r.get("embeddings") or [None])[0]
                if not vec:
                    raise RuntimeError("Ollama returned empty embedding for probe")
                need_dim = len(l2_normalize([float(x) for x in vec]))
            except ResponseError as e:
                if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
                    raise RuntimeError(
                        f"Ollama embed model {om!r} not available. Run: ollama pull {om.split(':', 1)[0]}"
                    ) from e
                raise
            dim_rows = con.execute(
                f"SELECT DISTINCT len(embedding) FROM {TABLE} WHERE embedding IS NOT NULL"
            ).fetchall()
            stored_dims = [int(d[0]) for d in dim_rows if d and d[0] is not None]
            if stored_dims and any(d != need_dim for d in stored_dims):
                print(
                    f"Embedding size mismatch: table has lengths {stored_dims} "
                    f"but {om!r} produces {need_dim}. Clearing {TABLE} and re-embedding all chunks."
                )
                con.execute(f"DELETE FROM {TABLE}")
                existing_ids = set()
                to_add = list(current_rows)
                print(f"Full re-embed: {len(to_add)} chunks")

        if to_add:
            if prov == "ollama":
                om = _ollama_embed_model()
                for i in tqdm(range(0, len(to_add), BATCH_SIZE), desc="Embedding (Ollama)"):
                    batch = to_add[i : i + BATCH_SIZE]
                    texts = [t for (_cid, _src, t) in batch]
                    try:
                        resp = ollama.embed(model=om, input=texts)
                    except ResponseError as e:
                        if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
                            raise RuntimeError(
                                f"Ollama embedding model {om!r} is not installed. Run: ollama pull {om.split(':', 1)[0]}"
                            ) from e
                        raise
                    embs = resp.get("embeddings") or []
                    if len(embs) != len(texts):
                        raise RuntimeError("Ollama embed returned unexpected embedding count")
                    embs = [l2_normalize([float(x) for x in emb]) for emb in embs]

                    rows = []
                    for (cid, src, text), emb in zip(batch, embs):
                        rows.append((cid, src, text, emb))

                    con.executemany(
                        f"INSERT INTO {TABLE} (chunk_id, source, text, embedding) VALUES (?, ?, ?, ?)",
                        rows,
                    )
            else:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(LOCAL_EMBED_MODEL)
                for i in tqdm(range(0, len(to_add), BATCH_SIZE), desc="Embedding (sentence-transformers)"):
                    batch = to_add[i : i + BATCH_SIZE]
                    texts = [t for (_cid, _src, t) in batch]
                    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

                    rows = []
                    for (cid, src, text), emb in zip(batch, embs):
                        rows.append((cid, src, text, emb.tolist()))

                    con.executemany(
                        f"INSERT INTO {TABLE} (chunk_id, source, text, embedding) VALUES (?, ?, ?, ?)",
                        rows,
                    )

        count = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
        print(f"Done. {TABLE} rows now: {count} (DB: {DB_PATH})")
    finally:
        if staging_remove and os.path.isdir(staging_remove):
            shutil.rmtree(staging_remove, ignore_errors=True)
            print(f"Removed staging directory: {staging_remove}")


if __name__ == "__main__":
    main()
