import os
import logging
import argparse
import duckdb
import ollama
from sentence_transformers import SentenceTransformer

DB_PATH = "kb.duckdb"
TABLE = "kb_chunks_local"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM = "llama3.1:8b"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

embedder = SentenceTransformer(EMBED_MODEL)

def retrieve(question: str, k: int):
    """Search engine: find exact or closest documents by similarity."""
    q_emb = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    con = duckdb.connect(DB_PATH)

    sql = f"""
    SELECT
      source,
      chunk_id,
      text,
      (
        SELECT SUM(q[i] * embedding[i])
        FROM range(1, len(embedding)+1) r(i)
      ) AS score
    FROM {TABLE}
    CROSS JOIN (SELECT ?::FLOAT[] AS q)
    ORDER BY score DESC
    LIMIT ?;
    """
    return con.execute(sql, [q_emb, k]).fetchall()

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

def main():
    p = argparse.ArgumentParser(description="Ask questions against your local KB (DuckDB + local embeddings + Ollama).")
    p.add_argument("question", help="The question to ask")
    p.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    p.add_argument("--model", default=DEFAULT_LLM, help="Ollama model name (e.g., llama3.1:8b)")
    p.add_argument("--show-sources", action="store_true", help="Print retrieved sources before the answer")
    args = p.parse_args()

    retrieved = retrieve(args.question, args.k)

    if args.show_sources:
        print("\nTop sources:")
        for i, (source, chunk_id, _text, score) in enumerate(retrieved, 1):
            print(f"  {i}. score={score:.4f} | {source} | {chunk_id}")

    system, user = build_prompt(args.question, retrieved)

    resp = ollama.chat(
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    print("\n" + resp["message"]["content"])

if __name__ == "__main__":
    main()
