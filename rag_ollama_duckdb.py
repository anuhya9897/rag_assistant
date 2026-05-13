import argparse
import duckdb
from sentence_transformers import SentenceTransformer
import ollama

DB_PATH = "kb.duckdb"
TABLE = "kb_chunks_local"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1:8b"

def retrieve(query: str, k: int):
    """Search engine: find exact or closest documents by similarity."""
    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

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

def build_prompt(query: str, retrieved):
    context_blocks = []
    for idx, (source, chunk_id, text, score) in enumerate(retrieved, 1):
        context_blocks.append(
            f"[Source {idx}] score={score:.4f}\n"
            f"path: {source}\n"
            f"chunk_id: {chunk_id}\n"
            f"content:\n{text}\n"
        )

    context = "\n\n".join(context_blocks)

    system = (
        "The documents below were found by the search engine: exact or closest match to the user's question.\n\n"
        "Your role: understand the user's intent, then generate the best possible answer.\n"
        "Use only the provided documents. Cite with [Source 1], [Source 2], etc. where appropriate.\n"
        "If the documents do not contain enough to answer, say so clearly (e.g. 'I don't know based on the documents.').\n"
        "Be clear, accurate, and direct. No disclaimers or meta-commentary."
    )

    user = f"Question: {query}\n\nDocuments (exact or closest match):\n{context}"
    return system, user

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    args = parser.parse_args()

    retrieved = retrieve(args.query, args.k)

    print("\nTop sources:")
    for i, (source, chunk_id, _text, score) in enumerate(retrieved, 1):
        print(f"  {i}. score={score:.4f} | {source} | {chunk_id}")

    system, user = build_prompt(args.query, retrieved)

    resp = ollama.chat(
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    print("\nAnswer:\n")
    print(resp["message"]["content"])

if __name__ == "__main__":
    main()
