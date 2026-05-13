import argparse
import duckdb
from sentence_transformers import SentenceTransformer
import ollama

DB_PATH = "kb.duckdb"
TABLE = "kb_chunks_local"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1:8b"

embedder = SentenceTransformer(EMBED_MODEL)

def retrieve(question: str, k: int):
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
    context_blocks = []
    for idx, (source, chunk_id, text, score) in enumerate(retrieved, 1):
        context_blocks.append(
            f"[Source {idx}] score={score:.4f}\npath: {source}\nchunk_id: {chunk_id}\ncontent:\n{text}\n"
        )

    system = (
        "You answer questions using ONLY the provided sources.\n"
        "Every paragraph MUST end with at least one citation in the form [Source N].\n"
        "If a step is based on multiple sources, cite all relevant ones like [Source 1][Source 4].\n"
        "If the answer is not in the sources, say exactly: 'I don't know based on the provided sources.'\n"
        "Be concise and procedural.\n"
        "Do not add disclaimers or commentary."
    )

    user = f"Question: {question}\n\nSources:\n\n" + "\n\n".join(context_blocks)
    return system, user

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model", default=OLLAMA_MODEL)
    p.add_argument("--show_sources", action="store_true")
    args = p.parse_args()

    retrieved = retrieve(args.question, args.k)

    if args.show_sources:
        for i, (source, chunk_id, _text, score) in enumerate(retrieved, 1):
            print(f"{i}. score={score:.4f} | {source} | {chunk_id}")

    system, user = build_prompt(args.question, retrieved)
    resp = ollama.chat(
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    print(resp["message"]["content"])

if __name__ == "__main__":
    main()
