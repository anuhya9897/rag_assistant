import argparse
import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "kb.duckdb"
TABLE = "kb_chunks_local"
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User question / search query")
    parser.add_argument("--k", type=int, default=5, help="Top-K results")
    args = parser.parse_args()

    # Embed query (normalized => cosine similarity == dot product)
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([args.query], normalize_embeddings=True)[0].astype(np.float32)

    con = duckdb.connect(DB_PATH)
    rows = con.execute(f"SELECT chunk_id, source, text, embedding FROM {TABLE}").fetchall()

    scored = []
    for chunk_id, source, text, emb in rows:
        v = np.array(emb, dtype=np.float32)
        # stored embeddings were normalized too, so dot product = cosine sim
        score = float(np.dot(q_emb, v))
        scored.append((score, chunk_id, source, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: args.k]

    for i, (score, chunk_id, source, text) in enumerate(top, 1):
        print(f"\n[{i}] score={score:.4f}")
        print(f"source: {source}")
        print(f"chunk_id: {chunk_id}")
        print("text:")
        print(text[:1200])  # cap output
        if len(text) > 1200:
            print("...")

if __name__ == "__main__":
    main()
