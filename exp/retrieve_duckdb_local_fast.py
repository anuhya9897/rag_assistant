import argparse
import duckdb
from sentence_transformers import SentenceTransformer

DB_PATH = "kb.duckdb"
TABLE = "kb_chunks_local"
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([args.query], normalize_embeddings=True)[0].tolist()

    con = duckdb.connect(DB_PATH)

    # Dot product in SQL:
    # sum over i: q[i] * emb[i]
    sql = f"""
    SELECT
    source,
    chunk_id,
    text,
    (
        SELECT SUM(q[i] * embedding[i])
        FROM range(1, array_length(embedding)+1) r(i)
    ) AS score
    FROM {TABLE}
    CROSS JOIN (SELECT ?::FLOAT[] AS q)
    ORDER BY score DESC
    LIMIT ?;
    """


    rows = con.execute(sql, [q_emb, args.k]).fetchall()

    for idx, (source, chunk_id, text, score) in enumerate(rows, 1):
        print(f"\n[{idx}] score={score:.4f}")
        print(f"source: {source}")
        print(f"chunk_id: {chunk_id}")
        print("text:")
        print(text[:1200] + ("..." if len(text) > 1200 else ""))

if __name__ == "__main__":
    main()
