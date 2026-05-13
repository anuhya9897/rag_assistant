import duckdb
from tqdm import tqdm

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from sentence_transformers import SentenceTransformer

# ---------- Config ----------
# Use the exact folder you used in load_kb.py (your metadata showed data\Knowledgebase\...)
KB_DIR = r"data\Knowledgebase"
DB_PATH = "kb.duckdb"
TABLE = "kb_chunks_local"

INCLUDE_EXTS = [".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Local embedding model (fast, commonly used for semantic search)
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

# ---------- DuckDB ----------
con = duckdb.connect(DB_PATH)
con.execute(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  chunk_id VARCHAR PRIMARY KEY,
  source   VARCHAR,
  text     VARCHAR,
  embedding FLOAT[]
);
""")

# ---------- Load + Chunk ----------
reader = SimpleDirectoryReader(
    input_dir=KB_DIR,
    recursive=True,
    required_exts=INCLUDE_EXTS,
)
documents = reader.load_data()

splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = splitter.get_nodes_from_documents(documents)

print(f"Loaded {len(documents)} documents -> {len(nodes)} chunks")

# ---------- Resume / skip existing ----------
existing = set(r[0] for r in con.execute(f"SELECT chunk_id FROM {TABLE}").fetchall())
to_process = [n for n in nodes if n.node_id not in existing]

print(f"Existing in DB: {len(existing)} | To embed/store: {len(to_process)}")

# ---------- Embed locally ----------
model = SentenceTransformer(LOCAL_EMBED_MODEL)

for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Embedding (local)"):
    batch_nodes = to_process[i:i + BATCH_SIZE]
    texts = [n.text for n in batch_nodes]

    # returns a numpy array shape: (batch, dim)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    rows = []
    for n, emb in zip(batch_nodes, embs):
        md = n.metadata or {}
        src = md.get("file_path") or md.get("file_name") or ""
        rows.append((n.node_id, src, n.text, emb.tolist()))

    con.executemany(f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?)", rows)

count = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
print(f"Done. Wrote {count} rows to {DB_PATH} ({TABLE})")
