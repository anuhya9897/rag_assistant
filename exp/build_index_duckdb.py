import os
import time
import duckdb
from tqdm import tqdm
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI

# ---------- Config ----------
KB_DIR = r"data/Knowledgebase"  # adjust if your folder name differs
DB_PATH = "kb.duckdb"
TABLE = "kb_chunks"

INCLUDE_EXTS = [".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64

# ---------- Env ----------
load_dotenv()  # loads .env into environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or set it in the shell.")

client = OpenAI(api_key=api_key)

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

# ---------- Dedup / resume ----------
existing = set(r[0] for r in con.execute(f"SELECT chunk_id FROM {TABLE}").fetchall())
to_process = [n for n in nodes if n.node_id not in existing]

print(f"Existing in DB: {len(existing)} | To embed/store: {len(to_process)}")

def embed_batch(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# ---------- Embed + Insert ----------
for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Embedding"):
    batch_nodes = to_process[i:i + BATCH_SIZE]
    texts = [n.text for n in batch_nodes]

    for attempt in range(3):
        try:
            embs = embed_batch(texts)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

    rows = []
    for n, emb in zip(batch_nodes, embs):
        md = n.metadata or {}
        src = md.get("file_path") or md.get("file_name") or ""
        rows.append((n.node_id, src, n.text, emb))

    con.executemany(f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?)", rows)

count = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
print(f"Done. Wrote {count} rows to {DB_PATH} ({TABLE})")
