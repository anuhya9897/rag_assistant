from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

INCLUDE_EXTS = [
    ".md", ".markdown", ".txt",
    ".html", ".htm",
    ".pdf",
    ".docx",
]

reader = SimpleDirectoryReader(
    input_dir="data/Knowledgebase",
    recursive=True,
    required_exts=INCLUDE_EXTS,
)
documents = reader.load_data()
print(f"Loaded {len(documents)} raw documents")

splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Created {len(nodes)} chunks")

print("\n--- Sample chunk ---")
print(nodes[0].text[:500])
print("\n--- Sample metadata ---")
print(nodes[0].metadata)
