# chroma_indexer.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# local chroma DB persisted directory
persist_dir = "./chromadb_data"
client = chromadb.Client(Settings(persist_directory=persist_dir))

# embedding model (small & fast)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_code_chunks(text: str, file_name: str = "uploaded_file", max_lines: int = 30):
    """
    Split the uploaded text (code or documentation) into chunks by lines.
    Returns list of chunk dicts with id, filepath, start/end, doc.
    """
    lines = text.splitlines()
    chunks = []
    idx = 0
    for i in range(0, len(lines), max_lines):
        block = lines[i:i+max_lines]
        chunk_text = "\n".join(block).strip()
        if not chunk_text:
            continue
        chunks.append({
            "id": str(idx),
            "filepath": file_name,
            "name": file_name,
            "start": i + 1,
            "end": i + len(block),
            "doc": chunk_text
        })
        idx += 1
    # if file has no newline (single-line), still create a chunk
    if not chunks and text.strip():
        chunks.append({
            "id": "0",
            "filepath": file_name,
            "name": file_name,
            "start": 1,
            "end": len(text.splitlines()) or 1,
            "doc": text.strip()
        })
    return chunks

def index_uploaded_text(text: str, file_name: str = "uploaded_file"):
    """
    Index uploaded text into ChromaDB.
    Recreates the 'uploaded_docs' collection each time for simplicity.
    """
    chunks = extract_code_chunks(text, file_name=file_name)
    if not chunks:
        raise ValueError("No text to index.")
    docs = [c['doc'] for c in chunks]
    ids = [c['id'] for c in chunks]
    metadatas = [{"filepath": c['filepath'], "start": c['start'], "end": c['end']} for c in chunks]

    embeddings = embedder.encode(docs, convert_to_numpy=True)

    collection_name = "uploaded_docs"
    # remove existing collection for fresh indexing
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(name=collection_name)
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    return collection
