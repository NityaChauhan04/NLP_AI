# retriever.py
from chroma_indexer import client, embedder

def retrieve_from_chroma(query: str, top_k: int = 3):
    """
    Given a natural language query, return top_k relevant chunks from ChromaDB.
    Each hit is a dict: {doc, meta, score}
    """
    collection_name = "uploaded_docs"
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        # no collection indexed yet
        return []

    collection = client.get_collection(collection_name)
    q_emb = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    hits = []
    # results fields are lists-of-lists
    docs_list = results.get('documents', [[]])[0]
    metas_list = results.get('metadatas', [[]])[0]
    dists_list = results.get('distances', [[]])[0]

    for i in range(len(docs_list)):
        hits.append({
            "doc": docs_list[i],
            "meta": metas_list[i],
            "score": 1 - dists_list[i]  # convert distance -> similarity-like
        })
    return hits
