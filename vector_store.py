

import chromadb

from embeddings import GeminiEmbeddingFunction, embed_texts

DB_PATH = "chroma_db"
COLLECTION_NAME = "pdf_chunks"


def get_collection():
    """Open (or create) the collection that holds our PDF chunks.

    - PersistentClient(path=...) makes Chroma save to disk instead of memory.
    - embedding_function tells Chroma how to turn text into vectors when we ADD
      documents (it will call our GeminiEmbeddingFunction automatically).
    - metadata space="cosine" makes it measure closeness by *direction* (meaning),
      which is the right choice for text embeddings.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[str], source: str = "pdf") -> int:
    """Embed and store a list of text chunks. Returns how many were stored.

    We give each chunk a unique id like "report.pdf-0", "report.pdf-1", ...
    Because Chroma embeds on add, this is where the Gemini embedding calls happen.
    """
    if not chunks:
        return 0

    collection = get_collection()
    ids = [f"{source}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def reset_collection():
    """Delete everything stored (start fresh before ingesting a new PDF)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def query(question: str, n_results: int = 4) -> list[str]:
    """Return the `n_results` chunks most relevant to `question`.

    Steps:
      1. Embed the QUESTION with task_type RETRIEVAL_QUERY (the matching side of
         the document/query pair — improves accuracy).
      2. Ask Chroma for the nearest stored chunk vectors.
      3. Return their original text.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vector = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(n_results, collection.count()),
    )
    return results["documents"][0]


def count() -> int:
    """How many chunks are currently stored."""
    return get_collection().count()
