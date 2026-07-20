

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pdf_loader import load_pdf_chunks
from vector_store import add_chunks, reset_collection, count


def ingest(path: str, keep_existing: bool = False) -> int:
    """Read, chunk, embed and store one PDF. Returns the number of chunks stored."""
    print(f"📖 Reading and chunking: {path}")
    chunks = load_pdf_chunks(path)

    if not chunks:
        print("⚠️  No readable text found. Is this a scanned/image-only PDF?")
        return 0

    print(f"   -> {len(chunks)} chunks")

    if not keep_existing:
        reset_collection()

    source = path.replace("\\", "/").split("/")[-1]
    print("🧠 Embedding with Gemini and storing in ChromaDB (this calls the API)...")
    n = add_chunks(chunks, source=source)

    print(f"✅ Done. Stored {n} chunks. Database now holds {count()} chunks total.")
    return n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
