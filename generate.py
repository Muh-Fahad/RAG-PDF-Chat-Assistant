
from dotenv import load_dotenv

from vector_store import query
from embeddings import make_client

load_dotenv()
client = make_client()

CHAT_MODEL = "gemini-2.5-flash"


def build_prompt(question: str, chunks: list[str]) -> str:
    """Combine the retrieved chunks and the question into one instruction."""
    context = "\n\n---\n\n".join(chunks)
    return (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below, which was taken from a PDF. If the answer is not in the context, "
        "say you couldn't find it in the document.\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}"
    )


def answer(question: str, n_results: int = 4) -> dict:
    """Answer a question about the ingested PDF.

    Returns a dict with the answer text and the chunks we used, so the UI can
    show *why* it answered that way (helps you trust/debug the retrieval).
    """
    chunks = query(question, n_results=n_results)

    if not chunks:
        return {
            "answer": "⚠️ No PDF has been ingested yet (the database is empty). "
                      "Upload/ingest a PDF first, then ask.",
            "chunks": [],
        }

    prompt = build_prompt(question, chunks)
    try:
        resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
        text = resp.text
    except Exception as e:
        text = f"⚠️ Error from Gemini: {e}"

    return {"answer": text, "chunks": chunks}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    result = answer(q)
    print("Q:", q)
    print("\nA:", result["answer"])
    print(f"\n(used {len(result['chunks'])} retrieved chunks)")
