

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()


def make_client():
    """Create the Gemini client.

    If GOOGLE_GENAI_USE_VERTEXAI=true (see .env), we authenticate to Vertex AI with
    the service-account credentials in GOOGLE_APPLICATION_CREDENTIALS — this has far
    higher rate limits than the free AI Studio tier. Otherwise we fall back to the
    plain API key.
    """
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true":
        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


client = make_client()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768

BATCH_SIZE = 50
PAUSE_BETWEEN_BATCHES = 1.0
MAX_RETRIES = 5


def _embed_one_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed up to BATCH_SIZE texts in a single call, retrying on rate limits."""
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBED_DIM,
                ),
            )
            return [e.values for e in result.embeddings]
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429 and attempt < MAX_RETRIES:
                print(f"   [wait] rate limit hit, waiting {delay:.0f}s (retry {attempt}/{MAX_RETRIES})...")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Turn each string in `texts` into a vector, in safe batches.

    task_type is either:
      "RETRIEVAL_DOCUMENT" -> for chunks we are storing
      "RETRIEVAL_QUERY"    -> for the user's question we are searching with

    Instead of one huge request, we loop over the list BATCH_SIZE at a time. This
    is what lets a big PDF go through without a single oversized call.
    """
    vectors: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        vectors.extend(_embed_one_batch(batch, task_type))

        done = min(start + BATCH_SIZE, total)
        if total > BATCH_SIZE:
            print(f"   embedded {done}/{total} chunks")
            if done < total:
                time.sleep(PAUSE_BETWEEN_BATCHES)

    return vectors


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Adapter so ChromaDB can create embeddings with Gemini.

    Chroma calls this object with the texts it wants embedded. We tell it to treat
    them as documents (RETRIEVAL_DOCUMENT). The user's question is embedded
    separately with RETRIEVAL_QUERY in vector_store.query().
    """

    def __call__(self, input: Documents) -> Embeddings:
        return embed_texts(list(input), task_type="RETRIEVAL_DOCUMENT")


if __name__ == "__main__":
    vecs = embed_texts(["yearly salary", "annual compensation", "the cat sat on a mat"])
    print(f"Got {len(vecs)} vectors, each {len(vecs[0])} numbers long.")

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb)

    print("salary  vs compensation:", round(cosine(vecs[0], vecs[1]), 3), "(should be high)")
    print("salary  vs cat/mat:     ", round(cosine(vecs[0], vecs[2]), 3), "(should be low)")
