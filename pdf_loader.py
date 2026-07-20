
from pypdf import PdfReader


def extract_text(file) -> str:
    """Return all readable text from a PDF.

    `file` can be a path string or a file-like object (Gradio gives us a path).
    """
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split `text` into overlapping chunks of roughly `chunk_size` characters.

    Example with chunk_size=1000, overlap=150:
        chunk 1 -> characters    0 .. 1000
        chunk 2 -> characters  850 .. 1850   (last 150 chars repeat)
        chunk 3 -> characters 1700 .. 2700
        ...

    The overlap keeps ideas that straddle a boundary intact.
    """
    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def load_pdf_chunks(file, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Convenience helper: PDF file -> list of text chunks in one call."""
    text = extract_text(file)
    return split_into_chunks(text, chunk_size, overlap)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    chunks = load_pdf_chunks(path)
    print(f"Extracted {len(chunks)} chunks from {path}")
    if chunks:
        print("\n--- first chunk preview ---")
        print(chunks[0][:400])
