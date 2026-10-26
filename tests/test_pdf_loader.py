from unittest.mock import MagicMock, patch

import pytest

from pdf_loader import extract_text, load_pdf_chunks, split_into_chunks


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t  \n "])
def test_blank_text_yields_no_chunks(text):
    assert split_into_chunks(text) == []


def test_overlap_equal_to_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        split_into_chunks("some text", chunk_size=100, overlap=100)


def test_overlap_larger_than_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        split_into_chunks("some text", chunk_size=100, overlap=250)


def test_text_shorter_than_chunk_size_returns_one_chunk():
    chunks = split_into_chunks("short text", chunk_size=1000, overlap=150)
    assert chunks == ["short text"]


def test_chunks_advance_by_chunk_size_minus_overlap():
    text = "ABCDEFGHIJKLMNOPQRST"

    chunks = split_into_chunks(text, chunk_size=10, overlap=3)

    assert chunks == ["ABCDEFGHIJ", "HIJKLMNOPQ", "OPQRST"]


def test_consecutive_chunks_share_exactly_overlap_characters():
    text = "ABCDEFGHIJKLMNOPQRST"
    overlap = 3

    chunks = split_into_chunks(text, chunk_size=10, overlap=overlap)

    assert chunks[0][-overlap:] == chunks[1][:overlap] == "HIJ"


def test_zero_overlap_produces_no_repeated_text():
    text = "ABCDEFGHIJKLMNOPQRST"

    chunks = split_into_chunks(text, chunk_size=10, overlap=0)

    assert chunks == ["ABCDEFGHIJ", "KLMNOPQRST"]
    assert "".join(chunks) == text


def test_every_character_of_the_source_appears_somewhere():
    text = "".join(str(i % 10) for i in range(250))

    chunks = split_into_chunks(text, chunk_size=40, overlap=10)

    for index in range(0, len(text), 25):
        window = text[index:index + 15]
        assert any(window in chunk for chunk in chunks), f"lost window at {index}"


def test_chunks_are_stripped_of_surrounding_whitespace():
    text = "    padded start" + " " * 40 + "padded end    "

    chunks = split_into_chunks(text, chunk_size=20, overlap=5)

    for chunk in chunks:
        assert chunk == chunk.strip()


def test_whitespace_only_chunks_are_dropped_not_stored_blank():
    text = "START" + " " * 100 + "END"

    chunks = split_into_chunks(text, chunk_size=20, overlap=0)

    assert "" not in chunks
    assert all(chunk.strip() for chunk in chunks)


def _reader_with_pages(*page_texts):
    reader = MagicMock()
    reader.pages = []
    for value in page_texts:
        page = MagicMock()
        page.extract_text.return_value = value
        reader.pages.append(page)
    return reader


def test_extract_text_joins_pages_with_newlines():
    with patch("pdf_loader.PdfReader", return_value=_reader_with_pages("page one", "page two")):
        assert extract_text("whatever.pdf") == "page one\npage two\n"


def test_extract_text_survives_pages_returning_none():
    with patch("pdf_loader.PdfReader", return_value=_reader_with_pages("good", None, "also good")):
        result = extract_text("whatever.pdf")

    assert result == "good\n\nalso good\n"


def test_extract_text_on_empty_pdf_returns_empty_string():
    with patch("pdf_loader.PdfReader", return_value=_reader_with_pages()):
        assert extract_text("empty.pdf") == ""


def test_load_pdf_chunks_passes_sizing_through_to_the_splitter():
    with patch("pdf_loader.extract_text", return_value="ABCDEFGHIJKLMNOPQRST") as extractor:
        chunks = load_pdf_chunks("doc.pdf", chunk_size=10, overlap=3)

    extractor.assert_called_once_with("doc.pdf")
    assert chunks == ["ABCDEFGHIJ", "HIJKLMNOPQ", "OPQRST"]


def test_image_only_pdf_yields_no_chunks():
    with patch("pdf_loader.extract_text", return_value="\n\n\n"):
        assert load_pdf_chunks("scanned.pdf") == []
