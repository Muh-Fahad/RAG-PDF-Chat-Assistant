from unittest.mock import patch

import pytest

from ingest import ingest


def test_unreadable_pdf_stores_nothing_and_leaves_the_database_alone():
    with patch("ingest.load_pdf_chunks", return_value=[]), \
         patch("ingest.reset_collection") as reset, \
         patch("ingest.add_chunks") as add, \
         patch("ingest.count", return_value=0):
        stored = ingest("scanned.pdf")

    assert stored == 0
    reset.assert_not_called()
    add.assert_not_called()


def test_returns_the_number_of_chunks_stored():
    with patch("ingest.load_pdf_chunks", return_value=["a", "b", "c"]), \
         patch("ingest.reset_collection"), \
         patch("ingest.add_chunks", return_value=3), \
         patch("ingest.count", return_value=3):
        assert ingest("doc.pdf") == 3


def test_previous_pdf_is_cleared_by_default():
    with patch("ingest.load_pdf_chunks", return_value=["a"]), \
         patch("ingest.reset_collection") as reset, \
         patch("ingest.add_chunks", return_value=1), \
         patch("ingest.count", return_value=1):
        ingest("doc.pdf")

    reset.assert_called_once()


def test_keep_existing_appends_instead_of_replacing():
    with patch("ingest.load_pdf_chunks", return_value=["a"]), \
         patch("ingest.reset_collection") as reset, \
         patch("ingest.add_chunks", return_value=1), \
         patch("ingest.count", return_value=2):
        ingest("doc.pdf", keep_existing=True)

    reset.assert_not_called()


def test_reset_happens_before_anything_is_added():
    calls = []
    with patch("ingest.load_pdf_chunks", return_value=["a"]), \
         patch("ingest.reset_collection", side_effect=lambda: calls.append("reset")), \
         patch("ingest.add_chunks", side_effect=lambda *a, **k: calls.append("add") or 1), \
         patch("ingest.count", return_value=1):
        ingest("doc.pdf")

    assert calls == ["reset", "add"]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("report.pdf", "report.pdf"),
        ("folder/report.pdf", "report.pdf"),
        (r"C:\Users\hp\Documents\report.pdf", "report.pdf"),
        (r"E:\mixed/separators\report.pdf", "report.pdf"),
        ("./report.pdf", "report.pdf"),
    ],
)
def test_source_is_the_bare_filename_on_either_path_style(path, expected):
    with patch("ingest.load_pdf_chunks", return_value=["a"]), \
         patch("ingest.reset_collection"), \
         patch("ingest.add_chunks", return_value=1) as add, \
         patch("ingest.count", return_value=1):
        ingest(path)

    assert add.call_args.kwargs["source"] == expected


def test_chunks_from_the_loader_are_passed_through_to_storage():
    chunks = ["chunk one", "chunk two"]
    with patch("ingest.load_pdf_chunks", return_value=chunks), \
         patch("ingest.reset_collection"), \
         patch("ingest.add_chunks", return_value=2) as add, \
         patch("ingest.count", return_value=2):
        ingest("doc.pdf")

    assert add.call_args.args[0] == chunks
