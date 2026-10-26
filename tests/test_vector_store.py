from unittest.mock import MagicMock, patch

import vector_store
from vector_store import add_chunks, count, query, reset_collection


def _collection(count_value=0, documents=None):
    collection = MagicMock()
    collection.count.return_value = count_value
    collection.query.return_value = {"documents": [documents or []]}
    return collection


def test_empty_chunk_list_short_circuits_before_touching_chroma():
    with patch("vector_store.get_collection") as get_collection:
        assert add_chunks([]) == 0

    get_collection.assert_not_called()


def test_add_chunks_returns_the_number_stored():
    with patch("vector_store.get_collection", return_value=_collection()):
        assert add_chunks(["a", "b", "c"]) == 3


def test_ids_are_source_prefixed_and_sequential():
    collection = _collection()
    with patch("vector_store.get_collection", return_value=collection):
        add_chunks(["a", "b", "c"], source="report.pdf")

    assert collection.add.call_args.kwargs["ids"] == [
        "report.pdf-0",
        "report.pdf-1",
        "report.pdf-2",
    ]


def test_each_chunk_records_its_source_and_position():
    collection = _collection()
    with patch("vector_store.get_collection", return_value=collection):
        add_chunks(["a", "b"], source="report.pdf")

    assert collection.add.call_args.kwargs["metadatas"] == [
        {"source": "report.pdf", "chunk_index": 0},
        {"source": "report.pdf", "chunk_index": 1},
    ]


def test_documents_are_passed_through_unmodified():
    collection = _collection()
    chunks = ["first chunk", "second chunk"]
    with patch("vector_store.get_collection", return_value=collection):
        add_chunks(chunks)

    assert collection.add.call_args.kwargs["documents"] == chunks


def test_default_source_is_used_when_none_given():
    collection = _collection()
    with patch("vector_store.get_collection", return_value=collection):
        add_chunks(["a"])

    assert collection.add.call_args.kwargs["ids"] == ["pdf-0"]


def test_query_on_empty_database_returns_nothing_and_embeds_nothing():
    with patch("vector_store.get_collection", return_value=_collection(count_value=0)), \
         patch("vector_store.embed_texts") as embed:
        assert query("anything") == []

    embed.assert_not_called()


def test_question_is_embedded_as_a_query_not_a_document():
    collection = _collection(count_value=10, documents=["chunk"])
    with patch("vector_store.get_collection", return_value=collection), \
         patch("vector_store.embed_texts", return_value=[[0.1] * 4]) as embed:
        query("what is the salary?")

    assert embed.call_args.kwargs["task_type"] == "RETRIEVAL_QUERY"
    assert embed.call_args.args[0] == ["what is the salary?"]


def test_query_returns_the_stored_chunk_text():
    collection = _collection(count_value=5, documents=["chunk one", "chunk two"])
    with patch("vector_store.get_collection", return_value=collection), \
         patch("vector_store.embed_texts", return_value=[[0.1] * 4]):
        assert query("q") == ["chunk one", "chunk two"]


def test_n_results_is_clamped_to_what_is_actually_stored():
    collection = _collection(count_value=2, documents=["a", "b"])
    with patch("vector_store.get_collection", return_value=collection), \
         patch("vector_store.embed_texts", return_value=[[0.1] * 4]):
        query("q", n_results=10)

    assert collection.query.call_args.kwargs["n_results"] == 2


def test_n_results_is_respected_when_below_the_stored_count():
    collection = _collection(count_value=50, documents=["a"])
    with patch("vector_store.get_collection", return_value=collection), \
         patch("vector_store.embed_texts", return_value=[[0.1] * 4]):
        query("q", n_results=4)

    assert collection.query.call_args.kwargs["n_results"] == 4


def test_reset_deletes_the_named_collection():
    client = MagicMock()
    with patch("vector_store.chromadb.PersistentClient", return_value=client):
        reset_collection()

    client.delete_collection.assert_called_once_with(vector_store.COLLECTION_NAME)


def test_reset_is_a_no_op_when_the_collection_does_not_exist():
    client = MagicMock()
    client.delete_collection.side_effect = Exception("collection not found")

    with patch("vector_store.chromadb.PersistentClient", return_value=client):
        reset_collection()


def test_count_reports_the_collection_size():
    with patch("vector_store.get_collection", return_value=_collection(count_value=42)):
        assert count() == 42
