from unittest.mock import MagicMock

import pytest
from google.genai import errors

import embeddings
from embeddings import BATCH_SIZE, MAX_RETRIES, GeminiEmbeddingFunction, embed_texts


def _result_for(texts):
    result = MagicMock()
    result.embeddings = [MagicMock(values=[float(i)] * 4) for i in range(len(texts))]
    return result


def _client_returning_one_vector_per_text():
    client = MagicMock()
    client.models.embed_content.side_effect = lambda model, contents, config: _result_for(contents)
    return client


def _client_error(code):
    err = errors.ClientError.__new__(errors.ClientError)
    err.code = code
    return err


def test_small_input_is_sent_as_a_single_call(monkeypatch):
    client = _client_returning_one_vector_per_text()
    monkeypatch.setattr(embeddings, "client", client)

    vectors = embed_texts(["a", "b", "c"])

    assert len(vectors) == 3
    assert client.models.embed_content.call_count == 1


def test_input_is_split_into_batches_of_batch_size(monkeypatch):
    client = _client_returning_one_vector_per_text()
    monkeypatch.setattr(embeddings, "client", client)
    texts = [f"chunk {i}" for i in range(120)]

    vectors = embed_texts(texts)

    assert len(vectors) == 120
    assert client.models.embed_content.call_count == 3

    sent = [len(call.kwargs["contents"]) for call in client.models.embed_content.call_args_list]
    assert sent == [BATCH_SIZE, BATCH_SIZE, 20]


def test_exactly_one_batch_is_not_split(monkeypatch):
    client = _client_returning_one_vector_per_text()
    monkeypatch.setattr(embeddings, "client", client)

    embed_texts([f"c{i}" for i in range(BATCH_SIZE)])

    assert client.models.embed_content.call_count == 1


def test_vectors_come_back_in_submission_order(monkeypatch):
    client = MagicMock()
    counter = {"n": 0}

    def ordered(model, contents, config):
        result = MagicMock()
        result.embeddings = []
        for _ in contents:
            result.embeddings.append(MagicMock(values=[float(counter["n"])]))
            counter["n"] += 1
        return result

    client.models.embed_content.side_effect = ordered
    monkeypatch.setattr(embeddings, "client", client)

    vectors = embed_texts([f"c{i}" for i in range(120)])

    assert [v[0] for v in vectors] == [float(i) for i in range(120)]


def test_empty_input_makes_no_api_call(monkeypatch):
    client = _client_returning_one_vector_per_text()
    monkeypatch.setattr(embeddings, "client", client)

    assert embed_texts([]) == []
    client.models.embed_content.assert_not_called()


def test_documents_are_embedded_as_retrieval_document_by_default(monkeypatch):
    client = _client_returning_one_vector_per_text()
    monkeypatch.setattr(embeddings, "client", client)

    embed_texts(["stored chunk"])

    assert client.models.embed_content.call_args.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_query_task_type_is_passed_through(monkeypatch):
    client = _client_returning_one_vector_per_text()
    monkeypatch.setattr(embeddings, "client", client)

    embed_texts(["a question"], task_type="RETRIEVAL_QUERY")

    assert client.models.embed_content.call_args.kwargs["config"].task_type == "RETRIEVAL_QUERY"


def test_chroma_adapter_always_embeds_as_documents(monkeypatch):
    client = _client_returning_one_vector_per_text()
    monkeypatch.setattr(embeddings, "client", client)

    GeminiEmbeddingFunction()(["one", "two"])

    assert client.models.embed_content.call_args.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_rate_limit_is_retried_until_it_succeeds(monkeypatch):
    client = MagicMock()
    client.models.embed_content.side_effect = [
        _client_error(429),
        _client_error(429),
        _result_for(["a"]),
    ]
    monkeypatch.setattr(embeddings, "client", client)

    vectors = embed_texts(["a"])

    assert len(vectors) == 1
    assert client.models.embed_content.call_count == 3


def test_backoff_doubles_between_retries(monkeypatch):
    waits = []
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: waits.append(s))

    client = MagicMock()
    client.models.embed_content.side_effect = [
        _client_error(429),
        _client_error(429),
        _result_for(["a"]),
    ]
    monkeypatch.setattr(embeddings, "client", client)

    embed_texts(["a"])

    assert waits == [2.0, 4.0]


def test_retries_are_capped_and_then_the_error_escapes(monkeypatch):
    client = MagicMock()
    client.models.embed_content.side_effect = _client_error(429)
    monkeypatch.setattr(embeddings, "client", client)

    with pytest.raises(errors.ClientError):
        embed_texts(["a"])

    assert client.models.embed_content.call_count == MAX_RETRIES


def test_non_rate_limit_errors_fail_immediately(monkeypatch):
    client = MagicMock()
    client.models.embed_content.side_effect = _client_error(400)
    monkeypatch.setattr(embeddings, "client", client)

    with pytest.raises(errors.ClientError):
        embed_texts(["a"])

    assert client.models.embed_content.call_count == 1
