from unittest.mock import MagicMock, patch

import generate
from generate import answer, build_prompt


def test_prompt_contains_the_question():
    prompt = build_prompt("What is the salary?", ["some context"])
    assert "What is the salary?" in prompt


def test_prompt_contains_every_retrieved_chunk():
    chunks = ["first fact", "second fact", "third fact"]

    prompt = build_prompt("q", chunks)

    for chunk in chunks:
        assert chunk in prompt


def test_chunks_are_separated_so_they_do_not_run_together():
    prompt = build_prompt("q", ["alpha", "beta"])
    assert "alpha\n\n---\n\nbeta" in prompt


def test_prompt_restricts_the_model_to_the_supplied_context():
    prompt = build_prompt("q", ["ctx"])

    assert "ONLY the context" in prompt
    assert "say you couldn't find it" in prompt


def test_context_is_delimited_so_the_model_can_see_where_it_ends():
    prompt = build_prompt("q", ["ctx"])
    assert "--- CONTEXT ---" in prompt
    assert "--- END CONTEXT ---" in prompt


def test_question_is_placed_after_the_context_block():
    marker = "ZZ-UNIQUE-QUESTION-ZZ"

    prompt = build_prompt(marker, ["ctx"])

    assert prompt.index("--- END CONTEXT ---") < prompt.index(marker)


def test_no_chunks_returns_guidance_without_calling_the_model():
    client = MagicMock()
    with patch("generate.query", return_value=[]), \
         patch.object(generate, "client", client):
        result = answer("anything")

    client.models.generate_content.assert_not_called()
    assert result["chunks"] == []
    assert "No PDF has been ingested yet" in result["answer"]


def test_model_text_is_returned_with_the_chunks_used():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="The salary is 50,000.")

    with patch("generate.query", return_value=["salary: 50,000"]), \
         patch.object(generate, "client", client):
        result = answer("What is the salary?")

    assert result["answer"] == "The salary is 50,000."
    assert result["chunks"] == ["salary: 50,000"]


def test_retrieved_chunks_are_what_gets_sent_to_the_model():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="ok")

    with patch("generate.query", return_value=["the retrieved fact"]), \
         patch.object(generate, "client", client):
        answer("q")

    sent = client.models.generate_content.call_args.kwargs["contents"]
    assert "the retrieved fact" in sent


def test_n_results_is_forwarded_to_retrieval():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="ok")

    with patch("generate.query", return_value=["c"]) as retrieve, \
         patch.object(generate, "client", client):
        answer("q", n_results=7)

    assert retrieve.call_args.kwargs["n_results"] == 7


def test_configured_chat_model_is_used():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="ok")

    with patch("generate.query", return_value=["c"]), \
         patch.object(generate, "client", client):
        answer("q")

    assert client.models.generate_content.call_args.kwargs["model"] == generate.CHAT_MODEL


def test_model_failure_is_reported_without_raising():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("503 backend unavailable")

    with patch("generate.query", return_value=["ctx"]), \
         patch.object(generate, "client", client):
        result = answer("q")

    assert "Error from Gemini" in result["answer"]
    assert "503 backend unavailable" in result["answer"]


def test_chunks_are_still_returned_when_the_model_fails():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("boom")

    with patch("generate.query", return_value=["ctx one", "ctx two"]), \
         patch.object(generate, "client", client):
        result = answer("q")

    assert result["chunks"] == ["ctx one", "ctx two"]


def test_answer_always_returns_both_keys():
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text="ok")

    with patch("generate.query", return_value=["c"]), \
         patch.object(generate, "client", client):
        result = answer("q")

    assert set(result) == {"answer", "chunks"}
