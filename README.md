# 📚 PDF Chat Assistant — RAG Edition

Ask questions about a PDF and get answers grounded in its actual content.

This is the **RAG upgrade** of the simpler "chat with your PDF" app. The old version
stuffed the *entire* PDF into every prompt. This version is smarter: it finds only the
few chunks of text that are relevant to your question and sends *those* to Gemini.

---

## Day 1 — The concepts (read this first)

### What is RAG?

**RAG = Retrieval-Augmented Generation.**

Instead of asking the LLM to answer from memory, we:

1. **Retrieve** — find the pieces of *your* documents that relate to the question.
2. **Augment** — paste those pieces into the prompt as context.
3. **Generate** — let the LLM write the answer using that context.

```
Question ──▶ [ find relevant chunks ] ──▶ [ chunks + question ] ──▶ Gemini ──▶ Answer
                     ▲
              your PDF, pre-processed
```

### Why do LLMs "forget" information?

An LLM does **not** have your documents inside it. Two hard limits:

1. **Training cutoff** — it only knows what it saw during training. Your private PDF,
   last week's report, your company handbook — it has never seen them.
2. **Context window** — the model can only read a limited amount of text per request.
   A 300-page PDF simply does not fit, and even when a big document *does* fit, models
   tend to lose track of details buried in the middle ("lost in the middle"). It also
   costs more and runs slower to send huge prompts every time.

The naive fix — "just paste the whole PDF into the prompt" — is what our older app did.
It breaks on large PDFs and wastes tokens on text that has nothing to do with the question.

### Why are embeddings needed?

To retrieve the *relevant* chunks, we need a way to measure **meaning**, not just keywords.
Searching for the word "salary" would miss a paragraph that says "annual compensation."

An **embedding** turns a piece of text into a list of numbers (a *vector*) that captures its
meaning. Texts with similar meaning get similar vectors — so they sit close together in
"vector space."

```
"annual compensation"  ─▶  [0.12, -0.98, 0.44, ...]
"yearly salary"        ─▶  [0.13, -0.95, 0.41, ...]   ← close by = similar meaning
"the cat sat on a mat" ─▶  [0.88,  0.02, -0.71, ...]  ← far away = unrelated
```

To answer a question we embed the **question**, then ask the vector database:
*"which chunks have vectors closest to this one?"* Those are the chunks we send to Gemini.

---

## How the code is organized (maps to the roadmap)

| Roadmap day | File | What it does |
|---|---|---|
| Day 1 | `README.md` | Concepts (this file) |
| Day 2 | [`pdf_loader.py`](pdf_loader.py) | Read a PDF, extract text, split into chunks |
| Day 3 | [`embeddings.py`](embeddings.py) | Turn chunks into vectors with Gemini |
| Day 3 | [`vector_store.py`](vector_store.py) | Store vectors in ChromaDB |
| Day 4 | [`generate.py`](generate.py) | Search relevant chunks + ask Gemini |
| Day 5 | [`app.py`](app.py) | The complete chat assistant (web UI) |
| — | [`ingest.py`](ingest.py) | Command-line tool to load a PDF into the database |

---

## Setup

```bash
# 1. (optional but recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell

# 2. install dependencies
pip install -r requirements.txt

# 3. add your Gemini API key
copy .env.example .env          # then edit .env and paste your key
```

## Run

```bash
# Option A — the full web app (upload a PDF, then chat)
python app.py

# Option B — ingest from the command line, then chat in the app
python ingest.py sample.pdf
python app.py
```

The first question after ingesting proves the whole loop works end-to-end.
