
import gradio as gr

from ingest import ingest
from generate import answer


def load_pdf(file):
    """Ingest the uploaded PDF into the vector database and report status."""
    if file is None:
        return "⚠️ Please choose a PDF file first."

    try:
        n = ingest(file)
    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            return ("⚠️ Gemini embedding quota/rate limit hit while embedding this PDF. "
                    "It's too big for the free tier. Try a smaller PDF, or switch to "
                    "local embeddings.")
        return f"⚠️ Ingest failed: {msg[:300]}"

    if n == 0:
        return "⚠️ No readable text found. Is this a scanned/image-only PDF?"

    return f"✅ Ingested {n} chunks. Ask your questions below!"


def respond(message, chat_history):
    """Answer the user's question using the ingested PDF (RAG)."""
    if not message.strip():
        return "", chat_history

    result = answer(message)
    reply = result["answer"]

    if result["chunks"]:
        sources = "\n\n".join(
            f"**[{i + 1}]** {c[:300]}{'…' if len(c) > 300 else ''}"
            for i, c in enumerate(result["chunks"])
        )
        reply += f"\n\n<details><summary>📎 Sources used</summary>\n\n{sources}\n</details>"

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", chat_history


with gr.Blocks(title="Chat with your PDF (RAG)") as demo:
    gr.Markdown("# 📚 Chat with your PDF — RAG Edition")
    gr.Markdown(
        "Upload a PDF, then ask questions. Answers are grounded in the most "
        "relevant chunks retrieved from the document (not the whole PDF)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            status = gr.Markdown("No PDF ingested yet.")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=460, label="Q&A")
            msg = gr.Textbox(
                placeholder="Ask something about the PDF and press Enter...",
                show_label=False,
            )

    pdf_file.change(load_pdf, inputs=[pdf_file], outputs=[status])

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])


if __name__ == "__main__":
    demo.launch()
