
import gradio as gr

from ingest import ingest
from generate import answer


# --------------------------------------------------------------------------- #
# Logic (unchanged behaviour, just wired to a nicer UI)
# --------------------------------------------------------------------------- #
def load_pdf(file):
    """Ingest the uploaded PDF into the vector database and report status."""
    if file is None:
        return "### ⚠️ No file yet\nPlease choose a PDF file to get started."

    try:
        n = ingest(file)
    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            return ("### ⚠️ Quota limit hit\n"
                    "Gemini's embedding quota/rate limit was reached while embedding "
                    "this PDF. It's too big for the free tier — try a smaller PDF, or "
                    "switch to local embeddings.")
        return f"### ⚠️ Ingest failed\n```\n{msg[:300]}\n```"

    if n == 0:
        return ("### ⚠️ No readable text\n"
                "Couldn't extract any text. Is this a scanned / image-only PDF?")

    return (f"### ✅ Ready to chat\n"
            f"Ingested **{n} chunks**. Ask your questions on the right →")


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


def clear_chat():
    """Empty the conversation."""
    return []


def use_example(example):
    """Put an example question into the input box."""
    return example


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1150px !important; margin: 0 auto !important; }

/* Hero header */
#hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #a855f7 100%);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
#hero h1 { color: #fff !important; font-size: 2rem !important; margin: 0 0 6px 0 !important; }
#hero p  { color: rgba(255,255,255,0.92) !important; margin: 0 !important; font-size: 1.02rem !important; }

/* Card panels */
.panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px !important;
    padding: 16px !important;
    background: var(--background-fill-secondary);
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

/* Status box */
#status_box {
    border-radius: 12px !important;
    padding: 14px 16px !important;
    background: var(--background-fill-primary);
    border: 1px dashed var(--border-color-primary);
}

/* Chat window */
#chatbox { border-radius: 14px !important; }

/* Example chips */
.example-row button {
    border-radius: 999px !important;
    font-size: 0.85rem !important;
}

footer { display: none !important; }
#footnote { text-align: center; color: var(--body-text-color-subdued); font-size: 0.85rem; margin-top: 10px; }
"""

EXAMPLES = [
    "What is this document about?",
    "Summarize the key points.",
    "What are the main conclusions?",
]

with gr.Blocks(title="Chat with your PDF — RAG") as demo:

    gr.HTML(
        """
        <div id="hero">
            <h1>📚 Chat with your PDF</h1>
            <p>Upload a document, then ask anything. Answers are grounded in the most
            relevant passages retrieved from your PDF — powered by RAG.</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # ---- Left: upload + status ---- #
        with gr.Column(scale=2, elem_classes="panel"):
            gr.Markdown("### 1 · Upload")
            pdf_file = gr.File(
                label="Drop a PDF here",
                file_types=[".pdf"],
                type="filepath",
                height=150,
            )
            status = gr.Markdown(
                "### 📄 Waiting for a PDF\nUpload a file to begin.",
                elem_id="status_box",
            )
            gr.Markdown(
                "**💡 Tips**\n"
                "- Text-based PDFs work best (not scanned images)\n"
                "- Larger files take longer to embed\n"
                "- Answers cite the passages they came from"
            )

        # ---- Right: chat ---- #
        with gr.Column(scale=3, elem_classes="panel"):
            gr.Markdown("### 2 · Ask")
            chatbot = gr.Chatbot(
                height=430,
                label="Conversation",
                elem_id="chatbox",
                placeholder="<center><h3>👋 Ask me about your PDF</h3>"
                            "<p>Upload a document, then type a question below.</p></center>",
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask something about the PDF and press Enter…",
                    show_label=False,
                    scale=8,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=90)
                clear_btn = gr.Button("🗑️", variant="secondary", scale=1, min_width=50)

            with gr.Row(elem_classes="example-row"):
                gr.Markdown("**Try:**")
                ex_buttons = [gr.Button(q, size="sm") for q in EXAMPLES]

    gr.HTML("<div id='footnote'>Built with RAG · Gemini embeddings · ChromaDB · Gradio</div>")

    # ---- Wiring ---- #
    pdf_file.change(load_pdf, inputs=[pdf_file], outputs=[status])

    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    send_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(clear_chat, outputs=[chatbot])

    for btn, q in zip(ex_buttons, EXAMPLES):
        btn.click(use_example, inputs=gr.State(q), outputs=[msg])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
