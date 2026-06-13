"""
app.py
──────
AI-Powered Academic Assistant — Streamlit front-end (Ollama edition).

Run with:
    streamlit run app.py

Requirements:
    - Ollama installed and running  (ollama serve)
    - At least one model pulled     (ollama pull llama3.2)
    - pip install -r requirements.txt
"""

import streamlit as st
from rag_pipeline import RAGPipeline


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Academic Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Fonts & base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Chat bubbles ── */
.user-msg {
    background: #EFF6FF;
    border-left: 4px solid #3B82F6;
    border-radius: 0 8px 8px 8px;
    padding: 12px 16px;
    margin: 10px 0 4px 0;
}
.assistant-msg {
    background: #F0FDF4;
    border-left: 4px solid #22C55E;
    border-radius: 0 8px 8px 8px;
    padding: 12px 16px;
    margin: 4px 0 10px 0;
}

/* ── Mode badges ── */
.badge-syllabus {
    background: #DBEAFE; color: #1D4ED8;
    border-radius: 12px; padding: 2px 10px;
    font-size: 0.78rem; font-weight: 600;
}
.badge-research {
    background: #F3E8FF; color: #7E22CE;
    border-radius: 12px; padding: 2px 10px;
    font-size: 0.78rem; font-weight: 600;
}

/* ── Source cards ── */
.source-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.82rem;
    line-height: 1.5;
}

/* ── Stat boxes ── */
.stat-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    font-size: 0.9rem;
}

/* ── Ollama status ── */
.ollama-ok  { color: #16A34A; font-weight: 600; }
.ollama-err { color: #DC2626; font-weight: 600; }

/* ── Example buttons ── */
div[data-testid="stHorizontalBlock"] button {
    text-align: left !important;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []        # {"role", "content", "meta"}

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

pipeline: RAGPipeline = st.session_state.pipeline


# ── Utility helpers ────────────────────────────────────────────────────────────

def mode_badge(mode: str) -> str:
    if mode == "research":
        return '<span class="badge-research"> Research Mode</span>'
    return '<span class="badge-syllabus"> Syllabus Mode</span>'


def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f" Sources used ({len(sources)} chunk(s))", expanded=False):
        for i, chunk in enumerate(sources, 1):
            preview = chunk["text"][:300]
            ellipsis = "…" if len(chunk["text"]) > 300 else ""
            st.markdown(
                f'<div class="source-card">'
                f'<strong>[{i}]</strong> {chunk["source"]} — '
                f'Page/Section {chunk["page"]}<br>'
                f'<em>{preview}{ellipsis}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )


def click_example(text: str):
    st.session_state.pending_query = text


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 Academic Assistant")
    st.caption("Local RAG · Powered by Ollama")

    # ── Ollama status ──────────────────────────────────────────────────────────
    st.markdown("###  Ollama Status")
    ollama_ok, ollama_msg = RAGPipeline.check_ollama()
    if ollama_ok:
        st.markdown(f'<p class="ollama-ok">● Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="ollama-err">● Not running</p>', unsafe_allow_html=True)
        st.code("ollama serve", language="bash")
    st.caption(ollama_msg)

    st.divider()

    # ── Model selection ────────────────────────────────────────────────────────
    st.markdown("###  Model")
    local_models = RAGPipeline.list_local_models()

    # Preferred defaults shown first if available
    preferred = ["llama3.2", "llama3.1", "mistral", "phi3", "gemma2",
                 "llama3.3", "llama3.2:1b", "codellama", "deepseek-r1"]
    all_options = local_models + [m for m in preferred if m not in local_models]

    selected_model = st.selectbox(
        "Choose model",
        options=all_options,
        index=0,
        help="Models shown in bold are already pulled on your machine.",
    )
    pipeline.model = selected_model

    temp = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0,
        value=pipeline.temperature, step=0.05,
        help="Lower = more factual. Higher = more creative.",
    )
    pipeline.temperature = temp

    st.divider()

    # ── Document upload ────────────────────────────────────────────────────────
    st.markdown("###  Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Syllabus, lecture notes, textbook chapters, research papers…",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.ingested_files:
                with st.spinner(f"Indexing {uf.name}…"):
                    added, msg = pipeline.ingest(uf.read(), uf.name)
                st.markdown(msg)
                if added > 0:
                    st.session_state.ingested_files.add(uf.name)

    st.divider()

    # ── Indexed documents ──────────────────────────────────────────────────────
    st.markdown("###  Indexed Documents")
    doc_list = pipeline.document_list()

    if not doc_list:
        st.caption("No documents indexed yet.")
    else:
        for doc in doc_list:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f" `{doc}`")
            if col2.button("✕", key=f"del_{doc}", help=f"Remove {doc}"):
                _, rmsg = pipeline.remove_document(doc)
                st.session_state.ingested_files.discard(doc)
                st.success(rmsg)
                st.rerun()

    st.divider()

    # ── Knowledge base stats ───────────────────────────────────────────────────
    st.markdown("###  Knowledge Base")
    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="stat-box"><strong>{len(doc_list)}</strong><br>Docs</div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="stat-box"><strong>{pipeline.chunk_count()}</strong><br>Chunks</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Clear actions ──────────────────────────────────────────────────────────
    ca, cb = st.columns(2)
    if ca.button(" Clear Docs", use_container_width=True):
        pipeline.clear_all()
        st.session_state.ingested_files.clear()
        st.rerun()
    if cb.button(" Clear Chat", use_container_width=True):
        st.session_state.chat_history.clear()
        st.rerun()

    st.divider()
    st.caption(
        "**Mode is detected automatically:**\n\n"
        " **Syllabus** — answers only from your uploaded docs\n\n"
        " **Research** — structured academic guidance"
    )


# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("# 🎓 AI Academic Assistant")
st.caption(
    "Ask syllabus questions grounded in your course documents, or explore any "
    "topic for research papers, project ideas, scope, and keywords — all running "
    "locally on your machine."
)

# ── Example queries (shown only when chat is empty) ───────────────────────────
if not st.session_state.chat_history:
    st.markdown("###  Try one of these…")
    ex_col1, ex_col2 = st.columns(2)

    syllabus_examples = [
        "What topics are covered in Week 3?",
        "Explain the grading policy.",
        "What are the course prerequisites?",
        "Summarise the learning objectives.",
    ]
    research_examples = [
        "Find related papers on transformer architectures.",
        "Give me project ideas on federated learning.",
        "What is the scope of reinforcement learning in robotics?",
        "Suggest keywords for graph neural networks research.",
    ]

    with ex_col1:
        st.markdown("** Syllabus Questions**")
        for ex in syllabus_examples:
            st.button(ex, key=f"s_{ex}", use_container_width=True,
                      on_click=click_example, args=(ex,))

    with ex_col2:
        st.markdown("** Research Guidance**")
        for ex in research_examples:
            st.button(ex, key=f"r_{ex}", use_container_width=True,
                      on_click=click_example, args=(ex,))

    st.divider()

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">'
            f' <strong>You</strong><br>{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        meta  = msg.get("meta", {})
        badge = mode_badge(meta.get("mode", "syllabus"))
        # Render markdown in assistant answers properly
        st.markdown(
            f'<div class="assistant-msg">'
            f' <strong>Assistant</strong> &nbsp;{badge}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(msg["content"])           # full markdown rendering
        render_sources(meta.get("sources", []))

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your question",
        value=st.session_state.pending_query,
        height=90,
        placeholder=(
            "e.g. What are the Week 4 topics?  "
            "·  Find papers on attention mechanisms.  "
            "·  Give me federated learning project ideas."
        ),
        label_visibility="collapsed",
    )
    submit = st.form_submit_button("Send ➤", use_container_width=True)

# Clear pending query after it's been pre-filled
if st.session_state.pending_query:
    st.session_state.pending_query = ""

# ── Handle submission ──────────────────────────────────────────────────────────
if submit and user_input.strip():
    question = user_input.strip()

    # Append user turn to history
    st.session_state.chat_history.append({
        "role":    "user",
        "content": question,
    })

    # Check Ollama is reachable before generating
    ollama_alive, _ = RAGPipeline.check_ollama()
    if not ollama_alive:
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": (
                " **Ollama is not running.**\n\n"
                "Start it in a terminal:\n```\nollama serve\n```\n"
                "Then send your question again."
            ),
            "meta": {"mode": "syllabus", "sources": []},
        })
    else:
        with st.spinner(f"🔍 Retrieving context · generating with **{pipeline.model}**…"):
            result = pipeline.query(question, st.session_state.chat_history[:-1])

        # Append a nudge if syllabus mode but no docs uploaded
        answer = result["answer"]
        if result["no_docs"] and result["mode"] == "syllabus":
            answer += (
                "\n\n>  **No documents uploaded.** "
                "Upload your course files in the sidebar for source-grounded answers."
            )

        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": answer,
            "meta": {
                "mode":    result["mode"],
                "sources": result["sources"],
            },
        })

    st.rerun()