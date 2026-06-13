"""
app.py
AI-Powered Academic Assistant — Streamlit front-end (Ollama edition).

Run with:
    streamlit run app.py
"""

from html import escape

import streamlit as st
from rag_pipeline import RAGPipeline


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Scholar — AI Academic Assistant",
    page_icon="book",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Design system ──────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&family=DM+Mono:wght@400;500&family=Material+Symbols+Rounded:wght@400');

:root {
    --bg: #f6f1e8;
    --panel: #ffffff;
    --panel-2: #f3efe8;
    --text: #1f2937;
    --muted: #64748b;
    --line: #d8dee8;
    --sidebar: #111827;
    --sidebar-2: #0b1220;
    --accent: #d97706;
    --accent-2: #b45309;
    --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}

*, *::before, *::after {
    box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .block-container {
    background: var(--bg) !important;
    color: var(--text) !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

/* Top bar / sidebar toggle */
[data-testid="stHeader"] {
    background: transparent !important;
}
button[data-testid="collapsedControl"] {
    font-family: "Material Symbols Rounded" !important;
    font-size: 20px !important;
    line-height: 1 !important;
    color: #cbd5e1 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
button[data-testid="collapsedControl"] * {
    font-family: "Material Symbols Rounded" !important;
}
button[data-testid="collapsedControl"] svg {
    display: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar) 0%, var(--sidebar-2) 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}
[data-testid="stSidebar"] * {
    color: #dbe4f0 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #f8fafc !important;
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #94a3b8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #1f2937 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #1f2937 !important;
    border: 1px dashed #475569 !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: #1f2937 !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 0.9rem !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #334155 !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}

/* Main typography */
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    color: var(--text) !important;
    font-weight: 400 !important;
}
p, li, span, label, caption, div {
    font-family: 'Inter', sans-serif !important;
}

/* Main area buttons */
[data-testid="stButton"] > button {
    background: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.8rem 1rem !important;
    box-shadow: none !important;
    text-align: left !important;
    white-space: normal !important;
    line-height: 1.35 !important;
}
[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    background: #fff7ed !important;
    color: #111827 !important;
}

/* Form submit button only */
[data-testid="stForm"] [data-testid="stButton"] > button {
    background: var(--sidebar) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    width: 100% !important;
    padding: 0.85rem 1rem !important;
    box-shadow: var(--shadow) !important;
    text-align: center !important;
}
[data-testid="stForm"] [data-testid="stButton"] > button:hover {
    background: var(--accent) !important;
}

/* Textarea */
[data-testid="stForm"] textarea {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.98rem !important;
    padding: 14px 16px !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stForm"] textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.12) !important;
}

/* User bubble */
.user-bubble {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin: 16px 0 6px 0;
}
.user-bubble .bubble-label {
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}
.user-bubble .bubble-inner {
    background: var(--sidebar);
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    max-width: 75%;
    font-size: 0.96rem;
    line-height: 1.6;
    box-shadow: var(--shadow);
}

/* Assistant bubble */
.assistant-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.assistant-name {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}
.assistant-body {
    background: #ffffff;
    border-radius: 6px 18px 18px 18px;
    padding: 16px 20px;
    border-left: 4px solid var(--accent);
    font-size: 0.96rem;
    line-height: 1.7;
    color: #1e293b;
    box-shadow: var(--shadow);
    max-width: 90%;
}
.assistant-body p, .assistant-body li, .assistant-body span {
    color: #1e293b !important;
}
.assistant-body h2, .assistant-body h3 {
    color: #0f172a !important;
}
.assistant-body code {
    font-family: 'DM Mono', monospace;
    font-size: 0.86em;
    background: #f1f5f9;
    padding: 1px 5px;
    border-radius: 4px;
    color: #0f172a;
}

/* Pills */
.pill {
    display: inline-block;
    font-size: 0.68rem;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 999px;
}
.pill-syllabus { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.pill-research { background: #ede9fe; color: #5b21b6; border: 1px solid #ddd6fe; }

/* Source reference */
.source-ref {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
.source-ref strong { color: #0f172a; font-weight: 700; }
.source-ref em { color: #64748b; font-style: normal; }

/* Example labels */
.ex-section-label {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b7280;
    margin-bottom: 10px;
    margin-top: 4px;
}

/* Stats */
.stat-row { display: flex; gap: 8px; margin: 6px 0; }
.stat-chip {
    flex: 1;
    background: #1f2937;
    border-radius: 12px;
    padding: 10px 12px;
    text-align: center;
}
.stat-chip .num {
    display: block;
    font-size: 1.2rem;
    font-family: 'DM Serif Display', Georgia, serif;
    color: #ffffff;
    line-height: 1.1;
}
.stat-chip .lbl {
    display: block;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-top: 2px;
    font-weight: 700;
}

/* Status */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-ok { background: #22c55e; box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.18); }
.dot-err { background: #ef4444; box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.18); }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    color: #64748b !important;
    font-weight: 600 !important;
}

/* Misc */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 999px; }
[data-testid="stSpinner"] p { color: #64748b !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }

/* Narrow remove buttons */
div[data-testid="column"] [data-testid="stButton"] > button {
    min-width: 44px !important;
    padding: 0.35rem 0.5rem !important;
    text-align: center !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ──────────────────────────────────────────────────────────────

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

pipeline: RAGPipeline = st.session_state.pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────

def pill(mode: str) -> str:
    if mode == "research":
        return '<span class="pill pill-research">Research</span>'
    return '<span class="pill pill-syllabus">Syllabus</span>'


def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"Source chunks retrieved: {len(sources)}", expanded=False):
        for i, chunk in enumerate(sources, 1):
            preview = escape(chunk["text"][:280])
            ellipsis = "..." if len(chunk["text"]) > 280 else ""
            source = escape(str(chunk["source"]))
            page = escape(str(chunk["page"]))
            st.markdown(
                f'<div class="source-ref">'
                f'<strong>[{i}] {source}</strong> · page/section {page}<br>'
                f'<em>{preview}{ellipsis}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )


def click_example(text: str):
    st.session_state.pending_query = text


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='margin-bottom:2px;'>Scholar</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.75rem;color:#475569;margin-top:0;font-family:Inter,sans-serif;'>"
        "AI Academic Assistant · runs locally</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    ollama_ok, ollama_msg = RAGPipeline.check_ollama()
    dot_cls = "dot-ok" if ollama_ok else "dot-err"
    dot_lbl = "Ollama connected" if ollama_ok else "Ollama not running"
    st.markdown(
        f'<p style="font-size:0.8rem;font-family:Inter,sans-serif;margin:0;">'
        f'<span class="status-dot {dot_cls}"></span>{dot_lbl}</p>',
        unsafe_allow_html=True,
    )
    if not ollama_ok:
        st.code("ollama serve", language="bash")
    st.caption(ollama_msg)
    st.divider()

    st.markdown("#### Model")
    local_models = RAGPipeline.list_local_models()
    preferred = [
        "llama3.2",
        "llama3.2:latest",
        "llama3.1",
        "mistral",
        "phi3",
        "gemma2",
        "llama3.3",
        "llama3.2:1b",
        "codellama",
    ]
    all_options = local_models + [m for m in preferred if m not in local_models]
    if not all_options:
        all_options = ["llama3.2"]

    selected_model = st.selectbox(
        "Active model",
        options=all_options,
        index=0,
        label_visibility="collapsed",
    )
    pipeline.model = selected_model
    pipeline.temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        pipeline.temperature,
        0.05,
        help="0 = precise, 1 = creative",
    )
    st.divider()

    st.markdown("#### Documents")
    st.caption("Upload PDFs, DOCX, TXT, or MD files")

    with st.container():
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.ingested_files:
                with st.spinner(f"Indexing {uf.name}..."):
                    added, msg = pipeline.ingest(uf.read(), uf.name)
                if added > 0:
                    st.success(f"Indexed {added} chunks")
                    st.session_state.ingested_files.add(uf.name)
                else:
                    st.warning(msg)

    st.divider()

    doc_list = pipeline.document_list()
    if doc_list:
        st.markdown("#### Indexed files")
        for doc in doc_list:
            c1, c2 = st.columns([0.9, 0.1])
            c1.markdown(
                f'<p style="font-size:0.82rem;font-family:Inter,sans-serif;'
                f'color:#dbe4f0;margin:4px 0;overflow:hidden;text-overflow:ellipsis;'
                f'white-space:nowrap;" title="{escape(doc)}">{escape(doc)}</p>',
                unsafe_allow_html=True,
            )
            if c2.button("×", key=f"del_{doc}", use_container_width=True):
                pipeline.remove_document(doc)
                st.session_state.ingested_files.discard(doc)
                st.rerun()

    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-chip"><span class="num">{len(doc_list)}</span><span class="lbl">Docs</span></div>'
        f'<div class="stat-chip"><span class="num">{pipeline.chunk_count()}</span><span class="lbl">Chunks</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    ca, cb = st.columns(2)
    if ca.button("Clear docs", use_container_width=True):
        pipeline.clear_all()
        st.session_state.ingested_files.clear()
        st.rerun()
    if cb.button("Clear chat", use_container_width=True):
        st.session_state.chat_history.clear()
        st.rerun()

    st.markdown(
        '<p style="font-size:0.7rem;color:#475569;font-family:Inter,sans-serif;'
        'line-height:1.6;margin-top:12px;">'
        'Mode is detected from your question.<br>'
        '<span style="color:#92400E;">Syllabus</span> — answers from your docs<br>'
        '<span style="color:#5B21B6;">Research</span> — papers, ideas, scope'
        '</p>',
        unsafe_allow_html=True,
    )


# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="font-size:2.2rem;margin-bottom:2px;letter-spacing:-0.02em;">'
    'Your Academic Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:0.95rem;color:#64748B;margin-top:0;margin-bottom:24px;'
    'font-family:Inter,sans-serif;max-width:600px;">'
    'Ask questions grounded in your course documents, or explore any topic '
    'for research papers, project ideas, and literature keywords.'
    '</p>',
    unsafe_allow_html=True,
)

if not st.session_state.chat_history:
    col_a, col_b = st.columns(2)

    syllabus_examples = [
        "What topics are covered in Week 3?",
        "Explain the grading policy.",
        "What are the course prerequisites?",
        "Summarise the learning objectives.",
    ]
    research_examples = [
        "Find related papers on transformer architectures.",
        "Give me project ideas on federated learning.",
        "What is the scope of RL in robotics?",
        "Keywords for graph neural networks research.",
    ]

    with col_a:
        st.markdown('<p class="ex-section-label">From your syllabus</p>', unsafe_allow_html=True)
        for ex in syllabus_examples:
            st.button(
                ex,
                key=f"s_{ex}",
                use_container_width=True,
                on_click=click_example,
                args=(ex,),
            )

    with col_b:
        st.markdown('<p class="ex-section-label">Research guidance</p>', unsafe_allow_html=True)
        for ex in research_examples:
            st.button(
                ex,
                key=f"r_{ex}",
                use_container_width=True,
                on_click=click_example,
                args=(ex,),
            )

    st.markdown("<br>", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">'
            f'<div class="bubble-label">You</div>'
            f'<div class="bubble-inner">{escape(msg["content"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        meta = msg.get("meta", {})
        mode = meta.get("mode", "syllabus")

        st.markdown(
            f'<div class="assistant-header">'
            f'<span class="assistant-name">Scholar</span>'
            f'{pill(mode)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="assistant-body">', unsafe_allow_html=True)
        st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        render_sources(meta.get("sources", []))
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ── Input ──────────────────────────────────────────────────────────────────────

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Ask anything",
        value=st.session_state.pending_query,
        height=90,
        placeholder="Ask about your syllabus, or explore a research topic...",
        label_visibility="collapsed",
    )
    submit = st.form_submit_button("Send", use_container_width=True)

if st.session_state.pending_query:
    st.session_state.pending_query = ""


# ── Handle submission ──────────────────────────────────────────────────────────

if submit and user_input.strip():
    question = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": question})

    ollama_alive, _ = RAGPipeline.check_ollama()
    if not ollama_alive:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": "Ollama is not running. Start it with `ollama serve` and try again.",
                "meta": {"mode": "syllabus", "sources": []},
            }
        )
    else:
        with st.spinner(f"Thinking with {pipeline.model}..."):
            result = pipeline.query(question, st.session_state.chat_history[:-1])

        answer = result["answer"]
        if result["no_docs"] and result["mode"] == "syllabus":
            answer += "\n\n*No documents uploaded yet — add files in the sidebar for source-grounded answers.*"

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "meta": {"mode": result["mode"], "sources": result["sources"]},
            }
        )

    st.rerun()