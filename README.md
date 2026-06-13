# AI-Powered Academic Assistant

> **Course-Aware Question Answering and Research Guidance for University Students**

A Streamlit web app that lets students upload course documents (PDF, DOCX, TXT, MD) and get accurate, grounded answers to syllabus questions and research-guidance queries — powered by RAG (Retrieval-Augmented Generation) and a **fully local Ollama model**. No API key. No internet required after setup.

---

## Project Structure

```
academic_assistant/
├── app.py               # Streamlit UI — the entry point
├── rag_pipeline.py      # Orchestrates retrieval + Ollama generation
├── document_loader.py   # Parses PDF / DOCX / TXT / MD into text chunks
├── vector_store.py      # FAISS-based embedding store (sentence-transformers)
├── prompts.py           # Prompt templates + syllabus/research mode classifier
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Module Explanation

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI: sidebar upload/management, model selector, Ollama status, chat interface, source display |
| `rag_pipeline.py` | `RAGPipeline` class — ingest documents, retrieve chunks, call local Ollama model, connectivity checks |
| `document_loader.py` | Per-format loaders (PDF page-by-page, DOCX section-by-section, TXT/MD paragraph-by-paragraph), overlapping chunking |
| `vector_store.py` | `VectorStore` wrapping FAISS IndexFlatL2; embeddings via `all-MiniLM-L6-v2`; supports add / search / remove / clear |
| `prompts.py` | Query classifier (keyword heuristic), context block builder, syllabus prompt, structured research guidance prompt |
| `requirements.txt` | All pip dependencies |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB+ |
| Python | 3.9+ | 3.11+ |
| OS | Windows / macOS / Linux | Any |
| GPU | Not required | NVIDIA (CUDA) or Apple Silicon (Metal) — much faster inference |

---

## Setup Guide

### Step 1 — Install Ollama

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download the installer from [ollama.com](https://ollama.com) and run it.

Verify:
```bash
ollama --version
```

---

### Step 2 — Pull a Language Model

```bash
# 8 GB RAM — fast, recommended starting point
ollama pull llama3.2

# 16 GB RAM — better reasoning and research guidance
ollama pull llama3.1

# 8 GB RAM — strong instruction following
ollama pull mistral

# 8 GB RAM — very lightweight
ollama pull phi3

# 32 GB RAM — highest quality (slow on CPU)
ollama pull llama3.3
```

Test it:
```bash
ollama run llama3.2 "Explain retrieval-augmented generation in two sentences."
```

---

### Step 3 — Download the Project

```bash
unzip academic_assistant.zip
cd academic_assistant
```

---

### Step 4 — Create a Virtual Environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

### Step 5 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB). Subsequent runs load from cache instantly.

---

### Step 6 — Start Ollama

Ollama often runs as a background service automatically. If not:

```bash
ollama serve
```

Keep this terminal open.

---

### Step 7 — Run the App

Open a second terminal:

```bash
streamlit run app.py
```

Visit **http://localhost:8501** in your browser. 

---

## How to Use

1. Check the **Ollama Status** indicator in the sidebar — it must show ● Connected.
2. **Select your model** from the dropdown (only pulled models are available locally).
3. Adjust the **Temperature** slider — lower is more factual, higher is more creative.
4. **Upload documents** — PDF, DOCX, TXT, or MD (syllabus, notes, textbook chapters).
5. Wait for the green "indexed" confirmation.
6. **Type a question** and press **Send ➤**.
7. The app auto-detects your intent:
   -  **Syllabus Mode** — answers only from your uploaded documents, with source citations.
   -  **Research Mode** — structured guidance: overview, papers, project ideas, scope, keywords.
8. Expand **Sources** under any answer to inspect the exact document chunks used.
9. Use sidebar buttons to remove individual documents or clear the session.

---

## Sample Test Queries

### Syllabus Mode
*Upload a course document first.*

| Query | Expected Behaviour |
|-------|--------------------|
| `What topics are covered in Week 3?` | Lists Week 3 content from the syllabus |
| `What is the grading breakdown?` | Returns marks distribution with source citation |
| `Explain the assignment submission policy.` | Extracts policy text from the document |
| `What are the course prerequisites?` | Lists prerequisites as stated in the document |
| `When is the final exam?` | Returns date/details from the uploaded file |
| `Summarise the learning objectives.` | Condenses all objectives found in the document |

### Research Mode
*Works with or without uploaded documents.*

| Query | Expected Behaviour |
|-------|--------------------|
| `Find related papers on transformer architectures` | Lists papers + search keywords |
| `Give me project ideas on federated learning` | 3–5 concrete implementable projects |
| `What is the scope of reinforcement learning in robotics?` | Scope, applications, open problems |
| `Suggest keywords for searching about graph neural networks` | 8–10 database-ready keywords |
| `What are future research directions in explainable AI?` | Open problems and research gaps |
| `What research papers relate to attention mechanisms?` | Papers from docs + known references |

---

## Architecture Overview

```
User uploads file
       │
       ▼
document_loader.py  ── parses text ──► chunks { text, source, page }
       │
       ▼
vector_store.py     ── embeds with all-MiniLM-L6-v2 ──► FAISS index
       │
       │   User asks question
       ▼
vector_store.search() ──► top-k relevant chunks
       │
       ▼
prompts.py  ── classifies mode (syllabus / research), builds grounded prompt
       │
       ▼
rag_pipeline.py ── calls Ollama (local LLM, http://localhost:11434)
       │
       ▼
app.py ── renders answer + source cards in Streamlit chat UI
```

Everything runs on your machine. **No data leaves your computer.**

---

## Model Recommendations

| Model | RAM | Speed | Best For |
|-------|-----|-------|----------|
| `llama3.2` | 8 GB | Fast | Everyday Q&A, syllabus questions |
| `llama3.1` | 16 GB | Medium | Research guidance, detailed answers |
| `mistral` | 16 GB | Medium | Strong instruction following |
| `phi3` | 8 GB | Fast | Lightweight, quick responses |
| `llama3.2:1b` | 4 GB | Very fast | Low-RAM machines |
| `llama3.3` | 32 GB | Slow on CPU | Highest quality responses |

Switch models from the sidebar dropdown any time — no restart needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Sidebar shows ● Not running | Run `ollama serve` in a terminal and refresh the page |
| `model not found` error | Pull the model: `ollama pull llama3.2` |
| Slow responses | Normal on CPU (5–30 s). Use a GPU or try `llama3.2:1b` for speed |
| Out of memory / crash | Use a smaller model: `ollama pull llama3.2:1b` |
| PDF text is empty | PDF is image-based (scanned). Convert with OCR before uploading |
| `ModuleNotFoundError` | Activate venv and re-run `pip install -r requirements.txt` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Embedding slow on first run | `all-MiniLM-L6-v2` downloads once (~80 MB); cached after that |

---

## Privacy

- All document parsing, embedding, and LLM inference happen **on your machine**.
- Nothing is sent to the internet during normal use.
- No data persists between sessions — closing the tab resets everything.

---

## License

MIT — free for academic and personal use.
