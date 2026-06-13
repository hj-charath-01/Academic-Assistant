"""
prompts.py
──────────
All prompt templates used by the RAG pipeline.

Two query modes:
  1. SYLLABUS  – answer strictly from uploaded course documents.
  2. RESEARCH  – structured academic guidance (papers, ideas, scope, keywords).

Mode is detected via keyword matching — no extra LLM call required.
"""

from typing import List, Dict, Any


# ── Mode detection ─────────────────────────────────────────────────────────────

RESEARCH_KEYWORDS = {
    "paper", "papers", "research", "publication", "journal", "conference",
    "study", "studies", "project", "idea", "ideas", "future work",
    "scope", "application", "applications", "keyword", "keywords",
    "literature", "references", "related work", "survey", "thesis",
    "dissertation", "proposal", "contribution", "methodology", "approach",
    "algorithm", "implementation", "experiment", "dataset", "benchmark",
    "state of the art", "open problem", "gap", "limitation", "direction",
    "trend", "overview", "review", "find papers", "suggest papers",
}


def classify_query(query: str) -> str:
    """
    Return "research" if the query is a research-guidance request,
    otherwise return "syllabus".
    """
    lower = query.lower()
    for kw in RESEARCH_KEYWORDS:
        if kw in lower:
            return "research"
    return "syllabus"


# ── Context builder ────────────────────────────────────────────────────────────

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Args:
        chunks : list of chunk dicts from VectorStore.search()

    Returns:
        Multi-line string with one numbered section per chunk.
    """
    if not chunks:
        return "(No relevant context retrieved from uploaded documents.)"

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page   = chunk.get("page", "?")
        text   = chunk.get("text", "").strip()
        lines.append(
            f"[{i}] Source: {source} | Page/Section: {page}\n{text}"
        )
    return "\n\n".join(lines)


# ── Syllabus prompt ────────────────────────────────────────────────────────────

SYLLABUS_SYSTEM = """You are an academic assistant for a university course management system.
Your ONLY knowledge source is the course material provided in the CONTEXT block.

Rules you must follow:
- Answer ONLY using information found in the CONTEXT.
- If the answer is not in the CONTEXT, say exactly:
  "I could not find this information in the uploaded course documents."
- Never fabricate, guess, or use knowledge outside the CONTEXT.
- Be concise (3-6 sentences unless a longer answer is clearly necessary).
- Cite the source document name and page/section for every key claim, e.g. (Source: lecture1.pdf, Page 3).
- Use plain, clear language suitable for a university student.
- Do not repeat the question back to the student."""


def syllabus_prompt(query: str, context: str) -> str:
    return f"""CONTEXT (from uploaded course documents):
{context}

STUDENT QUESTION:
{query}

Answer strictly from the context above. Cite sources inline."""


# ── Research guidance prompt ───────────────────────────────────────────────────

RESEARCH_SYSTEM = """You are an expert academic research advisor helping university students explore topics deeply.
You receive:
  1. CONTEXT – text excerpts from the student's uploaded course documents (may be empty).
  2. TOPIC   – the student's research question or topic of interest.

Your response MUST use exactly these section headings:

## Overview
2-4 sentences explaining the topic clearly.

## Related Research Papers / References
List 4-6 relevant academic papers or textbooks.
- For papers found in the uploaded CONTEXT, cite them as: Author(s) (if known), Title, [Source: filename, Page X].
- For well-known external papers, include: Author(s), "Title", Venue/Journal, Year.
- If you are not certain a paper exists, write: "Search Google Scholar for: <specific query string>" instead.
- Never invent paper titles, authors, or venues.

## Project Ideas
3-5 concrete project ideas a student could implement or investigate.
Each idea: 1-2 sentences describing what to build or study and why it is interesting.

## Scope & Applications
4-6 bullet points covering practical and theoretical applications of this topic.

## Future Work / Open Problems
3-4 unsolved problems or open research questions in this area.

## Useful Keywords for Literature Search
8-10 specific search terms for Google Scholar, IEEE Xplore, ACM Digital Library, or arXiv.

Important rules:
- Ground every claim in either the CONTEXT or well-established academic knowledge.
- Never fabricate references, statistics, or results.
- Keep language accessible to undergraduate and postgraduate students."""


def research_prompt(query: str, context: str) -> str:
    return f"""CONTEXT (from uploaded course documents — may be empty):
{context}

TOPIC / RESEARCH QUESTION:
{query}

Provide structured research guidance using the required section headings."""


# ── Dispatch ───────────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Classify the query and return the appropriate system prompt + user message.

    Returns:
        dict with keys: "mode", "system", "user"
    """
    mode    = classify_query(query)
    context = build_context_block(chunks)

    if mode == "research":
        return {
            "mode":   "research",
            "system": RESEARCH_SYSTEM,
            "user":   research_prompt(query, context),
        }
    return {
        "mode":   "syllabus",
        "system": SYLLABUS_SYSTEM,
        "user":   syllabus_prompt(query, context),
    }