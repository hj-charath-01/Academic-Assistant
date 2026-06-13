"""
rag_pipeline.py
───────────────
Orchestrates the full Retrieval-Augmented Generation (RAG) pipeline:

  Upload  →  chunk  →  embed  →  store
  Query   →  retrieve  →  prompt  →  generate (Ollama)  →  return

This module is the single entry-point used by app.py.
All LLM inference runs locally via Ollama — no API key required.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

import ollama

from document_loader import load_document
from vector_store import VectorStore
from prompts import build_prompt


# ── RAGPipeline ────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG system for the Academic Assistant.

    Attributes:
        store      : VectorStore holding all indexed document chunks
        top_k      : number of context chunks retrieved per query
        model      : Ollama model name (e.g. "llama3.2", "mistral")
        max_tokens : maximum tokens the LLM may generate
        temperature: controls answer creativity (lower = more grounded)
    """

    def __init__(
        self,
        top_k: int = 5,
        model: str = "llama3.2",
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ):
        self.store       = VectorStore()
        self.top_k       = top_k
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

    # ── Document ingestion ─────────────────────────────────────────────────────

    def ingest(self, file_bytes: bytes, filename: str) -> Tuple[int, str]:
        """
        Parse a document and add its chunks to the vector store.

        Args:
            file_bytes : raw bytes of the uploaded file
            filename   : original filename

        Returns:
            (num_chunks_added, status_message)
        """
        try:
            chunks = load_document(file_bytes, filename)
            if not chunks:
                return 0, f"⚠️ No extractable text found in **{filename}**."

            added = self.store.add_chunks(chunks)
            return added, (
                f"✅ **{filename}** indexed — "
                f"{added} chunk(s) added to the knowledge base."
            )
        except ValueError as e:
            return 0, f"❌ {e}"
        except Exception as e:
            return 0, f"❌ Unexpected error processing **{filename}**: {e}"

    def remove_document(self, filename: str) -> Tuple[int, str]:
        """
        Remove all chunks belonging to *filename* from the vector store.

        Returns:
            (num_chunks_removed, status_message)
        """
        removed = self.store.remove_by_source(filename)
        if removed == 0:
            return 0, f"⚠️ No chunks found for **{filename}**."
        return removed, f"🗑️ **{filename}** removed ({removed} chunk(s) deleted)."

    def clear_all(self):
        """Wipe the entire vector store."""
        self.store.clear()

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch the most relevant chunks for a query.

        Returns:
            Ranked list of chunk dicts (empty if store is empty).
        """
        return self.store.search(query, k=self.top_k)

    # ── Generation via Ollama ──────────────────────────────────────────────────

    def _call_llm(self, system: str, user_message: str) -> str:
        """
        Call the local Ollama model and return the assistant's text response.

        Ollama must be running locally (ollama serve).
        The model must have been pulled first (ollama pull <model>).

        Args:
            system       : system-prompt string
            user_message : full user-turn string

        Returns:
            Generated text content
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_message},
                ],
                options={
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p":       0.9,
                },
            )

            # ollama >= 0.4.x returns a Pydantic object, not a plain dict.
            # Support both styles defensively.
            if hasattr(response, "message"):
                # Object-style (ollama 0.4+)
                msg = response.message
                return msg.content if hasattr(msg, "content") else str(msg)
            elif isinstance(response, dict):
                # Dict-style (older ollama < 0.4)
                return response["message"]["content"]
            else:
                return str(response)

        except ollama.ResponseError as e:
            return (
                f"❌ **Ollama Error**: {e.error}\n\n"
                f"Make sure the model is pulled:\n```\nollama pull {self.model}\n```"
            )
        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                return (
                    "❌ **Cannot connect to Ollama.**\n\n"
                    "Start it in a terminal:\n```\nollama serve\n```\n"
                    "Then refresh this page."
                )
            return f"❌ **Unexpected Error**: {e}"

    # ── Main query entry-point ─────────────────────────────────────────────────

    def query(
        self,
        question: str,
        chat_history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a user question.

        Args:
            question     : the student's question or topic
            chat_history : prior messages (used for UI display only;
                           the LLM is called statelessly to keep answers
                           strictly grounded in retrieved context)

        Returns:
            dict with keys:
                answer  : str  — the generated response
                mode    : str  — "syllabus" | "research"
                sources : list — chunk dicts used as context
                no_docs : bool — True if no documents are uploaded
        """
        # 1. Retrieve relevant context chunks
        chunks  = self.retrieve(question)
        no_docs = self.store.is_empty()

        # 2. Build the right prompt for the detected mode
        prompt_data = build_prompt(question, chunks)
        mode   = prompt_data["mode"]
        system = prompt_data["system"]
        user   = prompt_data["user"]

        # 3. Generate the answer via Ollama
        answer = self._call_llm(system, user)

        # 4. Return structured result for the UI
        return {
            "answer":  answer,
            "mode":    mode,
            "sources": chunks,
            "no_docs": no_docs,
        }

    # ── Store metadata helpers ─────────────────────────────────────────────────

    def document_list(self) -> List[str]:
        return self.store.document_list()

    def chunk_count(self) -> int:
        return self.store.chunk_count()

    def is_ready(self) -> bool:
        """Return True if at least one document has been indexed."""
        return not self.store.is_empty()

    # ── Ollama connectivity check ──────────────────────────────────────────────

    @staticmethod
    def check_ollama() -> Tuple[bool, str]:
        """
        Ping the local Ollama server.

        Returns:
            (is_running: bool, message: str)
        """
        try:
            models = ollama.list()
            names  = [m["model"] for m in models.get("models", [])]
            if names:
                return True, f"Ollama running. Available models: {', '.join(names)}"
            return True, "Ollama running. No models pulled yet — run: ollama pull llama3.2"
        except Exception:
            return False, "Ollama not reachable. Run: ollama serve"

    @staticmethod
    def list_local_models() -> List[str]:
        """
        Return names of models already pulled to this machine.
        Falls back to a default list if Ollama is not reachable.
        """
        try:
            models = ollama.list()
            names  = [m["model"] for m in models.get("models", [])]
            return names if names else ["llama3.2"]
        except Exception:
            return ["llama3.2", "llama3.1", "mistral", "phi3", "gemma2"]