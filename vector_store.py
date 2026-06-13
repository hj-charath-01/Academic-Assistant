"""
vector_store.py
───────────────
Manages embeddings and similarity search using FAISS.

Embedding model : all-MiniLM-L6-v2  (fast, lightweight, fully offline)
Vector store    : FAISS IndexFlatL2  (exact nearest-neighbour search)

Public API:
    VectorStore.add_chunks(chunks)          → embed & index new chunks
    VectorStore.search(query, k)            → return top-k relevant chunks
    VectorStore.remove_by_source(filename)  → delete chunks from one document
    VectorStore.clear()                     → wipe entire store
    VectorStore.is_empty()                  → bool
    VectorStore.document_list()             → list of indexed filenames
    VectorStore.chunk_count()              → int
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional

# Heavy imports are deferred so Streamlit can render before models load
_faiss = None
_SentenceTransformer = None


def _import_deps():
    """Import FAISS and sentence-transformers once, on first use."""
    global _faiss, _SentenceTransformer
    if _faiss is None:
        import faiss as faiss_lib
        _faiss = faiss_lib
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        _SentenceTransformer = ST


# ── Embedding model singleton ──────────────────────────────────────────────────

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[Any] = None


def _get_model():
    """Return (and cache) the sentence-transformer model."""
    global _model
    _import_deps()
    if _model is None:
        _model = _SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings into L2-normalised float32 vectors.

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model   = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # L2-normalise so cosine similarity == dot product
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype("float32")


# ── VectorStore ────────────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory FAISS vector store with per-chunk metadata.

    Attributes:
        _index  : FAISS index (built lazily on first add)
        _chunks : list of chunk dicts mirroring index rows 1-to-1
        _dim    : embedding dimension (set on first add)
    """

    def __init__(self):
        self._index:  Optional[Any]         = None
        self._chunks: List[Dict[str, Any]]  = []
        self._dim:    Optional[int]         = None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _init_index(self, dim: int):
        _import_deps()
        self._dim   = dim
        self._index = _faiss.IndexFlatL2(dim)

    # ── Public ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and index a list of chunk dicts.

        Args:
            chunks : list of {"text": str, "source": str, "page": int}

        Returns:
            Number of vectors added
        """
        if not chunks:
            return 0

        texts   = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        if self._index is None:
            self._init_index(vectors.shape[1])

        self._index.add(vectors)
        self._chunks.extend(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k chunks most relevant to *query*.

        Args:
            query : natural-language question or topic string
            k     : number of results to return

        Returns:
            List of chunk dicts augmented with a "score" key
            (lower L2 distance = more similar). Empty list if store is empty.
        """
        if self.is_empty():
            return []

        q_vec             = embed_texts([query])
        k                 = min(k, len(self._chunks))
        distances, indices = self._index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk          = dict(self._chunks[idx])   # shallow copy
            chunk["score"] = float(dist)
            results.append(chunk)

        return results

    def remove_by_source(self, filename: str) -> int:
        """
        Remove all chunks that came from *filename*.

        FAISS IndexFlatL2 does not support deletion, so we rebuild the index
        from the remaining chunks. Acceptable for typical academic document counts.

        Returns:
            Number of chunks removed
        """
        if self.is_empty():
            return 0

        kept    = [c for c in self._chunks if c["source"] != filename]
        removed = len(self._chunks) - len(kept)

        if removed == 0:
            return 0

        # Rebuild from scratch
        self._index  = None
        self._chunks = []

        if kept:
            self.add_chunks(kept)

        return removed

    def clear(self):
        """Wipe the entire vector store."""
        self._index  = None
        self._chunks = []
        self._dim    = None

    def is_empty(self) -> bool:
        return self._index is None or len(self._chunks) == 0

    def document_list(self) -> List[str]:
        """Return deduplicated list of indexed filenames (insertion order)."""
        return list(dict.fromkeys(c["source"] for c in self._chunks))

    def chunk_count(self) -> int:
        return len(self._chunks)