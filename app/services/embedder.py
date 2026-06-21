"""
app/services/embedder.py

Phase 3: Deep AI Embedding Space

Responsibilities:
  - Singleton loader for the sentence-transformer model (CPU-only, thread-safe).
  - Sliding-window text chunker to handle long resumes without truncation.
  - FAISS flat-inner-product index for dense vector storage.
  - Async-safe encode / index / search / persist / reload interface.

Model: all-MiniLM-L6-v2  (384-dim, ~22 MB, fully offline after first download)
Index: IndexFlatIP  (exact cosine similarity via L2-normalised vectors)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("ir_data_matching_engine.embedder")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME: str = "model/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384                  # fixed for all-MiniLM-L6-v2

CHUNK_SIZE: int = 500                     # characters per chunk
CHUNK_OVERLAP: int = 100                  # sliding-window overlap in characters

# Default disk paths (relative to project root)
DEFAULT_INDEX_PATH: Path = Path("data") / "faiss.index"
DEFAULT_META_PATH: Path = Path("data") / "index_meta.npy"


# ---------------------------------------------------------------------------
# Thread-safe model singleton
# ---------------------------------------------------------------------------

class _ModelSingleton:
    """
    Ensures SentenceTransformer is initialised exactly once across threads.
    The heavy model download / load happens on first access only.
    """

    _instance: SentenceTransformer | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:          # double-checked locking
                    logger.info("Loading sentence-transformer model '%s' …", MODEL_NAME)
                    cls._instance = SentenceTransformer(MODEL_NAME)
                    logger.info("Model loaded — embedding dim: %d", EMBEDDING_DIM)
        return cls._instance


# ---------------------------------------------------------------------------
# Text chunker
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split *text* into overlapping character-level windows.

    Parameters
    ----------
    text      : normalised plain-text string (already lowercased by parser).
    chunk_size: maximum characters per chunk.
    overlap   : how many trailing characters of the previous chunk are
                repeated at the start of the next (sliding window).

    Returns
    -------
    List[str] — at least one chunk; empty string yields [''].
    """
    if not text:
        return [""]

    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += stride

    logger.debug("Chunked text (%d chars) → %d chunks", len(text), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _encode_chunks_sync(chunks: List[str]) -> np.ndarray:
    """
    Synchronous encoding — always runs in a thread-pool executor.

    Returns
    -------
    np.ndarray  shape (n_chunks, EMBEDDING_DIM), dtype float32, L2-normalised.
    """
    model = _ModelSingleton.get()
    vectors: np.ndarray = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,      # cosine sim ≡ dot product for unit vecs
    ).astype(np.float32)
    return vectors


async def encode_chunks(chunks: List[str]) -> np.ndarray:
    """Async wrapper — offloads CPU-bound encoding to a thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _encode_chunks_sync, chunks)


async def embed_text(text: str) -> np.ndarray:
    """
    Convenience function: chunk *text* then encode all chunks.

    Returns
    -------
    np.ndarray  shape (n_chunks, 384), float32, L2-normalised.
    """
    chunks = chunk_text(text)
    return await encode_chunks(chunks)


def mean_pool_embeddings(vectors: np.ndarray) -> np.ndarray:
    """
    Collapse chunk-level vectors to a single document vector via mean pooling,
    then re-normalise for consistent cosine comparisons.

    Parameters
    ----------
    vectors : shape (n_chunks, 384)

    Returns
    -------
    np.ndarray  shape (384,), float32
    """
    mean_vec = vectors.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    return mean_vec


# ---------------------------------------------------------------------------
# FAISS index manager
# ---------------------------------------------------------------------------

class FAISSIndexManager:
    """
    Thread-safe, async-compatible manager for a FAISS flat inner-product index.

    Stores one document vector per candidate (mean-pooled across chunks).
    Supports add, search, save-to-disk, and load-from-disk operations.

    Usage
    -----
    manager = FAISSIndexManager()
    manager.load_or_create()

    # index a candidate
    await manager.add_candidate("cand_001", resume_text)

    # search
    results = await manager.search(jd_text, top_k=10)
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> None:
        self.index_path = index_path
        self.meta_path = meta_path

        self._lock = threading.Lock()
        self._index: faiss.Index | None = None   # IndexFlatIP at runtime; base type for type-checker
        # Parallel list to the FAISS index rows — maps row → candidate_id
        self._candidate_ids: List[str] = []

    # ------------------------------------------------------------------ init

    def load_or_create(self) -> None:
        """
        Load an existing index from disk or create a fresh empty one.
        Must be called once during application startup (e.g. lifespan hook).
        """
        with self._lock:
            if self.index_path.exists() and self.meta_path.exists():
                logger.info("Loading existing FAISS index from '%s' …", self.index_path)
                self._index = faiss.read_index(str(self.index_path))
                assert self._index is not None, "faiss.read_index returned None"
                self._candidate_ids = list(np.load(str(self.meta_path), allow_pickle=True))
                logger.info(
                    "FAISS index loaded — %d vectors, dim=%d",
                    self._index.ntotal,
                    EMBEDDING_DIM,
                )
            else:
                logger.info("Creating new FAISS IndexFlatIP (dim=%d) …", EMBEDDING_DIM)
                self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
                self._candidate_ids = []

    # --------------------------------------------------------------- helpers

    def _assert_ready(self) -> None:
        if self._index is None:
            raise RuntimeError(
                "FAISSIndexManager.load_or_create() must be called before use."
            )

    # ------------------------------------------------------------------- add

    def _add_sync(self, candidate_id: str, doc_vector: np.ndarray) -> None:
        """Synchronous add — must run inside lock."""
        with self._lock:
            self._assert_ready()
            # Reshape to (1, dim) as required by FAISS
            vec = doc_vector.reshape(1, -1).astype(np.float32)
            self._index.add(vec)  # type: ignore[union-attr]
            self._candidate_ids.append(candidate_id)
            logger.debug(
                "Indexed candidate '%s' — total vectors: %d",
                candidate_id,
                self._index.ntotal,  # type: ignore[union-attr]
            )

    async def add_candidate(self, candidate_id: str, resume_text: str) -> None:
        """
        Embed *resume_text* and add its mean-pooled vector to the FAISS index.

        Parameters
        ----------
        candidate_id : stable unique identifier (e.g. UUID or filename stem)
        resume_text  : normalised text from the parser layer
        """
        self._assert_ready()
        chunk_vectors = await embed_text(resume_text)
        doc_vector = mean_pool_embeddings(chunk_vectors)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._add_sync, candidate_id, doc_vector
        )

    # ----------------------------------------------------------------- search

    def _search_sync(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        """Synchronous FAISS search — must run inside lock."""
        with self._lock:
            self._assert_ready()
            n_indexed = self._index.ntotal  # type: ignore[union-attr]
            if n_indexed == 0:
                return []

            k = min(top_k, n_indexed)
            q = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self._index.search(q, k)  # type: ignore[union-attr]

            results: List[Tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:          # FAISS sentinel for "no result"
                    continue
                results.append((self._candidate_ids[idx], float(score)))
            return results

    async def search(
        self, query_text: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Embed *query_text* (job description) and return the top-K candidates
        ranked by cosine similarity (inner product of L2-normalised vectors).

        Returns
        -------
        List of (candidate_id, similarity_score) tuples, highest score first.
        """
        self._assert_ready()
        chunk_vectors = await embed_text(query_text)
        query_vector = mean_pool_embeddings(chunk_vectors)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._search_sync, query_vector, top_k
        )

    # ------------------------------------------------------------------ save

    def _save_sync(self) -> None:
        with self._lock:
            self._assert_ready()
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))  # type: ignore[arg-type]
            np.save(str(self.meta_path), np.array(self._candidate_ids, dtype=object))
            logger.info(
                "FAISS index persisted — %d vectors → '%s'",
                self._index.ntotal,  # type: ignore[union-attr]
                self.index_path,
            )

    async def save(self) -> None:
        """Persist the index and metadata to disk asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_sync)

    # ---------------------------------------------------------------- helpers

    @property
    def total_candidates(self) -> int:
        """Number of documents currently indexed."""
        return len(self._candidate_ids)

    def is_ready(self) -> bool:
        """Returns True if the index is initialised and has at least one vector."""
        return self._index is not None and self._index.ntotal > 0  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Module-level singleton (imported by main.py lifespan)
# ---------------------------------------------------------------------------

index_manager = FAISSIndexManager()
