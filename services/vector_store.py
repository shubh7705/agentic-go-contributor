"""
services/vector_store.py — FAISS-backed vector store with hybrid retrieval.

Pipeline:
  Source files → chunking → SentenceTransformer embeddings → FAISS index

Hybrid score:
  score = 0.6 * embedding_score + 0.4 * rg_score

Implements:
  - VectorStore.build()   — index a repository
  - VectorStore.search()  — query with hybrid retrieval
  - persistence (save / load)
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from config.settings import settings
from services.embeddings import encode_texts, encode_single, get_embedding_dim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A text chunk extracted from a source file."""

    file_path: str          # Relative path within the repo
    content: str            # Raw text content of the chunk
    start_line: int         # Starting line number (1-indexed)
    end_line: int           # Ending line number (inclusive)
    chunk_index: int        # Sequential index within the file
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single retrieval result combining embedding and ripgrep scores."""

    chunk: Chunk
    embedding_score: float
    rg_score: float
    hybrid_score: float

    @property
    def file_path(self) -> str:
        return self.chunk.file_path


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_LINES = 50
_CHUNK_OVERLAP_LINES = 10


def chunk_file(
    file_path: str,
    content: str,
    chunk_size: int = _DEFAULT_CHUNK_LINES,
    overlap: int = _CHUNK_OVERLAP_LINES,
) -> list[Chunk]:
    """
    Split a file into overlapping line-based chunks.

    Args:
        file_path:  Relative path to the file (used as identifier).
        content:    Full text content of the file.
        chunk_size: Number of lines per chunk.
        overlap:    Number of lines shared between adjacent chunks.

    Returns:
        List of Chunk objects.
    """
    lines = content.splitlines()
    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    idx = 0

    for start in range(0, len(lines), step):
        end = min(start + chunk_size, len(lines))
        chunk_text = "\n".join(lines[start:end])
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    file_path=file_path,
                    content=chunk_text,
                    start_line=start + 1,
                    end_line=end,
                    chunk_index=idx,
                )
            )
            idx += 1
        if end >= len(lines):
            break

    return chunks


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    FAISS-backed vector store for repository source chunks.

    Supports hybrid retrieval combining semantic embeddings with
    optional ripgrep keyword scores.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._dim: int = get_embedding_dim()

    # ── Build ────────────────────────────────────────────────────────────────

    def build(
        self,
        file_contents: dict[str, str],
        *,
        chunk_size: int = _DEFAULT_CHUNK_LINES,
        overlap: int = _CHUNK_OVERLAP_LINES,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> None:
        """
        Index all provided file contents.

        Args:
            file_contents: Mapping of {relative_path: file_text}.
            chunk_size:    Lines per chunk.
            overlap:       Overlap between adjacent chunks.
            batch_size:    Encoding batch size.
            show_progress: Show tqdm progress bar during encoding.
        """
        logger.info("Building vector index from %d files …", len(file_contents))

        all_chunks: list[Chunk] = []
        for path, text in file_contents.items():
            all_chunks.extend(chunk_file(path, text, chunk_size, overlap))

        logger.info("Created %d chunks from %d files", len(all_chunks), len(file_contents))

        texts = [c.content for c in all_chunks]
        embeddings = encode_texts(texts, batch_size=batch_size, show_progress=show_progress)

        # Build an inner-product FAISS index (cosine similarity, since normalised)
        index = faiss.IndexFlatIP(self._dim)
        index.add(embeddings)  # type: ignore[arg-type]

        self._chunks = all_chunks
        self._index = index
        logger.info("FAISS index built (%d vectors, dim=%d)", index.ntotal, self._dim)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        rg_scores: dict[str, float] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Uses hybrid scoring:
            score = 0.6 * embedding_score + 0.4 * rg_score

        Args:
            query:     The natural-language query (typically the issue body).
            k:         Number of results to return (default: settings.retrieval_top_k).
            rg_scores: Optional mapping of {file_path: rg_score ∈ [0, 1]} from
                       ripgrep. Files not listed receive rg_score=0.

        Returns:
            List of RetrievalResult sorted by descending hybrid_score.
        """
        if self._index is None or not self._chunks:
            raise RuntimeError("VectorStore has not been built. Call build() first.")

        k = k or settings.retrieval_top_k
        rg_scores = rg_scores or {}

        # Retrieve more candidates so we can re-rank after hybrid scoring
        faiss_k = min(len(self._chunks), max(k * 5, 50))
        query_vec = encode_single(query).reshape(1, -1)
        distances, indices = self._index.search(query_vec, faiss_k)  # type: ignore[arg-type]

        results: list[RetrievalResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            emb_score = float(dist)  # already cosine (inner product on normalised vecs)
            rg_score = rg_scores.get(chunk.file_path, 0.0)
            hybrid = (
                settings.embedding_weight * emb_score
                + settings.rg_weight * rg_score
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    embedding_score=emb_score,
                    rg_score=rg_score,
                    hybrid_score=hybrid,
                )
            )

        # Sort by hybrid score, deduplicate by keeping best score per chunk
        seen: dict[tuple[str, int], RetrievalResult] = {}
        for r in sorted(results, key=lambda x: x.hybrid_score, reverse=True):
            key = (r.chunk.file_path, r.chunk.chunk_index)
            if key not in seen:
                seen[key] = r

        ranked = sorted(seen.values(), key=lambda x: x.hybrid_score, reverse=True)
        return ranked[:k]

    def top_files(
        self,
        query: str,
        *,
        k: int | None = None,
        rg_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k unique file paths with their best hybrid score.

        Returns:
            List of (file_path, score) tuples in descending order.
        """
        results = self.search(query, k=k, rg_scores=rg_scores)
        file_scores: dict[str, float] = {}
        for r in results:
            if r.file_path not in file_scores or r.hybrid_score > file_scores[r.file_path]:
                file_scores[r.file_path] = r.hybrid_score
        return sorted(file_scores.items(), key=lambda x: x[1], reverse=True)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Persist the index and chunk metadata to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as fh:
            pickle.dump(self._chunks, fh)
        logger.info("Vector store saved to %s", directory)

    def load(self, directory: Path) -> None:
        """Load a previously persisted index from disk."""
        index_path = directory / "index.faiss"
        chunks_path = directory / "chunks.pkl"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"No vector store found at {directory}")
        self._index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as fh:
            self._chunks = pickle.load(fh)
        logger.info(
            "Vector store loaded from %s (%d chunks)", directory, len(self._chunks)
        )

    # ── Utilities ────────────────────────────────────────────────────────────

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_built(self) -> bool:
        return self._index is not None and len(self._chunks) > 0
