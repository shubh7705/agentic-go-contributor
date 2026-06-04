"""
services/embeddings.py — Sentence-Transformer embedding service.

Wraps the sentence-transformers library to provide:
- Lazy model loading
- Batch encoding
- Cosine similarity utilities
- Numpy array type stubs
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazily load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded (dim=%d)", get_embedding_dim())
    return _model


def get_embedding_dim() -> int:
    """Return the embedding dimensionality of the current model."""
    return get_model().get_sentence_embedding_dimension()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_texts(
    texts: list[str],
    *,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of texts into embedding vectors.

    Args:
        texts:         Texts to encode.
        batch_size:    Number of texts per encoding batch.
        normalize:     If True, L2-normalise the output vectors (enables cosine
                       similarity via dot product).
        show_progress: Show a tqdm progress bar.

    Returns:
        Float32 numpy array of shape (len(texts), embedding_dim).
    """
    model = get_model()
    logger.debug("Encoding %d texts (batch_size=%d)", len(texts), batch_size)
    embeddings: np.ndarray = model.encode(  # type: ignore[assignment]
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_single(text: str, *, normalize: bool = True) -> np.ndarray:
    """
    Encode a single text string.

    Returns:
        Float32 numpy array of shape (embedding_dim,).
    """
    result = encode_texts([text], normalize=normalize)
    return result[0]


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalised vectors.

    If the vectors were already normalised during encoding (normalize=True)
    this reduces to a simple dot product.

    Args:
        vec_a: 1-D float32 array.
        vec_b: 1-D float32 array.

    Returns:
        Similarity score in [-1.0, 1.0].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def top_k_similar(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    k: int = 10,
) -> list[tuple[int, float]]:
    """
    Return the indices and scores of the top-k most similar candidates.

    Args:
        query_vec:      1-D float32 embedding.
        candidate_vecs: 2-D float32 array (n_candidates, dim).
        k:              Number of results to return.

    Returns:
        List of (index, score) tuples sorted by descending score.
    """
    scores: np.ndarray = candidate_vecs @ query_vec  # dot product (cosine if normalised)
    k = min(k, len(scores))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [(int(idx), float(scores[idx])) for idx in top_indices]
