"""
tests/test_services.py — Unit tests for the services layer.

These tests use mocking to avoid requiring actual API keys or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Embeddings tests (mocked)
# ---------------------------------------------------------------------------

class TestEmbeddings:
    @patch("services.embeddings._model", None)
    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_single_shape(self, mock_st_class):
        """encode_single should return a 1-D float32 array."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        import services.embeddings as emb
        emb._model = mock_model

        vec = emb.encode_single("hello world")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_cosine_similarity_identical(self):
        from services.embeddings import cosine_similarity

        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        from services.embeddings import cosine_similarity

        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-5)

    def test_top_k_similar(self):
        from services.embeddings import top_k_similar

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = np.array([
            [1.0, 0.0, 0.0],   # identical → highest
            [0.5, 0.5, 0.0],   # partial
            [0.0, 0.0, 1.0],   # orthogonal
        ], dtype=np.float32)

        results = top_k_similar(query, candidates, k=2)
        assert len(results) == 2
        assert results[0][0] == 0  # index 0 should be top


# ---------------------------------------------------------------------------
# Vector store tests (mocked)
# ---------------------------------------------------------------------------

class TestVectorStore:
    @pytest.fixture()
    def mock_embeddings(self, monkeypatch):
        """Return deterministic fake embeddings."""
        dim = 16

        def fake_encode_texts(texts, **kwargs):
            rng = np.random.default_rng(42)
            arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / norms

        def fake_encode_single(text, **kwargs):
            rng = np.random.default_rng(0)
            v = rng.standard_normal(dim).astype(np.float32)
            return v / np.linalg.norm(v)

        def fake_dim():
            return dim

        monkeypatch.setattr("services.vector_store.encode_texts", fake_encode_texts)
        monkeypatch.setattr("services.vector_store.encode_single", fake_encode_single)
        monkeypatch.setattr("services.vector_store.get_embedding_dim", fake_dim)
        return dim

    def test_build_and_search(self, mock_embeddings):
        from services.vector_store import VectorStore

        store = VectorStore()
        store._dim = mock_embeddings

        files = {
            "binding/json.go": "package binding\n\nfunc decodeJSON() {}\n" * 5,
            "render/html.go": "package render\n\nfunc Render() {}\n" * 5,
            "router/engine.go": "package router\n\nfunc New() {}\n" * 5,
        }
        store.build(files, show_progress=False)
        assert store.is_built
        assert store.num_chunks > 0

        results = store.search("JSON decoding", k=2)
        assert len(results) <= 2
        assert all(hasattr(r, "hybrid_score") for r in results)

    def test_top_files(self, mock_embeddings):
        from services.vector_store import VectorStore

        store = VectorStore()
        store._dim = mock_embeddings

        files = {
            "a.go": "package a\n" * 10,
            "b.go": "package b\n" * 10,
        }
        store.build(files, show_progress=False)
        top = store.top_files("any query", k=2)
        assert isinstance(top, list)
        assert len(top) <= 2
        assert all(isinstance(score, float) for _, score in top)

    def test_save_and_load(self, mock_embeddings, tmp_path):
        from services.vector_store import VectorStore

        store = VectorStore()
        store._dim = mock_embeddings
        store.build({"foo.go": "package foo\n" * 10}, show_progress=False)

        store.save(tmp_path / "index")

        store2 = VectorStore()
        store2._dim = mock_embeddings
        store2.load(tmp_path / "index")
        assert store2.is_built
        assert store2.num_chunks == store.num_chunks


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

class TestChunking:
    def test_chunk_file_basic(self):
        from services.vector_store import chunk_file

        content = "\n".join(f"line {i}" for i in range(200))
        chunks = chunk_file("test.go", content, chunk_size=50, overlap=10)
        assert len(chunks) > 1
        assert all(c.file_path == "test.go" for c in chunks)
        assert all(c.content.strip() for c in chunks)

    def test_chunk_file_small_file(self):
        from services.vector_store import chunk_file

        content = "package main\n"
        chunks = chunk_file("small.go", content, chunk_size=50)
        assert len(chunks) == 1
        # splitlines() + join strips the trailing newline — that's expected behaviour
        assert chunks[0].content.strip() == content.strip()

    def test_chunk_indices_sequential(self):
        from services.vector_store import chunk_file

        content = "\n".join(f"line {i}" for i in range(100))
        chunks = chunk_file("seq.go", content)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
