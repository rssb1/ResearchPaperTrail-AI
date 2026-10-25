"""
embeddings.py
-------------
Wraps the embeddings API call and provides simple in-memory cosine-similarity
retrieval over a list of pdf_processor.Chunk objects. No vector database —
everything lives in memory for the lifetime of the Streamlit session, which
is all an MVP needs.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from openai import OpenAI, APIError, APIConnectionError, AuthenticationError

from pdf_processor import Chunk

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 96  # keep well under API batch limits


class EmbeddingError(Exception):
    """Raised when the embeddings API call fails for a reason the UI should surface."""


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EmbeddingError(
            "OPENAI_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return OpenAI(api_key=api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings, batching requests. Returns one vector per input string."""
    if not texts:
        return []

    client = _get_client()
    all_vectors: List[List[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except AuthenticationError as e:
            raise EmbeddingError(
                "Authentication with the embeddings API failed — check OPENAI_API_KEY."
            ) from e
        except APIConnectionError as e:
            raise EmbeddingError(
                "Could not connect to the embeddings API. Check your network connection."
            ) from e
        except APIError as e:
            raise EmbeddingError(f"Embeddings API returned an error: {e}") from e
        except Exception as e:
            raise EmbeddingError(f"Unexpected error while embedding text: {e}") from e

        # Preserve input order using the index field the API returns.
        ordered = sorted(resp.data, key=lambda d: d.index)
        all_vectors.extend([d.embedding for d in ordered])

    return all_vectors


def embed_chunks(chunks: List[Chunk]) -> None:
    """Embed a list of Chunks in place, setting chunk.embedding for each."""
    if not chunks:
        return
    vectors = embed_texts([c.text for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector


def _cosine_sim_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm


def retrieve_top_chunks(
    query: str,
    chunks: List[Chunk],
    top_k: int = 6,
    min_score: float = 0.0,
) -> List[tuple[Chunk, float]]:
    """
    Embed `query` and return the top_k most similar chunks (chunk, score),
    sorted by descending similarity. Chunks without an embedding are skipped.
    """
    embedded = [c for c in chunks if c.embedding is not None]
    if not embedded:
        return []

    query_vec = np.array(embed_texts([query])[0], dtype=np.float32)
    matrix = np.array([c.embedding for c in embedded], dtype=np.float32)
    scores = _cosine_sim_matrix(query_vec, matrix)

    ranked = sorted(zip(embedded, scores), key=lambda pair: pair[1], reverse=True)
    results = [(c, float(s)) for c, s in ranked if s >= min_score]
    return results[:top_k]
