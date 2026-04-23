"""
PyRAG — Embedding Service
Uses sentence-transformers for high-quality semantic embeddings.
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model = None


def get_embedder():
    global _model
    if _model is None:
        print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("  [ok] Model loaded")
    return _model


def embed_texts(texts):
    model = get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query):
    model = get_embedder()
    embedding = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
    return embedding[0].tolist()
