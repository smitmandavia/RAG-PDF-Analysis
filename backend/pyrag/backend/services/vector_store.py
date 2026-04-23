"""
PyRAG — Vector Store Service
ChromaDB wrapper for storing and querying document embeddings.
"""

import re
import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, TOP_K, SIMILARITY_THRESHOLD

# Singleton client
_client = None
_collection = None

COLLECTION_NAME = "pyrag_documents"
SEMANTIC_WEIGHT = 0.72
KEYWORD_WEIGHT = 0.28
MIN_KEYWORD_SCORE = 0.12

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "give", "had", "has",
    "have", "he", "her", "here", "his", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "our", "please", "question", "recent",
    "she", "so", "that", "the", "their", "there", "this", "to", "user",
    "was", "we", "what", "when", "where", "which", "who", "why", "with",
    "work", "worked", "working", "you", "your"
}


def get_collection():
    """Get or create the ChromaDB collection."""
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"  [ok] ChromaDB collection ready ({_collection.count()} vectors)")

    return _collection


def add_chunks(doc_id: str, doc_name: str, chunks: list, embeddings: list[list[float]]):
    """
    Store document chunks with their embeddings.
    Each chunk gets a unique ID and metadata for retrieval.
    """
    collection = get_collection()

    ids = [f"{doc_id}_chunk_{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "document_id": doc_id,
            "document_name": doc_name,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "section_title": c.section_title
        }
        for c in chunks
    ]

    # ChromaDB has a batch limit, so we chunk the inserts
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9+#.\-]*", (text or "").lower())
    return [
        token.strip(".-")
        for token in tokens
        if len(token.strip(".-")) > 1 and token.strip(".-") not in STOPWORDS
    ]


def _keyword_score(query_text: str | None, chunk_text: str) -> float:
    """Score exact lexical overlap so names, dates, and technical terms survive reranking."""
    query_tokens = list(dict.fromkeys(_tokenize(query_text or "")))
    if not query_tokens:
        return 0.0

    chunk_lower = (chunk_text or "").lower()
    chunk_tokens = set(_tokenize(chunk_text or ""))

    matched_tokens = [token for token in query_tokens if token in chunk_tokens]
    coverage_score = len(matched_tokens) / len(query_tokens)

    frequency_hits = sum(min(chunk_lower.count(token), 3) for token in matched_tokens if len(token) > 2)
    frequency_score = min(frequency_hits / max(len(query_tokens) * 2, 1), 1.0)

    phrases = []
    for i in range(len(query_tokens) - 1):
        phrases.append(f"{query_tokens[i]} {query_tokens[i + 1]}")
    for i in range(len(query_tokens) - 2):
        phrases.append(f"{query_tokens[i]} {query_tokens[i + 1]} {query_tokens[i + 2]}")

    phrase_count = sum(1 for phrase in phrases if phrase in chunk_lower)
    phrase_score = min(phrase_count / max(len(phrases), 1), 1.0)

    return round((coverage_score * 0.65) + (frequency_score * 0.2) + (phrase_score * 0.15), 4)


def _hybrid_score(semantic_similarity: float, keyword_score: float) -> float:
    semantic = max(0.0, min(semantic_similarity, 1.0))
    keyword = max(0.0, min(keyword_score, 1.0))
    return round((semantic * SEMANTIC_WEIGHT) + (keyword * KEYWORD_WEIGHT), 4)


def query_chunks(query_embedding: list[float], top_k: int = TOP_K,
                 document_ids: list[str] = None,
                 query_text: str | None = None) -> list[dict]:
    """
    Find relevant chunks with semantic search, then rerank using keyword overlap.
    Returns list of dicts with text, metadata, and relevance score.
    """
    collection = get_collection()
    k = top_k or TOP_K

    if collection.count() == 0:
        return []

    # Build where filter for specific documents
    where_filter = None
    if document_ids:
        if len(document_ids) == 1:
            where_filter = {"document_id": document_ids[0]}
        else:
            where_filter = {"document_id": {"$in": document_ids}}

    candidate_count = min(max(k * 6, 20), collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Process results
    chunks = []
    if results and results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            # ChromaDB returns cosine distance; convert to similarity
            distance = results['distances'][0][i]
            semantic_similarity = 1 - distance  # cosine distance to similarity
            keyword_score = _keyword_score(query_text, results['documents'][0][i])
            relevance_score = _hybrid_score(semantic_similarity, keyword_score)

            if semantic_similarity < SIMILARITY_THRESHOLD and keyword_score < MIN_KEYWORD_SCORE:
                continue

            chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': relevance_score,
                'semantic_similarity': round(semantic_similarity, 4),
                'keyword_score': keyword_score
            })

    # Sort by hybrid relevance (highest first)
    chunks.sort(key=lambda x: x['similarity'], reverse=True)
    return chunks[:k]


def delete_document_chunks(doc_id: str):
    """Remove all chunks for a document."""
    collection = get_collection()

    # Get all chunk IDs for this document
    try:
        results = collection.get(
            where={"document_id": doc_id},
            include=[]
        )
        if results['ids']:
            collection.delete(ids=results['ids'])
    except Exception:
        pass  # Document might not have chunks yet


def get_total_chunks() -> int:
    """Get total number of chunks in the vector store."""
    collection = get_collection()
    return collection.count()


def get_document_chunks(doc_id: str) -> list[dict]:
    """Get all chunks for a document (for preview)."""
    collection = get_collection()

    try:
        results = collection.get(
            where={"document_id": doc_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        if results['ids']:
            for i in range(len(results['ids'])):
                chunks.append({
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })

            # Sort by chunk_index
            chunks.sort(key=lambda x: x['metadata']['chunk_index'])

        return chunks
    except Exception:
        return []
