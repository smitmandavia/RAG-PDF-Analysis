"""
PyRAG — RAG Engine
Orchestrates the full query pipeline:
Question → Embed → Retrieve → Generate → Response
"""

import time
from services.embedder import embed_query
from services.vector_store import query_chunks
from services.llm_provider import generate_answer
from db.database import save_chat, get_chat_history
from config import TOP_K


def _clip(text: str, max_chars: int) -> str:
    """Trim old turns so retrieval queries stay focused."""
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _get_recent_history(use_history: bool, history_limit: int | None) -> list[dict]:
    if not use_history:
        return []

    limit = history_limit if history_limit is not None else 4
    limit = max(0, min(limit, 10))
    if limit == 0:
        return []

    # Database returns newest first; prompts read more naturally oldest first.
    return list(reversed(get_chat_history(limit)))


def _build_contextual_query(question: str, chat_history: list[dict]) -> str:
    """
    Build a richer retrieval query so short follow-ups can inherit context.
    The LLM prompt still treats only retrieved document chunks as evidence.
    """
    if not chat_history:
        return question

    lines = ["Recent conversation for follow-up resolution:"]
    for turn in chat_history:
        lines.append(f"User: {_clip(turn.get('question', ''), 220)}")
        lines.append(f"Assistant: {_clip(turn.get('answer', ''), 320)}")
    lines.append(f"Current question: {question}")
    return "\n".join(lines)


async def ask(
    question: str,
    top_k: int = None,
    document_ids: list[str] = None,
    use_history: bool = True,
    history_limit: int = 4
) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Retrieve relevant chunks
    3. Generate answer with LLM
    4. Return answer + sources
    """
    start_time = time.time()
    k = top_k or TOP_K
    chat_history = _get_recent_history(use_history, history_limit)
    retrieval_query = _build_contextual_query(question, chat_history)

    # Step 1: Embed the question
    query_vector = embed_query(retrieval_query)

    # Step 2: Retrieve relevant chunks
    relevant_chunks = query_chunks(
        query_vector,
        top_k=k,
        document_ids=document_ids,
        query_text=retrieval_query
    )

    if not relevant_chunks:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents to answer this question. Please try rephrasing, or make sure relevant documents have been uploaded.",
            "sources": [],
            "question": question,
            "processing_time": round(time.time() - start_time, 2)
        }

    # Step 3: Generate answer
    answer = await generate_answer(question, relevant_chunks, chat_history=chat_history)

    # Step 4: Build sources list
    sources = []
    for chunk in relevant_chunks:
        sources.append({
            "document_name": chunk['metadata']['document_name'],
            "document_id": chunk['metadata']['document_id'],
            "page_number": chunk['metadata']['page_number'],
            "section_title": chunk['metadata'].get('section_title'),
            "chunk_text": chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'],
            "similarity_score": chunk['similarity']
        })

    processing_time = round(time.time() - start_time, 2)

    # Save to history
    save_chat(question, answer, sources, processing_time)

    return {
        "answer": answer,
        "sources": sources,
        "question": question,
        "processing_time": processing_time
    }
