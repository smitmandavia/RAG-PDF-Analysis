"""
PyRAG — Chat API Routes
"""

from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from services.rag_engine import ask
from db.database import get_chat_history

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Ask a question about uploaded documents."""
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    try:
        result = await ask(
            question=request.question.strip(),
            top_k=request.top_k,
            document_ids=request.document_ids,
            use_history=request.use_history,
            history_limit=request.history_limit
        )

        return ChatResponse(
            answer=result['answer'],
            sources=[
                {
                    "document_name": s['document_name'],
                    "document_id": s['document_id'],
                    "page_number": s['page_number'],
                    "chunk_text": s['chunk_text'],
                    "similarity_score": s['similarity_score']
                }
                for s in result['sources']
            ],
            question=result['question'],
            processing_time=result['processing_time']
        )

    except Exception as e:
        raise HTTPException(500, f"Error processing question: {str(e)}")


@router.get("/history")
async def chat_history(limit: int = 50):
    """Get recent chat history."""
    history = get_chat_history(limit)
    return {"history": history, "total": len(history)}
