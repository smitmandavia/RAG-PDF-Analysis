"""
PyRAG — Main Application
Intelligent Document Q&A powered by RAG
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import CORS_ORIGINS, HOST, PORT

# Import routes
from api.routes_documents import router as documents_router
from api.routes_chat import router as chat_router
from api.routes_health import router as health_router

app = FastAPI(
    title="PyRAG",
    description="Intelligent Document Q&A — Upload PDFs, ask questions, get cited answers.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(health_router)


@app.on_event("startup")
async def startup():
    print()
    print("  +-------------------------------------------+")
    print("  |   PyRAG - Document Q&A System             |")
    print("  |   Upload PDFs. Ask questions. Get answers.|")
    print("  +-------------------------------------------+")
    print()

    # Pre-load embedding service
    from services.embedder import get_embedder
    get_embedder()

    # Init vector store
    from services.vector_store import get_collection
    get_collection()

    print()
    print(f"  Server ready at http://localhost:{PORT}")
    print(f"  API docs at http://localhost:{PORT}/docs")
    print()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
