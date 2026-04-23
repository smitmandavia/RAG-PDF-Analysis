# 📄 PyRAG — Intelligent Document Q&A System

Upload PDFs. Ask questions. Get cited answers.

A full-stack Retrieval-Augmented Generation (RAG) pipeline built from scratch with FastAPI + React.

## ✨ Features

- **PDF Upload** — Drag-and-drop or click to upload, multi-file support
- **Smart Chunking** — Recursive text splitting that respects paragraph boundaries
- **Vector Search** — TF-IDF + random projection embeddings with ChromaDB storage
- **LLM Integration** — Claude API or local Ollama, switchable via env var
- **Cited Answers** — Every answer includes document name + page number sources
- **Source Preview** — Click any citation to see the exact chunk used
- **Chat History** — Persistent conversation history with SQLite

## 🏗️ Architecture

```
Frontend (React)         Backend (FastAPI)          Data Layer
┌────────────┐      ┌──────────────────┐      ┌─────────────┐
│ Upload Zone │─────▶│ Ingestion Pipeline│─────▶│ ChromaDB    │
│ Chat UI     │      │  PDF Parse       │      │ (vectors)   │
│ Source Panel│◀─────│  Chunk           │      ├─────────────┤
└────────────┘      │  Embed           │      │ SQLite      │
                    │  Store           │      │ (metadata)  │
                    ├──────────────────┤      ├─────────────┤
                    │ RAG Engine       │      │ File Storage│
                    │  Retrieve chunks │      │ (PDFs)      │
                    │  Build prompt    │      └─────────────┘
                    │  Call LLM        │
                    └──────────────────┘
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd pyrag
pip install -r requirements.txt
```

### 2. Set up LLM

**Option A — Claude API (recommended):**
```bash
export ANTHROPIC_API_KEY=your_key_here
export LLM_PROVIDER=claude
```

**Option B — Local Ollama:**
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5-coder:7b
export LLM_PROVIDER=ollama
```

### 3. Start the backend

```bash
cd backend
python main.py
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### 4. Open the frontend

The React frontend (`.jsx` file) runs as a Claude artifact or you can integrate it into a Vite project.

Alternatively, use the API directly:

```bash
# Upload a PDF
curl -X POST http://localhost:8000/api/documents/upload \
  -F "files=@your_document.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?"}'
```

## 📁 Project Structure

```
pyrag/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # All configuration
│   ├── api/
│   │   ├── routes_documents.py  # Upload, list, delete
│   │   ├── routes_chat.py       # Ask questions
│   │   └── routes_health.py     # Health check
│   ├── services/
│   │   ├── pdf_parser.py        # PDF → text (PyMuPDF)
│   │   ├── chunker.py           # Smart text chunking
│   │   ├── embedder.py          # TF-IDF embeddings
│   │   ├── vector_store.py      # ChromaDB wrapper
│   │   ├── llm_provider.py      # Claude / Ollama abstraction
│   │   ├── rag_engine.py        # Retrieval + generation
│   │   └── ingestion.py         # Full ingestion pipeline
│   ├── models/
│   │   └── schemas.py           # Pydantic models
│   └── db/
│       └── database.py          # SQLite layer
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

All settings in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Target tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K` | 5 | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | 0.25 | Minimum relevance score |
| `LLM_PROVIDER` | claude | "claude" or "ollama" |
| `MAX_UPLOAD_SIZE` | 50 MB | Max PDF file size |

## 🔧 Upgrading the Embedder

The default embedder uses TF-IDF + random projection (zero dependencies, works offline).
For better quality, swap to sentence-transformers:

```python
# In services/embedder.py, replace the class with:
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()

def embed_query(query):
    return model.encode([query], normalize_embeddings=True)[0].tolist()
```

## 📊 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/documents/upload` | POST | Upload PDF files |
| `/api/documents` | GET | List all documents |
| `/api/documents/{id}` | DELETE | Delete a document |
| `/api/documents/{id}/chunks` | GET | Preview chunks |
| `/api/chat/ask` | POST | Ask a question |
| `/api/chat/history` | GET | Get chat history |
| `/api/health` | GET | System status |

## License

MIT — Built for learning!
