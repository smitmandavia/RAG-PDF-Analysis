# PyRAG

PyRAG is a local full-stack document intelligence workspace for uploading PDFs, indexing their content, and asking grounded questions with cited source chunks.

## Stack

- Backend: FastAPI, PyMuPDF, ChromaDB, SQLite
- Retrieval: sentence-transformers embeddings plus hybrid keyword reranking
- LLM: Ollama by default, Claude API supported by environment variable
- Frontend: React, Vite, lucide-react

## Project Layout

```text
backend/pyrag/        FastAPI backend and RAG services
frontend/             Vite React frontend
```

## Backend Setup

```bash
cd backend/pyrag
pip install -r requirements.txt
cd backend
python main.py
```

The backend runs at `http://localhost:8000`.

Optional LLM settings:

```bash
set LLM_PROVIDER=ollama
set OLLAMA_MODEL=mistral
```

For Claude:

```bash
set LLM_PROVIDER=claude
set ANTHROPIC_API_KEY=your_key_here
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173` and proxies `/api` to the backend.

## Notes

Runtime artifacts are intentionally ignored by Git:

- Uploaded PDFs
- SQLite database files
- Chroma vector indexes
- Logs
- `node_modules`
- Vite build output

Upload PDFs again after cloning to build a fresh local index.
