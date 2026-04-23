"""
PyRAG Configuration
All settings in one place — easy to tune.
"""

import os
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = BASE_DIR / "chroma_db"
SQLITE_PATH = BASE_DIR / "pyrag.db"

STORAGE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ── Chunking ──
CHUNK_SIZE = 500          # Target tokens per chunk
CHUNK_OVERLAP = 50        # Overlap tokens between chunks
MIN_CHUNK_SIZE = 50       # Skip chunks smaller than this

# ── Embedding ──
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 256

# ── Retrieval ──
TOP_K = 5                          # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.05        # Minimum similarity score (0-1)

# ── LLM ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "claude" or "ollama"
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# ── Server ──
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
