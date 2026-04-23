"""
PyRAG — SQLite Database Layer
Stores document metadata and chat history.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from config import SQLITE_PATH


def get_connection():
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            page_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0,
            status TEXT DEFAULT 'processing',
            uploaded_at TEXT NOT NULL,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,  -- JSON
            processing_time REAL,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# ── Document CRUD ──

def insert_document(doc_id: str, filename: str, file_path: str, file_size: int):
    conn = get_connection()
    conn.execute(
        """INSERT INTO documents (id, filename, file_path, file_size, uploaded_at)
           VALUES (?, ?, ?, ?, ?)""",
        (doc_id, filename, file_path, file_size, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def update_document(doc_id: str, **kwargs):
    conn = get_connection()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [doc_id]
    conn.execute(f"UPDATE documents SET {sets} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_document(doc_id: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_documents():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document(doc_id: str):
    conn = get_connection()
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()


def get_documents_count():
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'ready'").fetchone()[0]
    conn.close()
    return count


def get_total_chunks():
    conn = get_connection()
    total = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) FROM documents WHERE status = 'ready'").fetchone()[0]
    conn.close()
    return total


# ── Chat History ──

def save_chat(question: str, answer: str, sources: list, processing_time: float):
    conn = get_connection()
    conn.execute(
        """INSERT INTO chat_history (question, answer, sources, processing_time, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (question, answer, json.dumps(sources), processing_time, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_chat_history(limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chat_history ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d['sources'] = json.loads(d['sources']) if d['sources'] else []
        results.append(d)
    return results


# Initialize on import
init_db()
