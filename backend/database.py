import sqlite3
import json
import time
import os
from typing import List, Dict, Any, Optional

DB_FILE = os.path.join(os.path.dirname(__file__), "chat_history.db")

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model_repo TEXT,
                quant_file TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        """)
        conn.commit()

init_db()

def create_session(session_id: str, title: str, model_repo: str = "", quant_file: str = "") -> Dict[str, Any]:
    now = time.time()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, title, model_repo, quant_file, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, title, model_repo, quant_file, now, now)
        )
        conn.commit()
    return {"id": session_id, "title": title, "model_repo": model_repo, "quant_file": quant_file, "created_at": now, "updated_at": now}

def get_all_sessions() -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, session_id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def add_message(session_id: str, role: str, content: str) -> Dict[str, Any]:
    now = time.time()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        cursor.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )
        conn.commit()
        msg_id = cursor.lastrowid
    return {"id": msg_id, "session_id": session_id, "role": role, "content": content, "created_at": now}

def update_session_title(session_id: str, title: str) -> Dict[str, Any]:
    now = time.time()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id)
        )
        conn.commit()
    return {"id": session_id, "title": title, "updated_at": now}

def delete_session(session_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()

