import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# default db path - can be overridden in tests
DEFAULT_DB = "sessions.db"


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB) -> None:
    """Create tables on first run. Safe to call multiple times."""
    conn = _get_conn(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- stores what the agent actually did for each query
            -- useful for debugging and for the eval harness
            CREATE TABLE IF NOT EXISTS turns (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT NOT NULL,
                query            TEXT NOT NULL,
                search_queries   TEXT DEFAULT '[]',
                urls_opened      TEXT DEFAULT '[]',
                context_snippets TEXT DEFAULT '[]',
                final_answer     TEXT DEFAULT '',
                timestamp        TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)
        conn.commit()
    finally:
        conn.close()


def create_session(db_path: str = DEFAULT_DB) -> str:
    sid = str(uuid.uuid4())
    now = _ts()
    conn = _get_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (sid, now, now)
        )
        conn.commit()
    finally:
        conn.close()
    return sid


def list_sessions(db_path: str = DEFAULT_DB) -> List[Dict]:
    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_message(session_id: str, role: str, content: str, db_path: str = DEFAULT_DB) -> None:
    now = _ts()
    conn = _get_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_conversation_history(session_id: str, db_path: str = DEFAULT_DB) -> List[Dict]:
    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_turn(
    session_id: str,
    query: str,
    search_queries: List[str],
    urls_opened: List[str],
    context_snippets: List[str],
    final_answer: str,
    db_path: str = DEFAULT_DB,
) -> None:
    conn = _get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO turns
               (session_id, query, search_queries, urls_opened, context_snippets, final_answer, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                query,
                json.dumps(search_queries),
                json.dumps(urls_opened),
                json.dumps(context_snippets[:5]),  # don't need all of them
                final_answer,
                _ts(),
            )
        )
        conn.commit()
    finally:
        conn.close()


def get_turn_history(session_id: str, db_path: str = DEFAULT_DB) -> List[Dict]:
    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
    finally:
        conn.close()

    result = []
    for r in rows:
        d = dict(r)
        d["search_queries"]   = json.loads(d["search_queries"])
        d["urls_opened"]      = json.loads(d["urls_opened"])
        d["context_snippets"] = json.loads(d["context_snippets"])
        result.append(d)
    return result


def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
