"""
src/db/sqlite_db.py — Persistent chat storage using SQLite.

Schema
------
conversations : id, title, created_at, updated_at
messages      : id, conversation_id, role, content, sources_json, timestamp

All public functions are safe to call concurrently — SQLite WAL mode is
enabled so reads never block writes.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── DB location ────────────────────────────────────────────────────────────────
_DB_PATH: Path | None = None   # set by init_db()


def _db_path() -> Path:
    if _DB_PATH is None:
        raise RuntimeError("Call init_db() before using the database.")
    return _DB_PATH


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(str(_db_path()), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ═════════════════════════════════════════════════════════════════════════════
# INIT
# ═════════════════════════════════════════════════════════════════════════════

def init_db(db_path: str | Path | None = None) -> None:
    """
    Create tables if they don't exist and set the DB path.

    Safe to call multiple times (idempotent).
    """
    global _DB_PATH

    if db_path is None:
        # Default: project_root/data/chat.db
        root = Path(__file__).resolve().parents[3]
        db_path = root / "data" / "chat.db"

    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT        PRIMARY KEY,
                title       TEXT        NOT NULL DEFAULT 'New conversation',
                created_at  REAL        NOT NULL,
                updated_at  REAL        NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id                  INTEGER     PRIMARY KEY AUTOINCREMENT,
                conversation_id     TEXT        NOT NULL
                                    REFERENCES conversations(id) ON DELETE CASCADE,
                role                TEXT        NOT NULL CHECK(role IN ('user','assistant')),
                content             TEXT        NOT NULL,
                sources_json        TEXT,
                model               TEXT,
                intent              TEXT,
                timestamp           REAL        NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_msg_conv
                ON messages(conversation_id, timestamp);
        """)

    log.info("SQLite DB ready: %s", _DB_PATH)


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATIONS
# ═════════════════════════════════════════════════════════════════════════════

def create_conversation(conversation_id: str | None = None) -> str:
    """
    Insert a new conversation row.  Returns the conversation_id.
    """
    cid = conversation_id or str(uuid.uuid4())
    now = time.time()
    with _conn() as con:
        con.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (cid, "New conversation", now, now),
        )
    log.debug("Created conversation: %s", cid)
    return cid


def update_conversation_title(conversation_id: str, title: str) -> None:
    """Set a human-readable title (derived from first user message)."""
    with _conn() as con:
        con.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title[:80], time.time(), conversation_id),
        )


def delete_conversation(conversation_id: str) -> None:
    """Delete conversation + all its messages (CASCADE)."""
    with _conn() as con:
        con.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
    log.info("Deleted conversation: %s", conversation_id)


def load_conversations() -> list[dict]:
    """
    Return all conversations ordered newest first.
    Each dict: id, title, created_at, updated_at, message_count, preview.
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT
                c.id, c.title, c.created_at, c.updated_at,
                COUNT(m.id)   AS message_count,
                MAX(CASE WHEN m.role='user' THEN m.content END) AS last_user_msg
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def touch_conversation(conversation_id: str) -> None:
    """Update updated_at so this conversation floats to the top."""
    with _conn() as con:
        con.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (time.time(), conversation_id),
        )


# ═════════════════════════════════════════════════════════════════════════════
# MESSAGES
# ═════════════════════════════════════════════════════════════════════════════

def save_message(
    conversation_id: str,
    role: str,
    content: str,
    sources: list | None = None,
    model: str = "",
    intent: str = "",
) -> None:
    """
    Persist a single message.  sources is serialised to JSON.
    """
    sources_json = json.dumps(sources) if sources else None
    with _conn() as con:
        con.execute(
            """INSERT INTO messages
               (conversation_id, role, content, sources_json, model, intent, timestamp)
               VALUES (?,?,?,?,?,?,?)""",
            (conversation_id, role, content, sources_json,
             model, intent, time.time()),
        )
        con.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (time.time(), conversation_id),
        )


def load_messages(conversation_id: str) -> list[dict]:
    """
    Return all messages for a conversation ordered by timestamp.
    sources_json is decoded back to list.
    """
    with _conn() as con:
        rows = con.execute(
            """SELECT role, content, sources_json, model, intent, timestamp
               FROM messages
               WHERE conversation_id=?
               ORDER BY timestamp ASC""",
            (conversation_id,),
        ).fetchall()

    result = []
    for r in rows:
        msg = dict(r)
        raw = msg.pop("sources_json", None)
        msg["sources"] = json.loads(raw) if raw else []
        result.append(msg)
    return result


def search_conversations(query: str) -> list[dict]:
    """Full-text search across message content. Returns matching conversation rows."""
    pattern = f"%{query}%"
    with _conn() as con:
        rows = con.execute("""
            SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at
            FROM conversations c
            JOIN messages m ON m.conversation_id = c.id
            WHERE m.content LIKE ?
            ORDER BY c.updated_at DESC
            LIMIT 30
        """, (pattern,)).fetchall()
    return [dict(r) for r in rows]