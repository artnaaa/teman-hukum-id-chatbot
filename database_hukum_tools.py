# database_hukum_tools.py
import sqlite3
from typing import List, Dict, Any, Optional

HUKUM_DB_PATH = "hukum_data.db"

def hukum_init_database():
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    # profiles
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            profile_text TEXT
        )
        """
    )
    # conversations
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    # chats (ensure new schema)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            conversation_id INTEGER,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        """
    )
    # migrate: add conversation_id if missing
    cur.execute("PRAGMA table_info(chats)")
    cols = [r[1] for r in cur.fetchall()]
    if "conversation_id" not in cols:
        try:
            cur.execute("ALTER TABLE chats ADD COLUMN conversation_id INTEGER")
        except Exception:
            pass
    # ensure at least one default conversation exists
    cur.execute("SELECT id FROM conversations ORDER BY id LIMIT 1")
    row = cur.fetchone()
    default_conv_id = None
    if not row:
        cur.execute("INSERT INTO conversations (title) VALUES (?)", ("Percakapan",))
        default_conv_id = cur.lastrowid
    else:
        default_conv_id = row[0]
    # backfill NULL conversation_id
    cur.execute("UPDATE chats SET conversation_id = ? WHERE conversation_id IS NULL", (default_conv_id,))

    # pdf chunks
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            page INTEGER,
            chunk_text TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    return "Hukum DB ready"

def hukum_save_profile(profile_text: str):
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO profiles (id, profile_text) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET profile_text=excluded.profile_text",
        (profile_text,),
    )
    conn.commit()
    conn.close()
    return True

def hukum_get_profile() -> str:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT profile_text FROM profiles WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else ""

def hukum_create_conversation(title: str) -> int:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO conversations (title) VALUES (?)", (title or "Percakapan",))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return cid

def hukum_list_conversations(limit: int = 100) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{k: r[k] for k in r.keys()} for r in rows]

def hukum_update_conversation_title(conversation_id: int, new_title: str) -> bool:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (new_title or "Percakapan", conversation_id),
    )
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return affected > 0

def hukum_delete_conversation(conversation_id: int) -> bool:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    # delete chats first
    cur.execute("DELETE FROM chats WHERE conversation_id = ?", (conversation_id,))
    # then delete the conversation row
    cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    return affected > 0

def hukum_add_message(role: str, content: str, conversation_id: int) -> int:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chats (role, content, conversation_id) VALUES (?, ?, ?)",
        (role, content, conversation_id),
    )
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id

def hukum_get_messages(limit: int = 50, conversation_id: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if conversation_id is None:
        cur.execute(
            "SELECT id, role, content, created_at, conversation_id FROM chats ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    else:
        cur.execute(
            "SELECT id, role, content, created_at, conversation_id FROM chats WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        )
    rows = cur.fetchall()
    conn.close()
    return [{k: r[k] for k in r.keys()} for r in rows][::-1]

def hukum_clear_conversation(conversation_id: Optional[int] = None):
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    if conversation_id is None:
        cur.execute("DELETE FROM chats")
    else:
        cur.execute("DELETE FROM chats WHERE conversation_id = ?", (conversation_id,))
    conn.commit()
    conn.close()
    return True

def hukum_upsert_pdf_chunk(filename: str, page: int, chunk_text: str) -> int:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO pdf_chunks (filename, page, chunk_text) VALUES (?, ?, ?)",
        (filename, page, chunk_text),
    )
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id

def hukum_get_pdf_chunks(limit: int = 100) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(HUKUM_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT id, filename, page, chunk_text FROM pdf_chunks ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{k: r[k] for k in r.keys()} for r in rows]
