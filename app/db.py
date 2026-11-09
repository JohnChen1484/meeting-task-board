"""SQLite 存储；批次与任务在同一事务内保存。"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Iterable, List, Optional

STATUSES = ["待分配", "待开始", "进行中", "已完成"]
UNKNOWN = "未明确"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    owner TEXT NOT NULL DEFAULT '未明确',
    due_date TEXT NOT NULL DEFAULT '未明确',
    due_date_raw TEXT NOT NULL DEFAULT '未明确',
    priority TEXT NOT NULL DEFAULT '未明确',
    priority_basis TEXT NOT NULL DEFAULT '无',
    status TEXT NOT NULL,
    source_meeting TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS saved_batches (
    extraction_id TEXT PRIMARY KEY,
    saved_at TEXT NOT NULL,
    chat_id TEXT NOT NULL DEFAULT ''
);
"""


class DuplicateBatchError(Exception):
    pass


def initial_status(owner: str) -> str:
    return "待分配" if not owner or owner.strip() in ("", UNKNOWN) else "待开始"


class TaskDB:
    def __init__(self, path: str):
        self.path = path
        self._lock = RLock()
        self._memory = None
        if path == ":memory:":
            self._memory = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            if "chat_id" not in {r["name"] for r in c.execute("PRAGMA table_info(saved_batches)")}:
                c.execute("ALTER TABLE saved_batches ADD COLUMN chat_id TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def _conn(self):
        # 内存连接跨 FastAPI 工作线程复用；锁保护完整事务。
        with self._lock:
            conn = self._memory if self._memory is not None else sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                if self._memory is None:
                    conn.close()

    def close(self):
        with self._lock:
            if self._memory is not None:
                self._memory.close()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def add_many(self, tasks: Iterable[dict], source_meeting: str, extraction_id: str, chat_id: str = "") -> List[dict]:
        now = self._now()
        saved = []
        with self._conn() as c:
            try:
                c.execute("INSERT INTO saved_batches (extraction_id, saved_at, chat_id) VALUES (?, ?, ?)", (extraction_id, now, chat_id))
            except sqlite3.IntegrityError:
                raise DuplicateBatchError("这批任务已保存过") from None
            for t in tasks:
                cur = c.execute(
                    """INSERT INTO tasks (task, owner, due_date, due_date_raw, priority, priority_basis,
                                          status, source_meeting, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (t["task"], t["owner"], t["due_date"], t["due_date_raw"], t["priority"],
                     t["priority_basis"], initial_status(t["owner"]), source_meeting, now, now),
                )
                saved.append(dict(c.execute("SELECT * FROM tasks WHERE id = ?", (cur.lastrowid,)).fetchone()))
        return saved

    def get(self, task_id: int) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list(self, status: Optional[str] = None) -> List[dict]:
        with self._conn() as c:
            if status:
                rows = c.execute("SELECT * FROM tasks WHERE status = ? ORDER BY id DESC", (status,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def update(self, task_id: int, fields: dict) -> Optional[dict]:
        with self._conn() as c:
            # 先获得写锁，再读取状态，避免并发修改使负责人/状态联动失效。
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                return None
            current = dict(row)
            allowed = {k: v for k, v in fields.items() if k in ("status", "owner", "due_date", "priority")}
            owner = allowed.get("owner", current["owner"])
            if owner == UNKNOWN and allowed.get("status") == "待开始":
                raise ValueError("请先指定负责人")
            if "owner" in allowed and "status" not in allowed:
                if current["status"] == "待分配" and owner != UNKNOWN:
                    allowed["status"] = "待开始"
                elif current["status"] == "待开始" and owner == UNKNOWN:
                    allowed["status"] = "待分配"
            if not allowed:
                return current
            allowed["updated_at"] = self._now()
            cols = ", ".join(f"{k} = ?" for k in allowed)
            c.execute(f"UPDATE tasks SET {cols} WHERE id = ?", (*allowed.values(), task_id))
            return dict(c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())

    def delete(self, task_id: int) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM tasks WHERE id = ?", (task_id,)).rowcount > 0
