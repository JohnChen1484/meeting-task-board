"""SQLite 存储；批次与任务在同一事务内保存。

去重分两层，都在服务端判定，不依赖浏览器内存：
1. 批次：/api/extract 把 extraction_id 登记到 pending_extractions，保存时只接受
   服务端发出、未过期、状态仍为 pending 的批次；用带条件的 UPDATE 抢占，
   同一批次只能保存一次（并发时也只有一个请求成功）。
2. 内容：按规范化后的会议记录 sha256 判断。同一份会议记录已保存过时默认拒绝，
   用户明确确认（force）后才允许再次保存——刷新页面后重新提取拿到新批次号，
   也绕不过这一层。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Iterable, List, Optional

STATUSES = ["待分配", "待开始", "进行中", "已完成"]
UNKNOWN = "未明确"
PENDING_TTL = timedelta(hours=24)

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
CREATE TABLE IF NOT EXISTS pending_extractions (
    extraction_id TEXT PRIMARY KEY,
    meeting_hash TEXT NOT NULL,
    meeting_preview TEXT NOT NULL DEFAULT '',
    task_summary TEXT NOT NULL DEFAULT '[]',
    task_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    saved_at TEXT,
    saved_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pending_meeting_hash ON pending_extractions (meeting_hash, status);
CREATE TABLE IF NOT EXISTS saved_batches (
    extraction_id TEXT PRIMARY KEY,
    saved_at TEXT NOT NULL,
    chat_id TEXT NOT NULL DEFAULT ''
);
"""


class DuplicateBatchError(Exception):
    pass


class UnknownExtractionError(Exception):
    """批次号不是服务端发出的，或已过期。"""


class DuplicateMeetingError(Exception):
    """同一份会议记录已经保存过。"""

    def __init__(self, saved_at: str, saved_count: int):
        super().__init__("这份会议记录已保存过")
        self.saved_at = saved_at
        self.saved_count = saved_count


def meeting_fingerprint(text: str) -> str:
    """规范化空白后取 sha256：只差空格、换行的同一份记录视为相同。"""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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

    def create_pending(self, extraction_id: str, meeting_text: str, tasks: Iterable[dict]) -> None:
        """提取成功后登记批次：批次号、会议记录指纹、任务摘要。"""
        names = [str(t.get("task", ""))[:40] for t in tasks]
        preview = (meeting_text or "").strip().replace("\n", " ")[:50]
        with self._conn() as c:
            c.execute(
                """INSERT INTO pending_extractions
                   (extraction_id, meeting_hash, meeting_preview, task_summary, task_count, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (extraction_id, meeting_fingerprint(meeting_text), preview,
                 json.dumps(names, ensure_ascii=False), len(names), self._now()),
            )

    def add_many(self, tasks: Iterable[dict], extraction_id: str, chat_id: str = "", force: bool = False) -> List[dict]:
        tasks = list(tasks)
        now = self._now()
        saved = []
        with self._conn() as c:
            row = c.execute("SELECT * FROM pending_extractions WHERE extraction_id = ?", (extraction_id,)).fetchone()
            if row is None:
                raise UnknownExtractionError("提取批次不存在")
            if row["status"] == "saved":
                raise DuplicateBatchError("这批任务已保存过")
            if datetime.now() - datetime.fromisoformat(row["created_at"]) > PENDING_TTL:
                raise UnknownExtractionError("提取批次已过期")
            # 先抢占批次（写锁），之后的读取都能看到其他请求已提交的结果。
            cur = c.execute(
                "UPDATE pending_extractions SET status = 'saved', saved_at = ?, saved_count = ? "
                "WHERE extraction_id = ? AND status = 'pending'",
                (now, len(tasks), extraction_id),
            )
            if cur.rowcount == 0:
                raise DuplicateBatchError("这批任务已保存过")
            if not force:
                prev = c.execute(
                    "SELECT saved_at, saved_count FROM pending_extractions "
                    "WHERE meeting_hash = ? AND status = 'saved' AND extraction_id != ? "
                    "ORDER BY saved_at DESC LIMIT 1",
                    (row["meeting_hash"], extraction_id),
                ).fetchone()
                if prev is not None:
                    raise DuplicateMeetingError(prev["saved_at"], prev["saved_count"] or 0)
            source_meeting = row["meeting_preview"]
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
