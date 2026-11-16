"""导师反馈：去重不能依赖前端内存。

场景：用户提取后刷新页面，前端丢失 extraction_id，重新提取得到新的批次号再保存。
服务端必须凭 pending_extractions 表自己判断：批次是否由服务端发出、是否已保存、
同一份会议记录是否已保存过。
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import sqlite3
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from app.ai_client import ExtractResult
from app.config import Settings
from app.main import create_app
from tests.test_api import FakeAI, TASKS

MEETING = "项目例会，2026-11-02（周一）。参会：孙悦、王磊。\n1. 孙悦 11 月 6 日前完成首轮客户回访。\n2. 报价单模板要更新，谁来做还没定。"


def make(path):
    return TestClient(create_app(Settings(db_path=path), FakeAI(ExtractResult(summary="纪要", tasks=TASKS))))


@pytest.fixture
def client(tmp_path):
    return make(str(tmp_path / "pending.db"))


def extract(client, text=MEETING):
    r = client.post("/api/extract", json={"meeting_text": text})
    assert r.status_code == 200
    return r.json()["extraction_id"]


def save(client, batch, force=False):
    return client.post("/api/tasks", json={"extraction_id": batch, "tasks": TASKS, "force": force})


def task_count(client):
    return len(client.get("/api/tasks").json()["tasks"])


def test_extract_registers_pending_with_task_summary(client):
    batch = extract(client)
    with closing(sqlite3.connect(client.app.state.db.path)) as db, db:
        row = db.execute("SELECT status, task_count, task_summary, meeting_preview FROM pending_extractions "
                         "WHERE extraction_id = ?", (batch,)).fetchone()
    assert row[0] == "pending" and row[1] == 2
    assert json.loads(row[2]) == ["完成首轮客户回访", "更新报价单模板"]
    assert row[3].startswith("项目例会")
    assert task_count(client) == 0  # 登记批次不等于保存任务


def test_refresh_then_reextract_same_meeting_is_blocked(client):
    """刷新页面 → 重新提取（新批次号）→ 保存：同一批任务不能再存一遍。"""
    assert save(client, extract(client)).status_code == 200
    second = save(client, extract(client))
    assert second.status_code == 409
    body = second.json()
    assert body["code"] == "duplicate_meeting" and body["saved_count"] == 2
    assert "确认要再保存一次吗" in body["detail"]
    assert task_count(client) == 2


def test_whitespace_only_difference_is_same_meeting(client):
    assert save(client, extract(client)).status_code == 200
    reformatted = "  " + MEETING.replace("\n", "\n\n   ") + "\n"
    assert save(client, extract(client, reformatted)).status_code == 409


def test_force_after_user_confirms(client):
    assert save(client, extract(client)).status_code == 200
    batch = extract(client)
    assert save(client, batch).status_code == 409
    # 被拒绝时整笔回滚，批次仍是 pending，用户确认后可以用同一批次再存。
    assert save(client, batch, force=True).status_code == 200
    assert task_count(client) == 4
    assert save(client, batch, force=True).status_code == 409  # 同一批次仍只能存一次


def test_different_meetings_are_independent(client):
    assert save(client, extract(client)).status_code == 200
    assert save(client, extract(client, MEETING + "\n3. 王磊周五前整理竞品资料。")).status_code == 200
    assert task_count(client) == 4


def test_forged_batch_id_is_rejected(client):
    r = save(client, str(uuid4()))
    assert r.status_code == 404 and "请重新提取" in r.json()["detail"]
    assert task_count(client) == 0


def test_expired_batch_is_rejected(client):
    batch = extract(client)
    with closing(sqlite3.connect(client.app.state.db.path)) as db, db:
        db.execute("UPDATE pending_extractions SET created_at = '2020-01-01T00:00:00' WHERE extraction_id = ?", (batch,))
    assert save(client, batch).status_code == 404
    assert task_count(client) == 0


def test_same_batch_twice(client):
    batch = extract(client)
    assert save(client, batch).status_code == 200
    r = save(client, batch)
    assert r.status_code == 409 and r.json()["detail"] == "这批任务已保存过"


def test_source_meeting_comes_from_server_record(client):
    batch = extract(client)
    r = client.post("/api/tasks", json={"extraction_id": batch, "meeting_text": "伪造的来源", "tasks": TASKS})
    assert r.status_code == 200
    assert all(t["source_meeting"].startswith("项目例会") for t in r.json()["tasks"])


def test_concurrent_saves_of_same_meeting_keep_one(tmp_path):
    """两个标签页各提取一次同一份记录，同时保存：只能成功一个。"""
    path = str(tmp_path / "race.db")
    c1, c2 = make(path), make(path)
    b1, b2 = extract(c1), extract(c2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = sorted(pool.map(lambda pair: save(*pair).status_code, [(c1, b1), (c2, b2)]))
    assert codes == [200, 409]
    assert task_count(c1) == 2
