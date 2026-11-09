"""接口测试：用假的 AI 客户端代替真实调用，数据库用临时文件。"""
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.ai_client import AIError, AITimeoutError, ExtractResult
from app.config import Settings
from app.main import create_app


class FakeAI:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def extract(self, text):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


TASKS = [
    {"task": "完成首轮客户回访", "owner": "孙悦", "due_date": "2026-11-06", "due_date_raw": "11月6日前",
     "priority": "未明确", "priority_basis": "无"},
    {"task": "更新报价单模板", "owner": "未明确", "due_date": "未明确", "due_date_raw": "未明确",
     "priority": "未明确", "priority_basis": "无"},
]


def make_client(tmp_path, ai):
    settings = Settings(ai_provider="mock", db_path=str(tmp_path / "test.db"))
    return TestClient(create_app(settings=settings, ai_client=ai))


def test_extract_returns_preview_without_saving(tmp_path):
    client = make_client(tmp_path, FakeAI(ExtractResult(summary="纪要", tasks=TASKS)))
    r = client.post("/api/extract", json={"meeting_text": "项目例会……"})
    assert r.status_code == 200
    assert len(r.json()["tasks"]) == 2 and r.json()["has_tasks"] is True
    assert client.get("/api/tasks").json()["tasks"] == []  # 预览不写库


def test_extract_ai_error_saves_nothing(tmp_path):
    client = make_client(tmp_path, FakeAI(error=AIError("扣子接口报错：code=4101")))
    r = client.post("/api/extract", json={"meeting_text": "项目例会……"})
    assert r.status_code == 502 and "4101" in r.json()["detail"]
    assert client.get("/api/tasks").json()["tasks"] == []


def test_extract_timeout_returns_504(tmp_path):
    client = make_client(tmp_path, FakeAI(error=AITimeoutError("超时")))
    assert client.post("/api/extract", json={"meeting_text": "x"}).status_code == 504


def test_extract_rejects_empty_text(tmp_path):
    ai = FakeAI(ExtractResult(summary="", tasks=[]))
    client = make_client(tmp_path, ai)
    assert client.post("/api/extract", json={"meeting_text": "   "}).status_code == 422
    assert ai.calls == 0


def test_save_sets_status_by_owner(tmp_path):
    client = make_client(tmp_path, FakeAI())
    r = client.post("/api/tasks", json={"extraction_id": str(uuid4()), "meeting_text": "项目例会，2026年10月28日……", "tasks": TASKS})
    assert r.status_code == 200 and r.json()["saved"] == 2
    by_task = {t["task"]: t for t in client.get("/api/tasks").json()["tasks"]}
    assert by_task["完成首轮客户回访"]["status"] == "待开始"
    assert by_task["更新报价单模板"]["status"] == "待分配"
    assert by_task["更新报价单模板"]["source_meeting"].startswith("项目例会")


def test_list_filter_and_patch(tmp_path):
    client = make_client(tmp_path, FakeAI())
    client.post("/api/tasks", json={"extraction_id": str(uuid4()), "meeting_text": "m", "tasks": TASKS})
    pending = client.get("/api/tasks", params={"status": "待分配"}).json()["tasks"]
    assert len(pending) == 1
    tid = pending[0]["id"]

    # 给待分配任务填负责人 → 自动变为待开始
    r = client.patch(f"/api/tasks/{tid}", json={"owner": "李娜"})
    assert r.status_code == 200 and r.json()["status"] == "待开始" and r.json()["owner"] == "李娜"

    r = client.patch(f"/api/tasks/{tid}", json={"status": "已完成"})
    assert r.json()["status"] == "已完成"

    assert client.patch(f"/api/tasks/{tid}", json={"status": "随便写"}).status_code == 422
    assert client.patch("/api/tasks/9999", json={"status": "进行中"}).status_code == 404
    assert client.get("/api/tasks", params={"status": "不存在"}).status_code == 400


def test_health_never_exposes_token(tmp_path):
    settings = Settings(ai_provider="mock", coze_api_token="SECRET-TOKEN", coze_bot_id="123",
                        db_path=str(tmp_path / "t.db"))
    client = TestClient(create_app(settings=settings, ai_client=FakeAI()))
    assert "SECRET-TOKEN" not in client.get("/api/health").text
