"""用 httpx.MockTransport 模拟扣子接口，验证"发起对话 → 轮询 → 取消息"流程（不联网）。"""
import httpx
import pytest

from app.ai_client import AIError, AITimeoutError, CozeClient
from app.config import Settings

ANSWER = '纪要\n```json\n[{"task": "更新报价单模板", "owner": "未明确"}]\n```'


def make(handler, timeout=5):
    settings = Settings(ai_provider="coze", coze_api_token="tok", coze_bot_id="bot1",
                        coze_timeout_seconds=timeout)
    http = httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "Bearer tok"})
    return CozeClient(settings, poll_interval=0.01, http=http)


def test_full_flow():
    polls = {"n": 0}

    def handler(req: httpx.Request):
        path = req.url.path
        if path == "/v3/chat":
            body = req.read().decode()
            assert '"bot_id":"bot1"' in body.replace(" ", "") and '"stream":false' in body.replace(" ", "")
            return httpx.Response(200, json={"code": 0, "data": {"id": "c1", "conversation_id": "v1", "status": "in_progress"}})
        if path == "/v3/chat/retrieve":
            polls["n"] += 1
            status = "completed" if polls["n"] >= 2 else "in_progress"
            return httpx.Response(200, json={"code": 0, "data": {"status": status}})
        if path == "/v3/chat/message/list":
            return httpx.Response(200, json={"code": 0, "data": [
                {"role": "assistant", "type": "verbose", "content": "{}"},
                {"role": "assistant", "type": "answer", "content": ANSWER},
                {"role": "assistant", "type": "follow_up", "content": "还想问什么？"},
            ]})
        return httpx.Response(404)

    r = make(handler).extract("会议记录")
    assert polls["n"] == 2
    assert r.tasks[0]["task"] == "更新报价单模板" and r.summary == "纪要"


def test_permission_error_is_reported():
    def handler(req):
        return httpx.Response(200, json={"code": 4101, "msg": "token permission denied"})
    with pytest.raises(AIError, match="4101"):
        make(handler).extract("x")


def test_failed_chat():
    def handler(req):
        if req.url.path == "/v3/chat":
            return httpx.Response(200, json={"code": 0, "data": {"id": "c", "conversation_id": "v", "status": "in_progress"}})
        return httpx.Response(200, json={"code": 0, "data": {"status": "failed", "last_error": {"msg": "model error"}}})
    with pytest.raises(AIError, match="model error"):
        make(handler).extract("x")


def test_timeout():
    def handler(req):
        if req.url.path == "/v3/chat":
            return httpx.Response(200, json={"code": 0, "data": {"id": "c", "conversation_id": "v", "status": "in_progress"}})
        return httpx.Response(200, json={"code": 0, "data": {"status": "in_progress"}})
    with pytest.raises(AITimeoutError):
        make(handler, timeout=0.05).extract("x")


def test_missing_config():
    with pytest.raises(ValueError, match="COZE_API_TOKEN"):
        CozeClient(Settings(ai_provider="coze"))
