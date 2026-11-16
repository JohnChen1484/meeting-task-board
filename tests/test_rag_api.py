"""知识库接口离线测试：模拟客户端/HTTP，不访问真实服务。"""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.ai_client import AIError, AITimeoutError, build_client, build_rag_client
from app.config import Settings
from app.main import create_app
from app.rag_answer import to_response

TOKEN = "SYNTHETIC-RAG-API-TOKEN"


def rag_settings(**changes):
    return Settings(**dict({"ai_provider": "mock", "db_path": ":memory:",
                            "rag_deploy_url": "https://rag.example.invalid", "rag_project_id": "123",
                            "rag_api_token": TOKEN, "coze_timeout_seconds": 3}, **changes))


class FakeRag:
    def __init__(self, answer="婚假为 **3 天**。\n请按制度申请。", error=None):
        self.answer, self.error = answer, error
        self.questions = []
        self.closed = False
        self._http = SimpleNamespace(close=self.close)

    def close(self):
        self.closed = True

    def stream_run(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return SimpleNamespace(answer=self.answer, session_id="fake-session")


def test_normal_answer_no_db_no_meeting_state_change(monkeypatch):
    fake = FakeRag()
    app = create_app(rag_settings())
    settings_seen = []

    def factory(settings):
        settings_seen.append(settings)
        return fake

    monkeypatch.setattr("app.main.build_rag_client", factory)

    class NoDatabase:
        def __getattr__(self, name):
            pytest.fail("知识库接口不得访问数据库：" + name)

    app.state.db = NoDatabase()
    with TestClient(app) as client:
        response = client.post("/api/rag/ask", json={"question": "  婚假有几天？  "})
        assert response.status_code == 200
        data = response.json()
        assert set(data) == {"answer", "structured", "status", "sources", "rewritten_query", "intent",
                             "elapsed_seconds"}
        assert data["answer"] == fake.answer and isinstance(data["elapsed_seconds"], (int, float))
        assert data["structured"] is False and data["status"] == "unknown"
        assert data["sources"] == [] and data["rewritten_query"] == data["intent"] == ""
        assert data["elapsed_seconds"] >= 0
        assert app.state.last_call is None
    assert fake.questions == ["婚假有几天？"] and fake.closed
    assert settings_seen[0].rag_api_token == TOKEN


def test_structured_answer_fields_and_body(monkeypatch):
    metadata = {"status": "answered", "sources": [
        {"file": "员工考勤与休假制度（v2）", "section": "第三章 其他假期", "status": "现行", "score": 0.61}
    ], "rewritten_query": "婚假有几天？", "intent": "knowledge"}
    fake = FakeRag(answer="**答案**：婚假 10 天。\n\n```json\n" + json.dumps(metadata, ensure_ascii=False) + "\n```")
    monkeypatch.setattr("app.main.build_rag_client", lambda *_: fake)
    with TestClient(create_app(rag_settings())) as client:
        response = client.post("/api/rag/ask", json={"question": "婚假？"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "**答案**：婚假 10 天。"
    assert data["structured"] is True
    for key in ("status", "sources", "rewritten_query", "intent"):
        assert data[key] == metadata[key]
    assert isinstance(data["elapsed_seconds"], (int, float)) and data["elapsed_seconds"] >= 0
    assert fake.closed


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": " \n\t"},
                                      {"question": "问" * 501}, {"question": " " + "问" * 500},
                                      {}, {"question": None}, {"question": 123}])
def test_bad_questions_are_400_without_call(monkeypatch, payload):
    monkeypatch.setattr("app.main.build_rag_client", lambda *_: pytest.fail("无效输入不得创建客户端"))
    with TestClient(create_app(rag_settings())) as client:
        response = client.post("/api/rag/ask", json=payload)
    assert response.status_code == 400
    assert "问题" in response.json()["detail"]


def test_exactly_500_characters_accepted(monkeypatch):
    fake = FakeRag()
    monkeypatch.setattr("app.main.build_rag_client", lambda *_: fake)
    with TestClient(create_app(rag_settings())) as client:
        assert client.post("/api/rag/ask", json={"question": "问" * 500}).status_code == 200
    assert fake.questions == ["问" * 500]


@pytest.mark.parametrize("field,value,label", [("rag_deploy_url", "", "RAG_DEPLOY_URL"),
                                               ("rag_project_id", "", "RAG_PROJECT_ID"),
                                               ("rag_api_token", "", "RAG_API_TOKEN")])
def test_missing_config_is_503_without_http(monkeypatch, field, value, label):
    monkeypatch.setattr("app.ai_client.CozeDeployClient", lambda *_a, **_k: pytest.fail("缺少配置不得创建客户端"))
    with TestClient(create_app(rag_settings(**{field: value}))) as client:
        response = client.post("/api/rag/ask", json={"question": "婚假有几天？"})
        assert client.get("/api/health").status_code == 200
    assert response.status_code == 503 and label in response.json()["detail"]


@pytest.mark.parametrize("changes", [{"rag_deploy_url": "http://" + TOKEN},
                                     {"rag_deploy_url": "https://rag.example.invalid/stream_run"},
                                     {"rag_project_id": TOKEN},
                                     {"coze_timeout_seconds": 0}, {"coze_timeout_seconds": float("nan")},
                                     {"coze_read_timeout_seconds": 0}, {"coze_read_timeout_seconds": float("nan")}])
def test_invalid_config_is_503_and_never_echoes_values(changes, caplog):
    with TestClient(create_app(rag_settings(**changes))) as client:
        response = client.post("/api/rag/ask", json={"question": "问题"})
    assert response.status_code == 503 and "配置无效" in response.json()["detail"]
    assert TOKEN not in response.text + caplog.text


@pytest.mark.parametrize("error,code,message", [
    (AITimeoutError(TOKEN + "超时"), 504, "超时"),
    (AIError(TOKEN + "无权限"), 502, "调用失败"),
    (RuntimeError(TOKEN), 502, "调用失败"),
])
def test_call_failures_chinese_error_no_secrets_no_state_change(monkeypatch, caplog, error, code, message):
    fake = FakeRag(error=error)
    monkeypatch.setattr("app.main.build_rag_client", lambda *_: fake)
    app = create_app(rag_settings())
    with TestClient(app) as client:
        response = client.post("/api/rag/ask", json={"question": "问题"})
        assert client.get("/api/tasks").json()["tasks"] == []
    assert response.status_code == code and message in response.json()["detail"]
    assert TOKEN not in response.text + caplog.text
    assert fake.closed and len(fake.questions) == 1 and app.state.last_call is None


def test_answer_redacts_token_and_empty_answer_fails(monkeypatch, caplog):
    fake = FakeRag(answer="回答：" + TOKEN)
    monkeypatch.setattr("app.main.build_rag_client", lambda *_: fake)
    with TestClient(create_app(rag_settings())) as client:
        response = client.post("/api/rag/ask", json={"question": "问题"})
        assert response.status_code == 200 and "[已脱敏]" in response.json()["answer"]
        assert TOKEN not in response.text + caplog.text
        fake.answer = " \n"
        assert client.post("/api/rag/ask", json={"question": "问题"}).status_code == 502


def test_structured_only_answer_is_empty_failure(monkeypatch):
    fake = FakeRag(answer=' \n```json\n{"status":"answered","sources":[]}\n```\n ')
    monkeypatch.setattr("app.main.build_rag_client", lambda *_: fake)
    with TestClient(create_app(rag_settings())) as client:
        response = client.post("/api/rag/ask", json={"question": "问题"})
    assert response.status_code == 502
    assert response.json()["detail"] == "知识库调用失败，请检查部署服务状态及 RAG API 令牌权限"
    assert fake.closed


def test_token_in_json_is_redacted_before_structured_parsing(monkeypatch, caplog):
    metadata = {"status": "answered", "sources": [
        {"file": TOKEN, "section": TOKEN, "status": TOKEN, "score": 0.61}
    ], "rewritten_query": TOKEN, "intent": TOKEN}
    fake = FakeRag(answer="回答：" + TOKEN + "\n```json\n" + json.dumps(metadata) + "\n```")
    monkeypatch.setattr("app.main.build_rag_client", lambda *_: fake)
    with TestClient(create_app(rag_settings())) as client:
        response = client.post("/api/rag/ask", json={"question": "问题"})
    assert response.status_code == 200
    data = response.json()
    assert data["structured"] is True and data["status"] == "answered"
    assert data["answer"] == "回答：[已脱敏]"
    assert data["sources"] == [{"file": "[已脱敏]", "section": "[已脱敏]", "status": "[已脱敏]", "score": 0.61}]
    assert data["rewritten_query"] == data["intent"] == "[已脱敏]"
    assert TOKEN not in response.text + caplog.text


def test_unicode_escaped_token_is_redacted_after_parsing(monkeypatch, caplog):
    metadata = {"status": "answered", "sources": [
        {"file": TOKEN, "section": TOKEN, "status": TOKEN, "score": 0.61}
    ], "rewritten_query": TOKEN, "intent": TOKEN}
    escaped_token = f"\\u{ord(TOKEN[0]):04x}" + TOKEN[1:]
    block = json.dumps(metadata).replace(TOKEN, escaped_token)
    fake = FakeRag(answer="回答：" + TOKEN + "\n```json\n" + block + "\n```")
    inputs_seen = []

    def parse_redacted(answer):
        inputs_seen.append(answer)
        assert TOKEN not in answer and "回答：[已脱敏]" in answer
        return to_response(answer)

    monkeypatch.setattr("app.main.build_rag_client", lambda *_: fake)
    monkeypatch.setattr("app.main.to_response", parse_redacted)
    with TestClient(create_app(rag_settings())) as client:
        response = client.post("/api/rag/ask", json={"question": "问题"})
    assert response.status_code == 200 and len(inputs_seen) == 1
    data = response.json()
    assert data["structured"] is True and data["status"] == "answered"
    assert data["answer"] == "回答：[已脱敏]"
    assert data["sources"] == [{"file": "[已脱敏]", "section": "[已脱敏]", "status": "[已脱敏]", "score": 0.61}]
    assert data["rewritten_query"] == data["intent"] == "[已脱敏]"
    assert TOKEN not in response.text + caplog.text


def test_env_config_independent_and_invalid_rag_does_not_break_meeting(monkeypatch):
    for key, value in {"RAG_DEPLOY_URL": "https://rag.example.invalid/", "RAG_PROJECT_ID": "bad-but-lazy",
                       "RAG_API_TOKEN": TOKEN, "COZE_TIMEOUT_SECONDS": "12.5"}.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert settings.rag_deploy_url == "https://rag.example.invalid"
    assert settings.rag_project_id == "bad-but-lazy" and settings.rag_api_token == TOKEN
    assert settings.coze_timeout_seconds == 12.5 and settings.coze_api_token == ""
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["ai_configured"] is True
        assert client.post("/api/rag/ask", json={"question": "问题"}).status_code == 503


def test_reused_sse_sends_raw_question_independent_config_and_unique_sessions(monkeypatch):
    requests, clients = [], []

    def handler(request):
        requests.append(request)
        wire = 'data: {"type":"answer","content":"知识库"}\n\n'
        wire += 'data: {"type":"answer","content":{"text":"回答"}}\n\n'
        wire += 'data: {"type":"message_end"}\n\n'
        return httpx.Response(200, text=wire)

    def factory(settings):
        client = build_rag_client(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))
        assert client.timeout >= 120 and client.read_timeout >= 60
        clients.append(client)
        return client

    monkeypatch.setattr("app.main.build_rag_client", factory)
    monkeypatch.setattr("app.ai_client.parse_answer", lambda *_: pytest.fail("知识库不得解析会议 JSON"))
    settings = rag_settings(coze_deploy_url="https://meeting.example.invalid", coze_project_id=999,
                            coze_api_token="SYNTHETIC-MEETING-TOKEN")
    with TestClient(create_app(settings)) as client:
        for _ in range(2):
            response = client.post("/api/rag/ask", json={"question": "婚假有几天？"})
            assert response.status_code == 200 and response.json()["answer"] == "知识库回答"
    sessions = []
    for request in requests:
        body = json.loads(request.content)
        assert str(request.url) == "https://rag.example.invalid/stream_run"
        assert request.headers["Authorization"] == "Bearer " + TOKEN
        assert body["project_id"] == 123
        assert body["content"]["query"]["prompt"][0]["content"]["text"] == "婚假有几天？"
        sessions.append(body["session_id"])
    assert len(sessions) == 2 and len(set(sessions)) == 2
    assert all(client._http.is_closed for client in clients)
    assert settings.coze_project_id == 999 and settings.coze_api_token == "SYNTHETIC-MEETING-TOKEN"


@pytest.mark.parametrize("timeout,read_timeout", [(3, 20), (120, 60), (180, 90)])
def test_rag_timeout_floors_leave_meeting_client_unchanged(timeout, read_timeout):
    settings = rag_settings(ai_provider="coze_deploy", coze_timeout_seconds=timeout,
                            coze_read_timeout_seconds=read_timeout,
                            coze_deploy_url="https://meeting.example.invalid", coze_project_id=999,
                            coze_api_token="SYNTHETIC-MEETING-TOKEN")
    rag_client = build_rag_client(settings, http=httpx.Client(transport=httpx.MockTransport(
        lambda *_: pytest.fail("检查超时配置不得发送请求"))))
    meeting_client = build_client(settings)
    try:
        assert rag_client.timeout == max(timeout, 120)
        assert rag_client.read_timeout == max(read_timeout, 60)
        assert meeting_client.timeout == timeout and meeting_client.read_timeout == read_timeout
        assert settings.coze_timeout_seconds == timeout and settings.coze_read_timeout_seconds == read_timeout
    finally:
        rag_client._http.close()
        meeting_client._http.close()


def test_page_routes_and_navigation_no_ai_call(monkeypatch):
    monkeypatch.setattr("app.main.build_rag_client", lambda *_: pytest.fail("打开页面不得调用智能体"))
    with TestClient(create_app(Settings(db_path=":memory:"))) as client:
        for path in ("/", "/rag"):
            response = client.get(path)
            assert response.status_code == 200
            assert 'href="/"' in response.text and 'href="/rag"' in response.text
            assert "会议纪要任务看板" in response.text and "知识库问答" in response.text
