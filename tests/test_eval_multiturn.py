"""scripts/eval_multiturn.py 的离线测试：用替身替换 urlopen，不联网、不读取 .env。"""
import io
import json
import re
import urllib.error
from pathlib import Path

import pytest

from scripts import eval_multiturn as em

CASES = Path(__file__).parent / "multiturn_cases.json"
CFG = {"url": "https://office.example.invalid/stream_run", "token": "SYNTHETIC-OFFICE-TOKEN", "project_id": 1}


class FakeResponse:
    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *exc):
        return False


def sse(answer: str, nested: bool = False) -> list[str]:
    lines = ["event: message_start\n", 'data: {"type": "message_start"}\n', "\n"]
    for i in range(0, len(answer), 5):
        piece = answer[i:i + 5]
        payload = {"data": {"content": {"answer": piece}}} if nested else {"type": "answer", "content": {"answer": piece}}
        lines += ["event: answer\n", "data: " + json.dumps(payload, ensure_ascii=False) + "\n", "\n"]
    return lines + ["event: message_end\n", 'data: {"type": "message_end"}\n']


@pytest.fixture
def fake_agent(monkeypatch):
    """按 session_id 记录历史的假智能体；回答里带上轮次，便于检查是否复用会话。"""
    calls = []
    history: dict[str, list[str]] = {}

    def fake_urlopen(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        sid = body["session_id"]
        text = body["content"]["query"]["prompt"][0]["content"]["text"]
        calls.append({"session_id": sid, "text": text, "auth": req.get_header("Authorization")})
        history.setdefault(sid, []).append(text)
        if "刚才润色的结果" in text:
            answer = "蓝鲸项目下周上线，请抓紧测试。" if any("蓝鲸" in t for t in history[sid]) else "本次对话中没有找到你说的内容。"
        else:
            answer = f"第{len(history[sid])}轮：{text[:20]}"
        return FakeResponse(sse(answer))

    monkeypatch.setattr(em.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_parse_answer_piece_supports_both_paths_and_ignores_other_events():
    assert em.parse_answer_piece({"type": "answer", "content": {"answer": "你好"}}) == "你好"
    assert em.parse_answer_piece({"data": {"content": {"answer": "世界"}}}) == "世界"
    assert em.parse_answer_piece({"type": "message_start"}) == ""
    assert em.parse_answer_piece({"content": "纯字符串"}) == ""


def test_ask_concatenates_stream_and_sends_bearer(fake_agent):
    answer = em.ask(CFG, "请记住项目代号：蓝鲸", "sid-1", timeout=5)
    assert answer.startswith("第1轮")
    assert fake_agent[0]["auth"] == "Bearer SYNTHETIC-OFFICE-TOKEN"


def test_ask_nested_format(monkeypatch):
    monkeypatch.setattr(em.urllib.request, "urlopen", lambda req, timeout: FakeResponse(sse("嵌套格式回答", nested=True)))
    assert em.ask(CFG, "x", "sid", timeout=5) == "嵌套格式回答"


def test_ask_http_error_does_not_leak_token(monkeypatch):
    def unauthorized(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b""))

    monkeypatch.setattr(em.urllib.request, "urlopen", unauthorized)
    with pytest.raises(em.AgentError) as info:
        em.ask(CFG, "x", "sid", timeout=5)
    assert str(info.value) == "HTTP 401"
    assert "SYNTHETIC" not in str(info.value)


def test_ask_empty_answer_is_an_error(monkeypatch):
    monkeypatch.setattr(em.urllib.request, "urlopen", lambda req, timeout: FakeResponse(sse("")))
    with pytest.raises(em.AgentError):
        em.ask(CFG, "x", "sid", timeout=5)


def test_check_turn_rules():
    turn = {"must_include": ["李娜"], "any_of": ["未明确", "待定"], "must_not_include": ["王总"]}
    assert em.check_turn("李娜负责，负责人未明确", turn) == []
    problems = em.check_turn("王总负责", turn)
    assert len(problems) == 3


def test_run_case_reuses_session_within_case_and_isolates_cases(fake_agent):
    materials = {"A": "会议记录"}
    case = {"id": "X1", "turns": [{"input": "{{A}}"}, {"input": "第二轮"}, {"input": "第三轮"}]}
    first = em.run_case(CFG, case, materials, timeout=5)
    second = em.run_case(CFG, {"id": "X2", "turns": [{"input": "新会话"}]}, materials, timeout=5)
    assert [t["answer"][:3] for t in first["turns"]] == ["第1轮", "第2轮", "第3轮"]
    assert first["turns"][0]["input"] == "会议记录"
    assert len({c["session_id"] for c in fake_agent[:3]}) == 1
    assert fake_agent[3]["session_id"] != fake_agent[0]["session_id"]
    assert second["turns"][0]["answer"].startswith("第1轮")


def test_run_case_skips_missing_material(fake_agent):
    result = em.run_case(CFG, {"id": "M2", "requires": ["B"], "turns": [{"input": "{{B}}"}]}, {"B": ""}, timeout=5)
    assert result["status"] == "SKIP"
    assert fake_agent == []


def test_probe_detects_memory(fake_agent):
    assert em.probe(CFG, timeout=5) is True
    assert fake_agent[0]["session_id"] == fake_agent[1]["session_id"]


def test_report_names_never_collide_and_contain_no_token(tmp_path, fake_agent):
    result = em.run_case(CFG, {"id": "X1", "turns": [{"input": "你好"}]}, {}, timeout=5)
    first, _ = em.write_report([result], tmp_path)
    second, _ = em.write_report([result], tmp_path)
    assert first != second
    assert re.fullmatch(r"multiturn_\d{8}_\d{6}(_\d+)?\.md", first.name)
    for path in tmp_path.iterdir():
        assert "SYNTHETIC-OFFICE-TOKEN" not in path.read_text(encoding="utf-8")


def test_load_config_requires_env(monkeypatch):
    for key in (em.ENV_URL, em.ENV_TOKEN, em.ENV_PROJECT):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit):
        em.load_config()


def test_cases_file_is_consistent():
    data = json.loads(CASES.read_text(encoding="utf-8"))
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == 18 and len(set(ids)) == 18
    materials = set(data["materials"])
    for case in data["cases"]:
        assert case["turns"], case["id"]
        assert set(case.get("requires", [])) <= materials
        for turn in case["turns"]:
            for name in re.findall(r"\{\{(\w+)\}\}", turn["input"]):
                assert name in materials, (case["id"], name)
