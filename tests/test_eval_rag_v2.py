"""RAG 评测 v2 的离线测试：不联网，用替身客户端。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_rag_v2 as ev

SET = Path(__file__).resolve().parents[1] / "rag/eval/rag_eval_set_v2.json"


def cases():
    return ev.load_cases(SET)


def case(cid):
    return next(c for c in cases() if c["id"] == cid)


def test_eval_set_covers_semantic_difficulties_and_gold_exists():
    items = cases()
    assert len(items) == 62
    tags = {t for c in items for t in c["tags"]}
    for t in ("同义改写", "表格数值", "多段汇总", "部分答案", "歧义需澄清", "诱导编造", "宽泛问题", "版本冲突", "无答案"):
        assert t in tags
    docs = {p.name for p in (SET.parents[1] / "kb_docs_md").glob("*.md")}
    norm = lambda s: "".join(s.split())
    texts = {p.name: norm(p.read_text(encoding="utf-8")) for p in (SET.parents[1] / "kb_docs_md").glob("*.md")}
    for c in items:
        for g in c["gold"]:
            assert g["file"] in docs
            assert any(norm(p) in texts[g["file"]] for p in g["any"]), c["id"]
        if c["expected_status"] in ("answered", "partial") and c["id"] not in ("R08",):
            assert c["gold"], c["id"]


def test_plain_answer_without_structured_block_is_not_penalised():
    checks = ev.evaluate_answer(case("V03"), "婚假 10 天（含法定婚假）。出处：《员工考勤与休假制度（v2）》第十条")
    decided = [c for c in checks if c["passed"] is not None]
    assert all(c["passed"] for c in decided)
    assert any(c["kind"] == "结构化" and c["passed"] is None for c in checks)


def test_deprecated_version_as_basis_fails():
    checks = ev.evaluate_answer(case("V03"), "婚假 10 天，依据《员工考勤与休假制度（v1）》")
    assert any(c["description"].startswith("不以已废止") and c["passed"] is False for c in checks)
    ok = ev.evaluate_answer(case("V03"), "婚假 10 天。注意《员工考勤与休假制度（v1）》已废止，其中的 3 天不再适用。")
    assert all(c["passed"] is not False for c in ok if c["description"].startswith("不以已废止"))


def test_structured_block_checked_for_status_and_sources():
    good = ('婚假 10 天。\n```json\n{"status": "answered", "sources": [{"file": "02_员工考勤与休假制度_v2", '
            '"section": "第十条"}]}\n```')
    assert all(c["passed"] is not False for c in ev.evaluate_answer(case("V03"), good))
    bad = ('婚假 3 天。\n```json\n{"status": "not_found", "sources": [{"file": "03_员工考勤与休假制度_v1_已废止"}]}\n```')
    failed = {c["description"] for c in ev.evaluate_answer(case("V03"), bad) if c["passed"] is False}
    assert "出处包含正确文档" in failed and "出处不含已废止文档" in failed
    assert any(d.startswith("状态应为 answered") for d in failed)


def test_partial_accepts_answered_and_clarify_requires_clarify():
    partial = '10 天；文档未规定工资。\n```json\n{"status": "answered", "sources": [{"file": "02_员工考勤与休假制度_v2"}]}\n```'
    assert all(c["passed"] is not False for c in ev.evaluate_answer(case("H01"), partial))
    clar = '请问您指的是哪一项的上限？\n```json\n{"status": "answered", "sources": []}\n```'
    assert any(c["passed"] is False for c in ev.evaluate_answer(case("C02"), clar))


def test_max_chars_catches_verbose_answer():
    long = "年假、病假、婚假、丧假。" + "补充说明" * 200
    assert any(c["kind"] == "冗余" and c["passed"] is False for c in ev.evaluate_answer(case("L01"), long))


def test_run_and_report_offline(tmp_path):
    class Fake:
        def stream_run(self, q):
            if "股票" in q:
                raise RuntimeError("超时")
            return SimpleNamespace(answer="知识库中未找到相关内容。", session_id="s")
    picked = ev.select_cases(cases(), "R01,R02,V03")
    results = ev.run_cases(picked, Fake(), sleep_fn=lambda _: None)
    assert {r["case"]["id"]: r["passed"] for r in results} == {"R01": False, "R02": True, "V03": False}
    report = ev.render_report(picked, results, "baseline", token="secret-token")
    assert "按语义难点" in report and "失败题" in report and "secret-token" not in report


def test_offline_preview_never_calls(tmp_path, capsys):
    assert ev.main(["--output-dir", str(tmp_path), "--only", "S01"]) == 0
    assert "离线预览" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["--only=ZZZ", "--limit=0"])
def test_bad_selection(tmp_path, bad):
    assert ev.main(["--output-dir", str(tmp_path), bad]) == 2


def test_global_checks_from_manual_review():
    s12 = ev.evaluate_answer(case("S12"), "P1 首次响应 15 分钟。\n**说明**：旧版制度已废止。")
    assert any(c["kind"] == "幻觉" and c["passed"] is False for c in s12)
    t04 = ev.evaluate_answer(case("T04"), "不超过 10%。\n**依据**：《绩效考核制度》第四条\n**说明**：无")
    assert any(c["kind"] == "冗余" and c["passed"] is False for c in t04)
    v03 = ev.evaluate_answer(case("V03"), "婚假 10 天。\n**说明**：《员工考勤与休假制度（v1）》已废止。")
    assert all(c["passed"] is not False for c in v03)
    f02 = ev.evaluate_answer(case("F02"), "系数为 2.0，即年终奖为2个月工资。")
    assert any(c["passed"] is False for c in f02)


def test_sources_may_use_document_titles():
    by_title = ('婚假 10 天。\n```json\n{"status": "answered", "sources": [{"file": "员工考勤与休假制度（v2）", '
                '"section": "第三章 其他假期"}]}\n```')
    assert all(c["passed"] is not False for c in ev.evaluate_answer(case("V03"), by_title))
    old = ('婚假 3 天。\n```json\n{"status": "answered", "sources": [{"file": "员工考勤与休假制度（v1）【已废止】"}]}\n```')
    failed = {c["description"] for c in ev.evaluate_answer(case("V03"), old) if c["passed"] is False}
    assert "出处不含已废止文档" in failed and "出处包含正确文档" in failed


def test_report_records_answer_version(tmp_path):
    class Fake:
        def __init__(self):
            self.answers = iter([
                "**答案**：10 天（《员工考勤与休假制度（v2）》）\n\n```json\n"
                '{"status": "answered", "sources": [], "version": "w4.1"}\n```',
                "知识库中未找到相关信息。",
            ])

        def stream_run(self, q):
            return type("R", (), {"answer": next(self.answers), "session_id": "s"})()

    cases = [c for c in ev.load_cases(ev.DEFAULT_EVAL_SET) if c["id"] in ("V03", "R01")]
    results = ev.run_cases(cases, Fake(), sleep_fn=lambda s: None)
    assert [r["version"] for r in results] == ["w4.1", ""]
    report = ev.render_report(cases, results, "t")
    assert "作答版本：w4.1 × 1；未标版本 × 1" in report
    assert "· w4.1" in report


def test_long_answer_eval_set_is_valid():
    path = ev.ROOT / "rag" / "eval" / "rag_eval_set_long.json"
    cases = ev.load_cases(path)
    assert len(cases) == 4 and all(c.get("max_chars") and c.get("gold") for c in cases)
    assert all("长回答" in c["tags"] for c in cases)
