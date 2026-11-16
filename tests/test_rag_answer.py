"""知识库结构化回答的纯离线解析与清洗测试。"""
import json

import pytest

from app.rag_answer import split_structured, to_response
from scripts import eval_rag_v2


def answer_with(data, body="婚假 **10 天**。"):
    return body + "\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```"


def test_normal_answer_and_shared_evaluator_parser():
    data = {"status": "answered", "sources": [{"file": "员工制度", "section": "第十条",
             "status": "现行", "score": 0.61}], "rewritten_query": "婚假有几天？", "intent": "knowledge"}
    answer = answer_with(data)
    assert eval_rag_v2.split_structured is split_structured
    assert split_structured(answer) == ("婚假 **10 天**。", data)
    assert to_response(answer) == {"answer": "婚假 **10 天**。", "structured": True, **data}


@pytest.mark.parametrize("answer", ["普通回答", "", "正文\n```json\n{bad json}\n```",
                                    "正文\n```json\n[]\n```", "正文\n```json\nnull\n```",
                                    "正文\n```json\n123\n```"])
def test_missing_invalid_or_non_object_json_preserves_text(answer):
    assert to_response(answer) == {"answer": answer, "structured": False, "status": "unknown",
                                   "sources": [], "rewritten_query": "", "intent": ""}


@pytest.mark.parametrize("answer", [None, 123, [], {}, False])
def test_non_string_answer_never_raises(answer):
    assert to_response(answer) == {"answer": "", "structured": False, "status": "unknown",
                                   "sources": [], "rewritten_query": "", "intent": ""}


@pytest.mark.parametrize("status", ["answered", "partial", "not_found", "clarify", "unverified",
                                    "chitchat", "refused", "error"])
def test_all_allowed_statuses(status):
    result = to_response(answer_with({"status": status}))
    assert result["structured"] is True and result["status"] == status


@pytest.mark.parametrize("status", ["unknown", "made_up", "ANSWERED", "", None, 1, [], {}])
def test_invalid_status_does_not_discard_valid_object(status):
    result = to_response(answer_with({"status": status}))
    assert result["structured"] is True and result["status"] == "unknown"


def test_empty_object_and_only_json_body():
    assert to_response(answer_with({}, body="")) == {
        "answer": "", "structured": True, "status": "unknown", "sources": [],
        "rewritten_query": "", "intent": ""}


def test_dirty_sources_filtered_and_extra_keys_removed():
    sources = [None, "文档", 1, [], {}, {"file": None}, {"file": 3}, {"file": ""}, {"file": " \t"},
               {"file": "有效", "section": [], "status": 2, "score": "0.8", "extra": "丢弃"},
               {"file": "缺省"}]
    result = to_response(answer_with({"sources": sources, "rewritten_query": [], "intent": 2,
                                     "extra": "丢弃"}))
    assert result["sources"] == [{"file": "有效", "section": "", "status": "", "score": None},
                                  {"file": "缺省", "section": "", "status": "", "score": None}]
    assert result["rewritten_query"] == result["intent"] == ""
    assert set(result) == {"answer", "structured", "status", "sources", "rewritten_query", "intent"}


@pytest.mark.parametrize("sources", [None, {}, "文档", 1, True])
def test_non_list_sources_are_empty(sources):
    assert to_response(answer_with({"sources": sources}))["sources"] == []


@pytest.mark.parametrize("score", ["0.61", None, True, False, [], {}, float("nan"),
                                  float("inf"), float("-inf")])
def test_non_numeric_or_non_finite_scores_become_null(score):
    result = to_response(answer_with({"sources": [{"file": "文档", "score": score}]}))
    assert result["sources"][0]["score"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("score", [0, 1, -2, 0.61, -0.5, 10 ** 400])
def test_finite_numbers_retained(score):
    result = to_response(answer_with({"sources": [{"file": "文档", "score": score}]}))
    assert result["sources"][0]["score"] == score


def test_keep_first_ten_valid_sources():
    sources = [{}] * 12 + [{"file": str(i)} for i in range(12)]
    assert [s["file"] for s in to_response(answer_with({"sources": sources}))["sources"]] == [
        str(i) for i in range(10)]


def test_only_last_json_block_is_metadata():
    earlier = '示例：\n```json\n{"example": "正文中的 JSON"}\n```'
    answer = answer_with({"status": "partial"}, body=earlier + "\n实际回答")
    result = to_response(answer)
    assert result["answer"] == earlier + "\n实际回答"
    assert result["status"] == "partial" and result["structured"] is True


def test_broken_last_block_does_not_fall_back_to_earlier_metadata():
    answer = answer_with({"status": "answered"}) + '\n```json\n{"status": bad}\n```'
    result = to_response(answer)
    assert result["answer"] == answer
    assert result["structured"] is False and result["status"] == "unknown"
