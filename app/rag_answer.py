"""拆分知识库回答，并整理供页面展示的结构化字段。"""
from __future__ import annotations

import json
import math
import re

JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
STATUSES = frozenset({"answered", "partial", "not_found", "clarify", "unverified",
                      "chitchat", "refused", "error"})


def split_structured(answer: str):
    """拆出回答正文与末尾的结构化 JSON（没有或解析失败时返回 None）。"""
    matches = list(JSON_BLOCK.finditer(answer or ""))
    if not matches:
        return answer or "", None
    last = matches[-1]
    try:
        data = json.loads(last.group(1))
    except json.JSONDecodeError:
        return answer, None
    return (answer[:last.start()] + answer[last.end():]).strip(), data if isinstance(data, dict) else None


def to_response(answer):
    """容忍缺失或脏数据，仅向页面暴露允许展示的字段。"""
    text = answer if isinstance(answer, str) else ""
    try:
        body, data = split_structured(text)
    except (TypeError, ValueError, RecursionError, OverflowError):
        body, data = text, None
    result = {"answer": body, "structured": data is not None, "status": "unknown",
              "sources": [], "rewritten_query": "", "intent": ""}
    if data is None:
        return result

    status = data.get("status")
    if isinstance(status, str) and status in STATUSES:
        result["status"] = status
    for key in ("rewritten_query", "intent"):
        value = data.get(key)
        if isinstance(value, str):
            result[key] = value

    sources = data.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            file = source.get("file")
            if not isinstance(file, str) or not file.strip():
                continue
            score = source.get("score")
            valid_score = (type(score) is int or
                           (type(score) is float and math.isfinite(score)))
            result["sources"].append({
                "file": file,
                "section": source.get("section") if isinstance(source.get("section"), str) else "",
                "status": source.get("status") if isinstance(source.get("status"), str) else "",
                "score": score if valid_score else None,
            })
            if len(result["sources"]) == 10:
                break
    return result
