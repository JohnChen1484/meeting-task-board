"""RAG 问答评测 v2（第三周）：默认离线预览，只有 --live 才调用部署的 /stream_run。

相比第一版：
- 评测集按语义难点打标签（tags），报告按难点、预期状态分别统计；
- 检查项增加：回答长度上限（max_chars，抓冗余）、不得以已废止文档为依据；
- 如果回答末尾带 ```json {"status": ..., "sources": [...]}```（工作流版的结构化输出），
  另外检查状态是否符合预期、出处是否包含正确文档、是否引用了已废止文档。
  旧的单智能体版没有这个块，这几项记为“不适用”，不计入失败。

令牌只从进程环境变量 RAG_API_TOKEN 读取（或隐藏输入），从不加载 .env 文件。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import getpass
import html
import json
import os
from pathlib import Path
import re
import sys
import time
import warnings
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SET = ROOT / "rag" / "eval" / "rag_eval_set_v2.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag_answer import split_structured

DEPRECATED_TITLE = "员工考勤与休假制度（v1）"
STATUS_OK = {"answered": {"answered"}, "partial": {"partial", "answered"},
             "not_found": {"not_found"}, "clarify": {"clarify"}}
KB_DOCS = ROOT / "rag" / "kb_docs_md"


def _title_map():
    """文档标题 → 文件名（不含扩展名）。工作流的 sources.file 可能写标题，也可能写文件名。"""
    mapping = {}
    if KB_DOCS.is_dir():
        for path in KB_DOCS.glob("*.md"):
            first = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), "")
            mapping[normalize(re.sub(r"^#+\s*", "", first))] = path.stem
    return mapping


_TITLES = None


def source_stem(value: str) -> str:
    key = normalize(str(value or "")).strip("《》")
    key = re.sub(r"\.(md|pdf|docx)$", "", key)
    global _TITLES
    if _TITLES is None:
        _TITLES = _title_map()
    return _TITLES.get(key, key)


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("评测集必须是含非空 items 数组的 JSON 对象")
    seen = set()
    for case in items:
        for key in ("id", "category", "question", "expected_status"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"{case.get('id')}: {key} 必须是非空文本")
        if case["id"] in seen:
            raise ValueError(f"重复题号：{case['id']}")
        seen.add(case["id"])
        if case["expected_status"] not in STATUS_OK:
            raise ValueError(f"{case['id']}: expected_status 不合法")
        groups = case.get("must_include")
        if not isinstance(groups, list) or not groups or any(not isinstance(g, list) or not g for g in groups):
            raise ValueError(f"{case['id']}: must_include 必须是非空关键词分组")
        if not isinstance(case.get("tags"), list) or not case["tags"]:
            raise ValueError(f"{case['id']}: tags 必须是非空数组")
    return items


def select_cases(cases, only=None, limit=None):
    if only:
        ids = {v.strip() for v in only.split(",") if v.strip()}
        unknown = ids - {c["id"] for c in cases}
        if unknown:
            raise ValueError("未知题号：" + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["id"] in ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit 必须大于 0")
        cases = cases[:limit]
    return cases


def evaluate_answer(case: dict, answer: str) -> list[dict]:
    body, data = split_structured(answer)
    text = normalize(body)
    checks = [{"description": "非空回答", "passed": bool(text), "kind": "内容"}]
    groups = case["must_include"]
    if case.get("should_refuse"):
        groups = [[t for g in groups for t in g]]
    for i, group in enumerate(groups, 1):
        hits = [t for t in group if normalize(t) in text]
        label = "拒答措辞（任一命中）" if case.get("should_refuse") else f"必含第 {i} 组"
        checks.append({"description": f"{label}：{' / '.join(group)}", "passed": bool(hits), "hits": hits, "kind": "内容"})
    for term in case.get("must_not_include", []):
        hit = normalize(term) in text
        checks.append({"description": f"不得出现：{term}", "passed": not hit, "hits": [term] if hit else [],
                       "kind": "内容"})
    # 全局检查（基线人工复核发现、关键词检查漏掉的问题）：
    # 1) 空栏目：“说明：无”之类应整行省略；
    if re.search(r"(说明|备注)\**[:：]\s*\**\s*无\s*($|\n)", body):
        checks.append({"description": "没有内容的栏目应省略（出现“说明：无”）", "passed": False, "kind": "冗余"})
    # 2) 编造版本信息：题目与已废止 v1 无关，回答却声称“旧版已废止”。
    about_leave = (any(g["file"].startswith("02_") for g in case.get("gold", []))
                   or any(w in case["question"] + body for w in ("请假", "年假", "考勤", "休假")))
    if not case.get("must_not_cite") and not about_leave and "废止" in body:
        checks.append({"description": "与旧版本无关的问题，回答却提到“已废止”", "passed": False, "kind": "幻觉"})
    if case.get("must_not_cite"):
        cited = DEPRECATED_TITLE in body and "废止" not in body
        checks.append({"description": "不以已废止的 v1 制度为依据", "passed": not cited, "kind": "幻觉"})
    if case.get("max_chars"):
        length = len(text)
        checks.append({"description": f"正文不超过 {case['max_chars']} 字（实际 {length}）",
                       "passed": length <= case["max_chars"], "kind": "冗余"})
    if data is None:
        checks.append({"description": "结构化输出（状态 / 出处）", "passed": None, "kind": "结构化"})
        return checks
    status = data.get("status")
    checks.append({"description": f"状态应为 {case['expected_status']}（实际 {status}）",
                   "passed": status in STATUS_OK[case["expected_status"]], "kind": "结构化"})
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    files = {source_stem(s.get("file") or s.get("doc") or "") for s in sources if isinstance(s, dict)}
    gold_files = {g["file"] for g in case.get("gold", [])}
    if gold_files and case["expected_status"] in ("answered", "partial"):
        stems = {Path(f).stem for f in gold_files}
        ok = bool(stems & files)
        checks.append({"description": "出处包含正确文档", "passed": ok, "kind": "结构化"})
    if case.get("must_not_cite"):
        bad = any("已废止" in f for f in files)
        checks.append({"description": "出处不含已废止文档", "passed": not bad, "kind": "结构化"})
    return checks


def run_cases(cases, client, sleep_fn=None):
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    results = []
    for index, case in enumerate(cases):
        if index:
            sleep_fn(2)
        started = time.monotonic()
        answer, session_id, error = "", "", None
        try:
            run = client.stream_run(case["question"])
            answer, session_id = run.answer, run.session_id
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            answer = getattr(exc, "partial_answer", "")
            session_id = getattr(exc, "session_id", "")
        checks = [{"description": "调用成功", "passed": error is None, "kind": "检索失败"}]
        checks.extend(evaluate_answer(case, answer))
        decided = [c for c in checks if c["passed"] is not None]
        results.append({"case": case, "answer": answer, "session_id": session_id, "error": error,
                        "elapsed": time.monotonic() - started, "checks": checks,
                        "length": len(normalize(split_structured(answer)[0])),
                        "passed": all(c["passed"] for c in decided)})
    return results


def render_report(cases, results, label, token=""):
    esc = lambda v: html.escape(str(v).replace(token, "[已脱敏]") if token else str(v)).replace("|", "&#124;")
    rate = lambda p, t: f"{p}/{t}（{p / t:.1%}）" if t else "—"
    lines = [f"# RAG 问答评测 v2：{esc(label)}", ""]
    if results is None:
        lines += [f"离线预览：未调用、未读取令牌。已选 {len(cases)} 题；加 --live 才联网。", ""]
        lines += [f"- {esc(c['id'])} [{esc('、'.join(c['tags']))}] {esc(c['question'])}" for c in cases]
        return "\n".join(lines) + "\n"
    passed = sum(r["passed"] for r in results)
    avg_time = sum(r["elapsed"] for r in results) / len(results)
    answered = [r for r in results if r["case"]["expected_status"] in ("answered", "partial")]
    avg_len = sum(r["length"] for r in answered) / len(answered) if answered else 0
    structured = sum(1 for r in results if any(c["kind"] == "结构化" and c["passed"] is not None for c in r["checks"]))
    lines += [f"总通过率：{rate(passed, len(results))}；平均耗时 {avg_time:.1f} 秒；有答案题平均正文长度 {avg_len:.0f} 字；"
              f"带结构化输出的回答 {structured}/{len(results)}。", ""]

    def table(title, key_fn):
        nonlocal lines
        total, ok = Counter(), Counter()
        for r in results:
            for k in key_fn(r["case"]):
                total[k] += 1
                ok[k] += r["passed"]
        lines += [f"## {title}", "", "| 分组 | 通过率 |", "| --- | --- |"]
        lines += [f"| {esc(k)} | {rate(ok[k], total[k])} |" for k in sorted(total)]
        lines.append("")

    table("按语义难点", lambda c: c["tags"])
    table("按预期状态", lambda c: [c["expected_status"]])
    kinds = defaultdict(lambda: [0, 0])
    for r in results:
        for c in r["checks"]:
            if c["passed"] is not None:
                kinds[c["kind"]][0] += c["passed"]
                kinds[c["kind"]][1] += 1
    lines += ["## 按问题类型（检查项通过率）", "", "| 类型 | 通过率 |", "| --- | --- |"]
    lines += [f"| {esc(k)} | {rate(v[0], v[1])} |" for k, v in sorted(kinds.items())]
    lines += ["", "## 失败题", ""]
    for r in results:
        if r["passed"]:
            continue
        c = r["case"]
        fails = "；".join(x["description"] for x in r["checks"] if x["passed"] is False)
        lines += [f"### {esc(c['id'])} [{esc('、'.join(c['tags']))}] {esc(c['question'])}", "",
                  f"失败项：{esc(fails)}", "", "<pre>" + esc(r["answer"]) + "</pre>", ""]
    lines += ["## 全部回答", ""]
    for r in results:
        c = r["case"]
        lines += [f"### {esc(c['id'])} · {'通过' if r['passed'] else '失败'} · {r['elapsed']:.1f} 秒 · {r['length']} 字",
                  "", f"问题：{esc(c['question'])}", "", "<pre>" + esc(r["answer"]) + "</pre>", ""]
        if r["error"]:
            lines += [f"调用错误：{esc(r['error'])}", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--only")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="baseline", help="报告标题和文件名里的标记，如 baseline / workflow")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)
    token, client = "", None
    try:
        cases = select_cases(load_cases(args.eval_set), args.only, args.limit)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M%S")
        path = args.output_dir / f"rag_v2_{re.sub(r'[^A-Za-z0-9_-]', '', args.label) or 'run'}_{stamp}.md"
        if not args.live:
            path.write_text(render_report(cases, None, args.label), encoding="utf-8")
            print("离线预览：" + str(path))
            return 0
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        from app.ai_client import CozeDeployClient
        from app.config import Settings, parse_project_id, validate_deploy_url
        url = validate_deploy_url(os.environ.get("RAG_DEPLOY_URL", ""))
        project = parse_project_id(os.environ.get("RAG_PROJECT_ID", ""))
        if project is None:
            raise ValueError("请设置 RAG_PROJECT_ID")
        token = os.environ.get("RAG_API_TOKEN", "").strip()
        if not token:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("RAG API Token（隐藏输入，不保存）：").strip()
        if not token:
            raise ValueError("RAG API Token 不能为空")
        client = CozeDeployClient(Settings(ai_provider="coze_deploy", coze_deploy_url=url, coze_project_id=project,
                                           coze_api_token=token, coze_timeout_seconds=120.0,
                                           coze_read_timeout_seconds=60.0, db_path=":memory:"))
        print(f"开始评测 {len(cases)} 题（每题间隔 2 秒）…", flush=True)
        results = run_cases(cases, client)
        path.write_text(render_report(cases, results, args.label, token), encoding="utf-8")
        passed = sum(r["passed"] for r in results)
        print(f"通过 {passed}/{len(results)}。报告：{path}")
        return 0 if passed == len(results) else 1
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        message = str(exc).replace(token, "[已脱敏]") if token else str(exc)
        print("RAG 评测未完成：" + message, file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client._http.close()


if __name__ == "__main__":
    raise SystemExit(main())
