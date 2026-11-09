"""Coze 任务提取评测：默认仅生成待执行报告，--live 才允许联网。

不导入 app.main、不创建数据库、不加载 .env。仅在 --live 下读取 Token。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import getpass
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_CALLS = 10
FIELDS = {"task", "owner", "due_date", "priority"}
OPS = {"exists", "equals", "not_equals", "contains_all", "all_equals", "count_equals"}


@dataclass
class CheckResult:
    description: str
    passed: bool
    actual: object


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CALLS:
        raise ValueError("评测用例必须是数组，数量为 1～10；超过上限时不会发起任何调用")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), (int, str)):
            raise ValueError("每个用例必须有 id")
        if case["id"] in ids:
            raise ValueError("用例 id 不可重复")
        ids.add(case["id"])
        if not isinstance(case.get("name"), str) or not isinstance(case.get("meeting_text"), str) or not case["meeting_text"].strip():
            raise ValueError("用例必须有名称和非空会议记录")
        if not isinstance(case.get("checks"), list) or not case["checks"]:
            raise ValueError("每个用例必须有检查项")
        for rule in case["checks"]:
            if not isinstance(rule, dict) or rule.get("op") not in OPS or not isinstance(rule.get("description"), str):
                raise ValueError("检查项操作或说明不合法")
            op = rule["op"]
            if op == "count_equals":
                if type(rule.get("value")) is not int or rule["value"] < 0:
                    raise ValueError("任务数量必须是非负整数")
            elif op != "all_equals":
                words = rule.get("keywords")
                if not isinstance(words, list) or not words or not all(isinstance(w, str) and w for w in words):
                    raise ValueError("任务选择器需要非空 keywords 数组")
            if op not in ("exists", "count_equals"):
                if rule.get("field") not in FIELDS:
                    raise ValueError("检查项字段不合法")
                value = rule.get("value")
                if op == "contains_all":
                    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
                        raise ValueError("contains_all 的 value 必须是非空文本数组")
                elif not isinstance(value, str):
                    raise ValueError("字段比较值必须是文本")
    return cases


def evaluate_check(tasks, rule):
    """关键词为“或”；多个匹配任务必须全部满足，防止重复任务掩盖错误。"""
    op = rule["op"]
    if op == "count_equals":
        return CheckResult(rule["description"], len(tasks) == rule["value"], len(tasks))
    selected = tasks if op == "all_equals" else [
        t for t in tasks if any(k in t.get("task", "") for k in rule["keywords"])
    ]
    if op == "exists":
        return CheckResult(rule["description"], bool(selected), [t.get("task") for t in selected])
    actual = [t.get(rule["field"]) for t in selected]
    expected = rule["value"]
    if op == "not_equals":
        passed = bool(actual) and all(isinstance(v, str) and v != expected for v in actual)
    elif op == "contains_all":
        passed = bool(actual) and all(isinstance(v, str) and all(word in v for word in expected) for v in actual)
    else:
        passed = bool(actual) and all(v == expected for v in actual)
    return CheckResult(rule["description"], passed, actual)


def run_cases(cases, client=None, sleeper=None):
    validate_cases(cases)  # 调用前先校验整个数据集，避免中途才发现超限。
    sleeper = sleeper or time.sleep
    results = []
    for index, case in enumerate(cases):
        result = {"case": case, "status": "未执行", "tasks": [], "warnings": [],
                  "chat_id": None, "conversation_id": None, "session_id": None, "checks": [], "error": None}
        if client is not None:
            if index:
                sleeper(2)  # 仅两次实际调用之间等待；失败后也等待，不自动重试。
            try:
                extracted = client.extract(case["meeting_text"])
                result.update(status="已执行", tasks=extracted.tasks, warnings=extracted.warnings,
                              chat_id=extracted.chat_id, conversation_id=extracted.conversation_id,
                              session_id=getattr(extracted, "session_id", None))
                result["checks"] = [evaluate_check(extracted.tasks, rule) for rule in case["checks"]]
            except Exception as exc:
                # 错误脱敏在最终写盘前统一执行，不输出 traceback/凭据。
                result.update(status="调用失败", error=str(exc),
                              chat_id=getattr(exc, "chat_id", None),
                              conversation_id=getattr(exc, "conversation_id", None),
                              session_id=getattr(exc, "session_id", None))
                result["checks"] = [CheckResult(rule["description"], False, "调用失败，无法判定") for rule in case["checks"]]
        results.append(result)
    return results


def md(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "&#124;").replace("\r\n", "\n").replace("\n", "<br>")


def render_report(results, mode, token=""):
    lines = ["# Coze 任务提取评测", "", f"运行模式：{mode}", "",
             "离线结果不证明真实模型效果；关键词检查通过也不等于完整语义准确。", ""]
    passed = total = good_cases = executed = 0
    for result in results:
        case = result["case"]
        lines += [f"## {md(case['id'])}. {md(case['name'])}", "", f"状态：{result['status']}", "",
                  "输入：", "", md(case["meeting_text"]), "",
                  f"chat_id：{md(result['chat_id'] or '未获得')}",
                  f"conversation_id：{md(result['conversation_id'] or '未获得')}",
                  f"session_id：{md(result.get('session_id') or '未获得')}", ""]
        if result["error"]:
            error = result["error"].replace(token, "[已脱敏]") if token else result["error"]
            lines += [f"错误：{md(error[:300])}", ""]
        if result["warnings"]:
            lines += ["字段警告：" + md("；".join(result["warnings"])), ""]
        lines += ["| 任务 | 负责人 | 截止日期 | 日期原文 | 优先级 | 优先级依据 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for task in result["tasks"]:
            lines.append("| " + " | ".join(md(task.get(k, "未明确")) for k in ("task", "owner", "due_date", "due_date_raw", "priority", "priority_basis")) + " |")
        if not result["tasks"]:
            lines.append("| （未执行/调用失败时无结果；成功时为空任务） | — | — | — | — | — |")
        lines += ["", "| 检查项 | 结果 | 实际值 |", "| --- | --- | --- |"]
        if result["checks"]:
            executed += 1
            good_cases += int(all(c.passed for c in result["checks"]))
            for check in result["checks"]:
                total += 1
                passed += int(check.passed)
                lines.append(f"| {md(check.description)} | {'通过' if check.passed else '失败'} | {md(json.dumps(check.actual, ensure_ascii=False))} |")
        else:
            for rule in case["checks"]:
                lines.append(f"| {md(rule['description'])} | 未执行 | — |")
        lines.append("")
    lines += ["## 汇总", ""]
    if total:
        lines += [f"检查项通过率：{passed}/{total}（{passed / total:.1%}）",
                  f"用例通过率：{good_cases}/{executed}（{good_cases / executed:.1%}）"]
    else:
        lines.append("未发起调用，全部检查项未执行，不计算通过率。")
    report = "\n".join(lines) + "\n"
    # 先整体脱敏，再 Markdown 转义，避免转义改变 Token 导致漏替换。
    if token:
        report = report.replace(token, "[已脱敏]").replace(md(token), "[已脱敏]")
        # JSON 转义后的 Token 也可能出现在实际值展示中。
        report = report.replace(md(json.dumps(token, ensure_ascii=False)[1:-1]), "[已脱敏]")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="显式允许真实 Coze 调用（可能产生费用）")
    parser.add_argument("--provider", choices=("coze", "coze_deploy"), help="默认读取进程 AI_PROVIDER；未设置时为 coze")
    parser.add_argument("--cases", type=Path, default=ROOT / "tests/cases/eval_cases.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)
    token = ""
    client = None
    report_file = None
    try:
        cases = validate_cases(json.loads(args.cases.read_text(encoding="utf-8")))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = "eval_" + datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M") + ".md"
        destination = args.output_dir / name
        # 发起调用前预留报告文件；同一分钟报告不覆盖，也不会因此白白调用 AI。
        report_file = destination.open("x", encoding="utf-8")
        if args.live:
            # 始终不加载 .env；真实评测禁止使用 Mock 冒充成功。
            os.environ["PYTHON_DOTENV_DISABLED"] = "1"
            from app.ai_client import build_client
            from app.config import Settings, validate_deploy_url
            provider = args.provider or os.environ.get("AI_PROVIDER", "coze").strip().lower()
            if provider not in {"coze", "coze_deploy"}:
                raise ValueError("真实评测仅支持 coze / coze_deploy，不允许 Mock")
            settings = Settings.from_env()
            settings.ai_provider = provider
            timeout = settings.coze_timeout_seconds
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("COZE_TIMEOUT_SECONDS 必须是正的有限秒数")
            if provider == "coze":
                parsed = urlsplit(settings.coze_api_base)
                if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                        or parsed.password is not None or parsed.query or parsed.fragment):
                    raise ValueError("API 地址必须是无账号、查询参数和片段的 HTTPS 基地址")
                if not settings.coze_bot_id:
                    raise ValueError("请通过环境变量设置 COZE_BOT_ID")
            else:
                settings.coze_deploy_url = validate_deploy_url(settings.coze_deploy_url)
                if settings.coze_project_id is None:
                    raise ValueError("请通过环境变量设置 COZE_PROJECT_ID")
            token = os.environ.get("COZE_API_TOKEN", "").strip()
            if not token:
                # 无可用终端时禁止 getpass 回退到回显输入。
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    token = getpass.getpass("Coze Token（隐藏输入，不保存）：").strip()
            if not token:
                raise ValueError("Token 不能为空")
            settings.coze_api_token = token
            client = build_client(settings)
        results = run_cases(cases, client=client)
        mode = f"真实 {provider}（--live）" if args.live else "离线预检（未调用 AI）"
        report = render_report(results, mode, token)
        report_file.write(report)
        report_file.flush()
        display_path = str(destination).replace(token, "[已脱敏]") if token else str(destination)
        print("报告已生成：" + display_path)
        return int(any(c.passed is False for r in results for c in r["checks"]))
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc).replace(token, "[已脱敏]") if token else str(exc)
        print("评测未完成：" + message[:300], file=sys.stderr)
        return 2
    finally:
        if report_file is not None:
            report_file.close()
        if client is not None:
            client._http.close()


if __name__ == "__main__":
    raise SystemExit(main())
