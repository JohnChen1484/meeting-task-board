#!/usr/bin/env python3
"""智能办公助手 V5 —— 多轮对话自动走查脚本（扣子编程 /stream_run 版本）。

用法（先在项目根目录加载 .env：set -a && source .env && set +a）：
    python scripts/eval_multiturn.py --cases tests/multiturn_cases.json --probe
    python scripts/eval_multiturn.py --cases tests/multiturn_cases.json --warmup
    python scripts/eval_multiturn.py --cases tests/multiturn_cases.json --only M1,T1

需要的环境变量（写在本机 .env，不要写进代码或命令行）：
    OFFICE_AGENT_URL    部署域名，如 https://xxxx.coze.site
    OFFICE_AGENT_TOKEN  部署页“执行智能体”Header 中的 API Token
    OFFICE_PROJECT_ID   扣子编程项目 ID

设计要点：
- 同一用例内所有轮次共用一个 session_id（测上下文记忆）；不同用例之间换新 session_id（测隔离）。
- 关键词检查只是初筛；报告里保留每轮完整回答，供人工复核。
- 报告文件名精确到秒，避免同一分钟重跑时报 File exists。
- 只用 Python 标准库，不需要额外安装依赖。
- SSE 解析兼容 content.answer / data.content.answer 两种路径；
  如果与现有 probe 脚本的解析不一致，以 probe 为准替换 parse_answer_piece()。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

ENV_URL = "OFFICE_AGENT_URL"
ENV_TOKEN = "OFFICE_AGENT_TOKEN"
ENV_PROJECT = "OFFICE_PROJECT_ID"


class AgentError(Exception):
    """调用智能体失败（网络、鉴权、超时、返回为空）。"""


def load_config() -> dict:
    missing = [k for k in (ENV_URL, ENV_TOKEN, ENV_PROJECT) if not os.environ.get(k)]
    if missing:
        sys.exit(f"缺少环境变量：{', '.join(missing)}。请先执行 set -a && source .env && set +a")
    project_raw = os.environ[ENV_PROJECT].strip()
    project_id: int | str = int(project_raw) if project_raw.isdigit() else project_raw
    return {
        "url": os.environ[ENV_URL].rstrip("/") + "/stream_run",
        "token": os.environ[ENV_TOKEN].strip(),
        "project_id": project_id,
    }


def parse_answer_piece(event: dict) -> str:
    """从一条 SSE data 事件中取出增量回答文本；不是回答事件时返回空串。"""
    for container in (event, event.get("data") if isinstance(event.get("data"), dict) else None):
        if not container:
            continue
        content = container.get("content")
        if isinstance(content, dict) and isinstance(content.get("answer"), str):
            return content["answer"]
    return ""


def ask(cfg: dict, text: str, session_id: str, timeout: float) -> str:
    body = {
        "content": {"query": {"prompt": [{"type": "text", "content": {"text": text}}]}},
        "type": "query",
        "session_id": session_id,
        "project_id": cfg["project_id"],
    }
    req = urllib.request.Request(
        cfg["url"],
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg['token']}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    parts: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    parts.append(parse_answer_piece(event))
    except urllib.error.HTTPError as e:  # 不打印请求头，避免泄露令牌
        raise AgentError(f"HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AgentError(f"网络错误或超时：{type(e).__name__}") from None
    answer = "".join(parts).strip()
    if not answer:
        raise AgentError("收到响应但没有解析出回答文本（检查 SSE 字段路径）")
    return answer


def fill(text: str, materials: dict) -> str:
    for key, value in materials.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def check_turn(answer: str, turn: dict) -> list[str]:
    """返回未通过的检查项；空列表表示关键词初筛通过。"""
    problems = []
    for kw in turn.get("must_include", []):
        if kw not in answer:
            problems.append(f"缺少「{kw}」")
    any_of = turn.get("any_of", [])
    if any_of and not any(kw in answer for kw in any_of):
        problems.append("以下至少应出现一个：" + " / ".join(any_of))
    for kw in turn.get("must_not_include", []):
        if kw in answer:
            problems.append(f"不应出现「{kw}」")
    return problems


def run_case(cfg: dict, case: dict, materials: dict, timeout: float) -> dict:
    missing = [m for m in case.get("requires", []) if not materials.get(m)]
    if missing:
        return {"id": case["id"], "status": "SKIP", "reason": f"素材 {','.join(missing)} 未填写", "turns": []}
    session_id = str(uuid.uuid4())
    turns_out, status = [], "PASS"
    for idx, turn in enumerate(case["turns"], start=1):
        prompt = fill(turn["input"], materials)
        started = time.monotonic()
        try:
            answer = ask(cfg, prompt, session_id, timeout)
            problems = check_turn(answer, turn)
        except AgentError as e:
            answer, problems = "", [f"调用失败：{e}"]
            status = "ERROR"
        elapsed = time.monotonic() - started
        if problems and status != "ERROR":
            status = "CHECK"  # 关键词未通过，需人工复核
        turns_out.append({"turn": idx, "input": prompt, "answer": answer,
                          "problems": problems, "seconds": round(elapsed, 1)})
        if status == "ERROR":
            break  # 后续轮次依赖前文，失败后不再继续
    return {"id": case["id"], "dimension": case.get("dimension", ""), "expect": case.get("expect", ""),
            "status": status, "session_id": session_id, "turns": turns_out}


def probe(cfg: dict, timeout: float) -> bool:
    sid = str(uuid.uuid4())
    print("记忆探测：同一 session_id 发送 2 轮……")
    # 两轮都在“文本优化”范围内，避免被人设当作越界请求拒答，造成误判
    ask(cfg, "帮我润色：蓝鲸项目下周上线，大家抓紧把测试做完。", sid, timeout)
    second = ask(cfg, "把刚才润色的结果压缩到 20 字以内。", sid, timeout)
    ok = "蓝鲸" in second
    print(f"第 2 轮回答：{second[:120]}")
    print("结果：通过，服务端按 session_id 保留了上下文" if ok
          else "结果：未通过，第 2 轮不记得第 1 轮。检查 session_id 是否一致，或让开发助手加会话记忆")
    return ok


def write_report(results: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, n = stamp, 1
    while (out_dir / f"multiturn_{stamp}.md").exists():  # 同一秒内重跑也不覆盖、不报错
        n += 1
        stamp = f"{base}_{n}"
    md_path = out_dir / f"multiturn_{stamp}.md"
    json_path = out_dir / f"multiturn_{stamp}.json"
    counts = {s: sum(r["status"] == s for r in results) for s in ("PASS", "CHECK", "ERROR", "SKIP")}
    lines = [
        f"# 多轮走查报告 {stamp}",
        "",
        f"初筛通过 {counts['PASS']} / 需人工复核 {counts['CHECK']} / 调用失败 {counts['ERROR']} / 跳过 {counts['SKIP']}",
        "",
        "关键词只是初筛：PASS 也要抽查，CHECK 必须人工判定。",
        "",
        "| 编号 | 维度 | 结果 | 未通过的检查 |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        issues = "；".join(f"第{t['turn']}轮 " + "，".join(t["problems"]) for t in r["turns"] if t["problems"])
        if r["status"] == "SKIP":
            issues = r.get("reason", "")
        lines.append(f"| {r['id']} | {r.get('dimension', '')} | {r['status']} | {issues.replace('|', '/')} |")
    for r in results:
        if r["status"] == "SKIP":
            continue
        lines += ["", f"## {r['id']} {r.get('dimension', '')} —— {r['status']}", "", f"期望：{r.get('expect', '')}", ""]
        for t in r["turns"]:
            lines += [f"**第 {t['turn']} 轮输入**（{t['seconds']} 秒）", "", "```text", t["input"], "```", "",
                      "**回答**", "", "```text", t["answer"] or "（无）", "```", ""]
            if t["problems"]:
                lines += ["检查未通过：" + "；".join(t["problems"]), ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="智能办公助手多轮走查")
    parser.add_argument("--cases", default="tests/multiturn_cases.json")
    parser.add_argument("--only", default="", help="只跑指定用例，逗号分隔，如 M1,T1")
    parser.add_argument("--warmup", action="store_true", help="正式运行前先发 1 条预热请求")
    parser.add_argument("--probe", action="store_true", help="只做 2 轮记忆探测")
    parser.add_argument("--timeout", type=float, default=120.0, help="单轮超时秒数")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    cfg = load_config()
    if args.probe:
        try:
            sys.exit(0 if probe(cfg, args.timeout) else 1)
        except AgentError as e:
            sys.exit(f"探测失败：{e}")

    data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    materials = data.get("materials", {})
    cases = data["cases"]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        unknown = wanted - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"找不到用例：{', '.join(sorted(unknown))}")

    if args.warmup:
        print("预热中……")
        try:
            ask(cfg, "你好", str(uuid.uuid4()), args.timeout)
        except AgentError as e:
            print(f"预热失败（{e}），继续运行；如首题超时请稍后重跑")

    results = []
    for case in cases:
        result = run_case(cfg, case, materials, args.timeout)
        results.append(result)
        print(f"{result['id']:<4} {result.get('dimension', ''):<10} {result['status']}")

    md_path, json_path = write_report(results, Path(args.out_dir))
    print(f"\n报告：{md_path}\n原始数据：{json_path}")


if __name__ == "__main__":
    main()
