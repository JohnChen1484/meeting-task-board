"""部署 SSE 格式探测：默认不联网，只有 --live 才发送一次固定请求。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import sys
from threading import Event, Lock
import warnings
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROBE_TEXT = "帮我整理一下这个会，并额外输出 JSON：会议日期：2026-10-29。紫杉项目周会：王一鸣负责在明天前提交蓝鲸接口文档。"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="显式允许一次真实部署服务请求")
    parser.add_argument("--url", help="部署基地址；优先于 COZE_DEPLOY_URL")
    parser.add_argument("--project-id", help="数字项目 ID；优先于 COZE_PROJECT_ID")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)
    if not args.live:
        print("离线模式：未发起调用、未读取令牌。由你确认后加 --live 才发送一次固定测试请求。")
        return 0

    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    from app.ai_client import AIError, CozeDeployClient, field_structure
    from app.config import Settings, parse_project_id, validate_deploy_url
    from app.errors import safe_error

    token = ""
    client = report = None
    status = None
    lines = []
    counts = Counter()
    structures = {}
    lock = Lock()
    accepting = Event()
    blocked = False
    run = None
    error = None
    try:
        url = validate_deploy_url(args.url if args.url is not None else os.environ.get("COZE_DEPLOY_URL", ""))
        project = parse_project_id(args.project_id if args.project_id is not None else os.environ.get("COZE_PROJECT_ID", ""))
        if project is None:
            raise ValueError("请设置 COZE_PROJECT_ID 或 --project-id")
        timeout = float(os.environ.get("COZE_TIMEOUT_SECONDS", "90"))
        token = os.environ.get("COZE_API_TOKEN", "").strip()
        if not token:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("部署 API Token（隐藏输入，不保存）：").strip()
        if not token:
            raise ValueError("部署 API Token 不能为空")
        settings = Settings(ai_provider="coze_deploy", coze_api_token=token, coze_deploy_url=url,
                            coze_project_id=project, coze_timeout_seconds=timeout)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / ("probe_raw_" + datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M") + ".txt")
        # 调用前排除文件覆盖/不可写，不让报告错误造成额外真实调用。
        report = path.open("x", encoding="utf-8")
        client = CozeDeployClient(settings)

        def line_received(line):
            nonlocal blocked
            with lock:
                if not accepting.is_set():
                    return
                if token in line:
                    blocked = True
                    raise AIError("检测到令牌，已阻止原始 SSE 写入")
                lines.append(line)

        def status_received(value):
            nonlocal status
            with lock:
                if accepting.is_set():
                    status = value

        def event_received(data):
            nonlocal blocked
            with lock:
                if not accepting.is_set():
                    return
                # 同时检查 JSON 解码后的内容，防止转义形式的令牌落盘。
                if token in json.dumps(data, ensure_ascii=False):
                    blocked = True
                    raise AIError("检测到令牌，已阻止原始 SSE 写入")
                kind = data["type"]
                counts[kind] += 1
                if kind not in structures:
                    structures[kind] = field_structure(data.get("content"))

        accepting.set()
        try:
            run = client.stream_run(PROBE_TEXT, on_line=line_received, on_status=status_received, on_event=event_received)
        except Exception as exc:
            error = exc
        finally:
            accepting.clear()
        with lock:
            raw = "\n".join(lines) + ("\n" if lines else "")
            answer = run.answer if run is not None else getattr(error, "partial_answer", "")
            unsafe = blocked or token in raw or token in answer
            if unsafe:
                report.write("检测到令牌，原始 SSE 未保存。\n")
                error = AIError("检测到令牌，原始 SSE 未保存")
            else:
                # 按收到的行顺序保存；行终止符统一为 LF。
                report.write(raw)
            report.flush()
            print("HTTP 状态码：" + (str(status) if status is not None else "未收到"))
            safe_counts = {safe_error(k, token): v for k, v in counts.items()}
            print("data.type 数量：" + json.dumps(safe_counts, ensure_ascii=False))
            for kind, shape in structures.items():
                print(safe_error(kind + " 首条 content 字段结构：" + json.dumps(shape, ensure_ascii=False), token))
        preview = answer.replace(token, "[已脱敏]")[:300]
        print("当前提取函数拼接的回答前 300 字：" + preview)
        print("原始 SSE 报告：" + str(path).replace(token, "[已脱敏]"))
        if error is not None:
            print("探测失败：" + safe_error(error, token), file=sys.stderr)
            return 1
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print("探测未完成：" + safe_error(exc, token), file=sys.stderr)
        return 2
    finally:
        accepting.clear()
        if report is not None:
            report.close()
        if client is not None:
            client._http.close()


if __name__ == "__main__":
    raise SystemExit(main())
