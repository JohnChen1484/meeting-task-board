"""AI 调用封装。

对外只暴露 `AIClient.extract(meeting_text) -> ExtractResult`，
以后换成其他大模型服务（如火山方舟、Claude API），只需新增一个实现同样方法的类。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Protocol
from threading import Event, Thread
from uuid import uuid4
import math

import httpx

from .config import Settings, parse_project_id, validate_deploy_url
from .errors import safe_error
from .validation import normalize_owner, validate_due_date, validate_priority

logger = logging.getLogger("meeting_task_board.ai")

TASK_FIELDS = ["task", "owner", "due_date", "due_date_raw", "priority", "priority_basis"]
UNKNOWN = "未明确"
PROMPT_PREFIX = "帮我整理一下这个会，并额外输出 JSON："


class AIError(Exception):
    """AI 调用失败（接口报错、对话失败等）。"""


class AITimeoutError(AIError):
    """AI 在限定时间内没有完成。"""


class AIParseError(AIError):
    """AI 回复中找不到或无法解析 JSON。"""


@dataclass
class ExtractResult:
    summary: str
    tasks: List[dict] = field(default_factory=list)
    raw_answer: str = ""
    warnings: List[str] = field(default_factory=list)
    chat_id: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None


# ---------------------------------------------------------------- JSON 解析

_JSON_BLOCK = re.compile(r"```[ \t]*json[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_answer(answer: str) -> ExtractResult:
    """从智能体回复中取出最后一个 ```json 代码块。

    - 代码块之前的文字作为纪要原文（summary）
    - 代码块必须是 JSON 数组；每项缺失的字段补「未明确」
    - 找不到代码块或解析失败时抛 AIParseError，不返回半成品
    """
    if not answer or not answer.strip():
        raise AIParseError("智能体返回了空回复")

    matches = list(_JSON_BLOCK.finditer(answer))
    if not matches:
        raise AIParseError("智能体回复中没有找到 ```json 代码块，请检查智能体人设中的「程序调用补充规则」")

    last = matches[-1]
    try:
        data = json.loads(last.group(1).strip())
    except json.JSONDecodeError as e:
        raise AIParseError(f"JSON 代码块格式错误：{e.msg}（第 {e.lineno} 行）") from e

    if not isinstance(data, list):
        raise AIParseError("JSON 代码块不是数组")

    tasks = []
    warnings = []
    labels = {"task": "任务内容", "owner": "负责人", "due_date": "截止日期",
              "due_date_raw": "截止日期原文", "priority": "优先级", "priority_basis": "优先级依据"}

    def original(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise AIParseError(f"JSON 数组第 {i} 项不是对象")
        task = {}
        preserved = []
        bad_date = None
        for key in TASK_FIELDS:
            value = item.get(key)
            # 缺失表示未知；显式非字符串值需提示，不强转为业务字符串。
            if key not in item:
                task[key] = UNKNOWN
                continue
            if key == "owner" and isinstance(value, list) and all(isinstance(v, str) for v in value):
                value = "、".join(v.strip() for v in value if v.strip())
            if isinstance(value, str):
                value = value.strip() or UNKNOWN
            if key == "priority" and isinstance(value, str):
                annotated = re.fullmatch(r"(高|中|低)[（(](.+)[）)]", value, re.DOTALL)
                if annotated:
                    value = annotated.group(1)
                    preserved.append(annotated.group(2).strip())
            validator = {"owner": normalize_owner, "due_date": validate_due_date,
                         "priority": validate_priority}.get(key)
            try:
                if not isinstance(value, str):
                    raise ValueError("字段必须是文本")
                task[key] = validator(value) if validator else value
            except ValueError:
                task[key] = UNKNOWN
                raw = original(item[key])
                reason = "不是标准日期" if key == "due_date" else "类型或取值不合法"
                warnings.append(f"第{i}条任务的{labels[key]}'{raw}'{reason}，已设为未明确")
                if key == "due_date":
                    bad_date = raw
                else:
                    preserved.append(f"{key}原值：{raw}")
        if bad_date is not None:
            # 同时保留已经存在的原文与非法日期，不丢弃任何一项。
            existing = task["due_date_raw"]
            task["due_date_raw"] = bad_date if existing == UNKNOWN else existing + "；返回日期原值：" + bad_date
        basis = task["priority_basis"]
        parts = ([] if basis in (UNKNOWN, "无") else [basis]) + preserved
        if parts:
            task["priority_basis"] = "；".join(dict.fromkeys(parts))
        if task["task"] == UNKNOWN and not any(w.startswith(f"第{i}条任务的任务内容") for w in warnings):
            warnings.append(f"第{i}条任务的任务内容未明确，请人工补充后保存")
        tasks.append(task)

    summary = answer[: last.start()].strip()
    return ExtractResult(summary=summary, tasks=tasks, raw_answer=answer, warnings=warnings)


# ---------------------------------------------------------------- 客户端接口

class AIClient(Protocol):
    def extract(self, meeting_text: str) -> ExtractResult: ...


class CozeClient:
    """扣子开发平台 v3 对话 API，非流式：发起对话 → 轮询状态 → 获取消息列表。"""

    def __init__(self, settings: Settings, poll_interval: float = 1.5, http: httpx.Client | None = None):
        if not settings.coze_api_token or not settings.coze_bot_id:
            raise ValueError("AI_PROVIDER=coze 时必须在 .env 中填写 COZE_API_TOKEN 和 COZE_BOT_ID")
        self.token = settings.coze_api_token
        self.base = settings.coze_api_base.rstrip("/")
        self.chat_path = settings.coze_chat_path
        self.retrieve_path = settings.coze_retrieve_path
        self.messages_path = settings.coze_messages_path
        for path in (self.chat_path, self.retrieve_path, self.messages_path):
            if not path.startswith("/") or path.startswith("//") or "#" in path:
                raise ValueError("Coze 接口路径必须是以单个 / 开头的相对路径，不能包含片段")
        self.bot_id = settings.coze_bot_id
        self.timeout = settings.coze_timeout_seconds
        self.poll_interval = poll_interval
        self._http = http or httpx.Client(
            timeout=20.0,
            headers={
                "Authorization": f"Bearer {settings.coze_api_token}",
                "Content-Type": "application/json",
            },
        )

    def _check(self, resp: httpx.Response, step: str) -> dict:
        if resp.status_code != 200:
            raise AIError(f"扣子接口{step}失败：HTTP {resp.status_code}")
        try:
            body = resp.json()
        except ValueError:
            raise AIError(f"扣子接口{step}返回的不是 JSON") from None
        if not isinstance(body, dict):
            raise AIError(f"扣子接口{step}响应必须是对象")
        if "code" in body and type(body["code"]) is not int:
            raise AIError(f"扣子接口{step}的 code 字段类型不正确")
        if body.get("code", 0) != 0:
            raise AIError(safe_error(f"扣子接口{step}报错：code={body['code']} msg={body.get('msg')}", self.token))
        return body

    @staticmethod
    def _chat_data(body: dict, step: str, initial: bool = False) -> dict:
        data = body.get("data")
        if not isinstance(data, dict):
            raise AIError(f"扣子接口{step}缺少有效的 data 对象")
        for key in (("id", "conversation_id", "status") if initial else ("status",)):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise AIError(f"扣子接口{step}缺少有效的 {key} 文本字段")
        for key in ("id", "conversation_id"):
            if key in data and (not isinstance(data[key], str) or not data[key].strip()):
                raise AIError(f"扣子接口{step}的 {key} 字段类型或内容不正确")
        if data["status"] not in {"created", "in_progress", "completed", "failed", "requires_action", "canceled"}:
            raise AIError(f"扣子接口{step}的 status 不受支持")
        if data.get("last_error") is not None and not isinstance(data["last_error"], dict):
            raise AIError(f"扣子接口{step}的 last_error 必须是对象")
        last_error = data.get("last_error") or {}
        if "msg" in last_error and not isinstance(last_error["msg"], str):
            raise AIError(f"扣子接口{step}的 last_error.msg 必须是文本")
        return data

    def extract(self, meeting_text: str) -> ExtractResult:
        deadline = time.monotonic() + self.timeout
        params = {}

        def failure(exc):
            for key in ("chat_id", "conversation_id"):
                value = params.get(key)
                setattr(exc, key, safe_error(value, self.token) if value else None)
            return exc

        def remaining():
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise AITimeoutError("智能体调用超过整体时限，请稍后重试")
            return budget

        def request(method, path, step, **kwargs):
            # 三个阶段共用截止时间；每个请求使用最新的剩余预算。
            timeout = min(20.0, remaining())
            done = Event()
            outcome = {}

            def perform():
                try:
                    outcome["response"] = self._http.request(method, self.base + path, timeout=timeout, **kwargs)
                except Exception as e:
                    outcome["error"] = e
                finally:
                    done.set()

            # httpx 的分阶段/空闲超时不能单独保证墙钟时限。
            # 等待也受截止时间约束；超时后不再开始后续轮询或获取消息。
            Thread(target=perform, daemon=True).start()
            if not done.wait(min(timeout, remaining())):
                remaining()
                raise AITimeoutError("连接扣子接口超时，请稍后重试")
            remaining()
            if "error" in outcome:
                raise outcome["error"]
            return self._check(outcome["response"], step)

        try:
            body = request("POST", self.chat_path, "发起对话", json={
                "bot_id": self.bot_id, "user_id": "meeting-task-board", "stream": False,
                "auto_save_history": True,
                "additional_messages": [{"role": "user", "content": PROMPT_PREFIX + meeting_text, "content_type": "text"}],
            })
            data = self._chat_data(body, "发起对话", initial=True)
            params = {"conversation_id": data["conversation_id"], "chat_id": data["id"]}
            logger.info("%s", safe_error(f"coze chat_id={data['id']} conversation_id={data['conversation_id']}", self.token))
            while data["status"] not in ("completed", "failed", "requires_action", "canceled"):
                time.sleep(min(self.poll_interval, remaining()))
                body = request("GET", self.retrieve_path, "查询状态", params=params)
                data = self._chat_data(body, "查询状态")
            if data["status"] == "failed":
                err = data.get("last_error") or {}
                raise AIError(f"智能体对话失败：{err.get('msg') or '未提供原因'}")
            if data["status"] != "completed":
                raise AIError(f"智能体对话未正常完成（状态：{data['status']}）")
            body = request("GET", self.messages_path, "获取消息", params=params)
            messages = body.get("data")
            if not isinstance(messages, list):
                raise AIError("扣子接口获取消息缺少有效的 data 数组")
            answers = []
            for message in messages:
                if not isinstance(message, dict):
                    raise AIError("扣子消息列表的每一项必须是对象")
                for key in ("role", "type", "content"):
                    if not isinstance(message.get(key), str):
                        raise AIError(f"扣子消息缺少有效的 {key} 文本字段")
                if message["role"] == "assistant" and message["type"] == "answer":
                    answers.append(message["content"])
            result = parse_answer("\n".join(answers))
            result.chat_id = safe_error(params["chat_id"], self.token)
            result.conversation_id = safe_error(params["conversation_id"], self.token)
            result.warnings = [safe_error(w, self.token) for w in result.warnings]
            remaining()
            return result
        except httpx.TimeoutException:
            raise failure(AITimeoutError("连接扣子接口超时，请稍后重试")) from None
        except httpx.HTTPError as e:
            raise failure(AIError(safe_error(f"连接扣子接口失败：{type(e).__name__}", self.token))) from None
        except AIError as e:
            raise failure(type(e)(safe_error(e, self.token))) from None


def field_structure(value):
    """仅保留 JSON 字段名及层级，不保留任何叶子值。"""
    if isinstance(value, dict):
        return {key: field_structure(child) for key, child in value.items()}
    if isinstance(value, list):
        return [field_structure(child) for child in value[:3]]
    return None


def extract_deploy_answer(data: dict, token: str = "") -> str | None:
    """部署 answer 格式的唯一适配点；未知结构不猜测深层字段。"""
    content = data.get("content")
    if isinstance(content, dict):
        for key in ("answer", "text"):
            if isinstance(content.get(key), str):
                return content[key]
    if isinstance(content, str):
        return content
    structure = json.dumps(field_structure(data), ensure_ascii=False)
    logger.warning("%s", safe_error("部署 answer 文本字段未识别；字段结构：" + structure, token))
    return None


def iter_sse_events(lines):
    """按 SSE 空行分帧，支持注释、多行 data 和 EOF 最后一帧。"""
    event = "message"
    data_lines = []

    def decode():
        if not data_lines:
            return None
        try:
            data = json.loads("\n".join(data_lines))
        except ValueError:
            raise AIError("部署服务 SSE 的 data 不是合法 JSON") from None
        if not isinstance(data, dict):
            raise AIError("部署服务 SSE 的 data 必须是 JSON 对象")
        return data

    for index, raw_line in enumerate(lines):
        line = raw_line.removeprefix("\ufeff") if index == 0 else raw_line
        if not line:
            if data_lines or event == "message_end":
                yield event, decode()
            event, data_lines = "message", []
        elif not line.startswith(":"):
            name, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if name == "event":
                event = value
            elif name == "data":
                data_lines.append(value)
    if data_lines or event == "message_end":
        yield event, decode()


@dataclass
class DeployRun:
    session_id: str
    answer: str = ""
    warnings: List[str] = field(default_factory=list)


class CozeDeployClient:
    """独立部署服务 /stream_run SSE 适配；每次请求使用全新会话。"""

    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        if not settings.coze_api_token:
            raise ValueError("AI_PROVIDER=coze_deploy 时必须配置 COZE_API_TOKEN（部署 API Token）")
        self.base = validate_deploy_url(settings.coze_deploy_url)
        self.project_id = parse_project_id(settings.coze_project_id)
        if self.project_id is None:
            raise ValueError("AI_PROVIDER=coze_deploy 时必须配置 COZE_PROJECT_ID")
        self.timeout = settings.coze_timeout_seconds
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("COZE_TIMEOUT_SECONDS 必须为正的有限秒数")
        self.token = settings.coze_api_token
        self._http = http or httpx.Client(timeout=20.0)

    def stream_run(self, full_text: str, *, on_line=None, on_status=None, on_event=None) -> DeployRun:
        """单次流式调用。回调供探测使用，必须只处理数据，不额外发请求。"""
        run = DeployRun(session_id="mtb-" + uuid4().hex)
        deadline = time.monotonic() + self.timeout
        done, stop = Event(), Event()
        outcome = {}
        parts = []

        def remaining():
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise AITimeoutError("部署服务调用超过整体时限，请稍后重试")
            return budget

        def receive():
            try:
                remaining()
                body = {
                    "content": {"query": {"prompt": [{"type": "text", "content": {"text": full_text}}]}},
                    "type": "query", "session_id": run.session_id, "project_id": self.project_id,
                }
                with self._http.stream("POST", self.base + "/stream_run", json=body,
                                       headers={"Authorization": f"Bearer {self.token}",
                                                "Content-Type": "application/json"},
                                       timeout=min(20.0, remaining())) as response:
                    if stop.is_set():
                        return
                    if on_status:
                        on_status(response.status_code)
                    remaining()
                    if response.status_code in (401, 403):
                        raise AIError("部署服务的 API Token 无效或无权限")
                    if response.status_code != 200:
                        raise AIError(f"部署服务请求失败：HTTP {response.status_code}")
                    def lines():
                        for line in response.iter_lines():
                            if stop.is_set():
                                return
                            remaining()
                            if on_line:
                                on_line(line)
                            yield line

                    for event_name, data in iter_sse_events(lines()):
                        if stop.is_set():
                            return
                        remaining()
                        if data is not None:
                            kind = data.get("type")
                            if not isinstance(kind, str):
                                raise AIError("部署服务 SSE 事件缺少有效的 type 文本字段")
                            if on_event:
                                on_event(data)
                            if kind == "error":
                                content = data.get("content")
                                message = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                                raise AIError(safe_error("部署服务返回错误：" + message, self.token))
                            if kind == "answer":
                                text = extract_deploy_answer(data, self.token)
                                if text is None:
                                    run.warnings.append("有 answer 事件未识别文本字段，请运行探测脚本确认格式")
                                else:
                                    parts.append(text)
                            if kind == "message_end":
                                break
                        if event_name == "message_end":
                            break
                    remaining()
                    run.answer = "".join(parts)
                    outcome["run"] = run
            except Exception as exc:
                outcome["error"] = exc
            finally:
                done.set()

        logger.info("%s", safe_error("coze_deploy session_id=" + run.session_id, self.token))
        Thread(target=receive, daemon=True).start()
        try:
            if not done.wait(remaining()):
                raise AITimeoutError("部署服务调用超过整体时限，请稍后重试")
            remaining()
            if "error" in outcome:
                raise outcome["error"]
            return outcome["run"]
        except Exception as exc:
            if isinstance(exc, (AITimeoutError, httpx.TimeoutException)):
                error = AITimeoutError("部署服务调用超时，请稍后重试")
            elif isinstance(exc, AIError):
                error = type(exc)(safe_error(exc, self.token))
            else:
                error = AIError(safe_error("部署服务连接或读取失败：" + type(exc).__name__, self.token))
            error.session_id = run.session_id
            error.partial_answer = "".join(parts)
            raise error from None
        finally:
            # 超时后禁止后台读取线程继续触发探测写入或处理后续事件。
            stop.set()

    def extract(self, meeting_text: str) -> ExtractResult:
        started = time.monotonic()
        run = self.stream_run(PROMPT_PREFIX + meeting_text)
        try:
            result = parse_answer(run.answer)
        except AIError as exc:
            error = type(exc)(safe_error(exc, self.token))
            error.session_id = run.session_id
            raise error from None
        if time.monotonic() - started > self.timeout:
            error = AITimeoutError("部署服务调用超过整体时限，请稍后重试")
            error.session_id = run.session_id
            raise error
        result.session_id = run.session_id
        result.warnings.extend(run.warnings)
        result.warnings = [safe_error(w, self.token) for w in result.warnings]
        return result



class MockClient:
    """不联网的演示客户端：返回固定格式的结果，用于在没有令牌时测试页面和流程。"""

    NON_MEETING_MARKER = "【演示：非会议】"

    def extract(self, meeting_text: str) -> ExtractResult:
        time.sleep(0.6)
        if self.NON_MEETING_MARKER in meeting_text:
            answer = "⚠️ 演示数据，与输入内容无关\n该内容不像会议记录，请问是否仍需整理？\n\n```json\n[]\n```"
        else:
            answer = (
                "⚠️ 演示数据，与输入内容无关\n## 会议纪要（演示数据，AI_PROVIDER=mock）\n"
                "**会议主题：** 项目例会\n**会议时间：** 2026-10-28\n**参会人员：** 未明确\n\n"
                "### 核心决定\n1. 下周一启动客户回访\n\n"
                "## 待办任务清单\n| 序号 | 任务事项 | 负责人 | 截止日期 | 优先级 |\n| --- | --- | --- | --- | --- |\n"
                "| 1 | 完成首轮客户回访 | 孙悦 | 2026-11-06 | 未明确 |\n"
                "| 2 | 更新报价单模板 | 未明确 | 未明确 | 未明确 |\n"
                "| 3 | 完成服务器到期续费 | 老周 | 未明确 | 高 |\n\n"
                "```json\n"
                + json.dumps(
                    [
                        {"task": "完成首轮客户回访", "owner": "孙悦", "due_date": "2026-11-06",
                         "due_date_raw": "11月6日前", "priority": "未明确", "priority_basis": "无"},
                        {"task": "更新报价单模板", "owner": "未明确", "due_date": "未明确",
                         "due_date_raw": "未明确", "priority": "未明确", "priority_basis": "无"},
                        {"task": "完成服务器到期续费", "owner": "老周", "due_date": "未明确",
                         "due_date_raw": "这两天", "priority": "高", "priority_basis": "推断：原文\"服务器到期\""},
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n```"
            )
        return parse_answer(answer)


def build_client(settings: Settings) -> AIClient:
    if settings.ai_provider == "coze":
        return CozeClient(settings)
    if settings.ai_provider == "coze_deploy":
        return CozeDeployClient(settings)
    if settings.ai_provider == "mock":
        return MockClient()
    raise ValueError(f"不支持的 AI_PROVIDER：{settings.ai_provider}（可选 mock / coze / coze_deploy）")
