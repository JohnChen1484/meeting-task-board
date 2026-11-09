"""会议任务看板 - FastAPI 入口。

启动：uvicorn app.main:app --reload
页面：http://127.0.0.1:8000
接口文档：http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .ai_client import AIClient, AIParseError, AITimeoutError, build_client
from .config import BASE_DIR, Settings
from .db import STATUSES, TaskDB, DuplicateBatchError
from .validation import TaskRules
from .errors import safe_error

MAX_MEETING_CHARS = 20000


def setup_logging() -> None:
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    root = logging.getLogger("meeting_task_board")
    if root.handlers:
        return
    handler = RotatingFileHandler(log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("meeting_task_board")
    if not root.handlers:
        root.addHandler(handler)
        root.addHandler(logging.StreamHandler())
    root.setLevel(logging.INFO)


logger = logging.getLogger("meeting_task_board.api")


# ---------------------------------------------------------------- 数据模型

class ExtractRequest(BaseModel):
    meeting_text: str = Field(..., description="会议记录原文")

    @field_validator("meeting_text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("会议记录不能为空")
        if len(v) > MAX_MEETING_CHARS:
            raise ValueError(f"会议记录过长（超过 {MAX_MEETING_CHARS} 字），请分段处理")
        return v


class TaskIn(TaskRules):
    task: str
    owner: str = "未明确"
    due_date: str = "未明确"
    due_date_raw: str = "未明确"
    priority: str = "未明确"
    priority_basis: str = "无"

    @field_validator("task")
    @classmethod
    def task_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("任务内容不能为空")
        return v.strip()

    @field_validator("due_date_raw", "priority_basis")
    @classmethod
    def default_unknown(cls, v: str) -> str:
        return v.strip() or "未明确"


class SaveRequest(BaseModel):
    chat_id: Optional[str] = None
    extraction_id: UUID

    @field_validator("extraction_id", mode="before")
    @classmethod
    def valid_extraction_id(cls, value):
        try:
            return UUID(str(value))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("提取批次 extraction_id 必须是有效的 UUID") from None

    meeting_text: str = ""
    tasks: List[TaskIn]


class TaskPatch(TaskRules):
    status: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in STATUSES:
            raise ValueError(f"状态必须是 {STATUSES} 之一")
        return v


# ---------------------------------------------------------------- 应用

def create_app(settings: Optional[Settings] = None, ai_client: Optional[AIClient] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging()
    db = TaskDB(settings.db_path)
    client = ai_client or build_client(settings)

    app = FastAPI(title="会议任务看板", version="1.0.0")
    app.state.db = db
    app.state.ai = client
    app.state.settings = settings
    app.state.last_call = None

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # 不回显 Pydantic 的 input/context，避免原始输入或令牌出现在错误响应中。
        messages = []
        for error in exc.errors():
            kind = error["type"]
            if kind == "value_error":
                message = error["msg"].removeprefix("Value error, ")
            elif kind == "missing":
                message = f"缺少必填字段：{error['loc'][-1]}"
            else:
                message = f"字段 {error['loc'][-1]} 的类型或格式不正确"
            messages.append(message)
        return JSONResponse(status_code=422, content={"detail": safe_error("；".join(messages), settings.coze_api_token)})

    def record_call(success, error=None):
        app.state.last_call = {
            "success": success,
            "time": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }

    @app.get("/api/health")
    def health():
        configured = settings.ai_provider == "mock"
        if settings.ai_provider == "coze":
            configured = bool(settings.coze_api_token and settings.coze_bot_id)
        elif settings.ai_provider == "coze_deploy":
            configured = bool(settings.coze_api_token and settings.coze_deploy_url and settings.coze_project_id is not None)
        return {"ok": True, "ai_provider": settings.ai_provider,
                "ai_configured": configured,
                "last_call": app.state.last_call}

    @app.post("/api/extract")
    def extract(req: ExtractRequest):
        """调用 AI 提取任务，只返回预览，不写数据库。"""
        logger.info("extract request, text_len=%d", len(req.meeting_text))
        try:
            result = app.state.ai.extract(req.meeting_text)
        except Exception as e:
            code = 504 if isinstance(e, AITimeoutError) else 502
            prefix = "AI 回复无法解析：" if isinstance(e, AIParseError) else "AI 调用失败："
            detail = safe_error(prefix + str(e), settings.coze_api_token)
            record_call(False, detail)
            logger.warning("%s", detail)
            trace = {key: safe_error(getattr(e, key), settings.coze_api_token) if getattr(e, key, None) else None
                     for key in ("chat_id", "conversation_id", "session_id")}
            return JSONResponse(status_code=code, content={"detail": detail, **trace})
        record_call(True)
        return {
            "extraction_id": str(uuid4()),
            "session_id": result.session_id,
            "chat_id": result.chat_id,
            "conversation_id": result.conversation_id,
            "warnings": result.warnings,
            "summary": result.summary,
            "tasks": result.tasks,
            "has_tasks": len(result.tasks) > 0,
        }

    @app.post("/api/tasks")
    def save_tasks(req: SaveRequest):
        """用户确认后批量保存。"""
        if not req.tasks:
            raise HTTPException(status_code=400, detail="没有可保存的任务")
        source = req.meeting_text.strip().replace("\n", " ")[:50]
        try:
            saved = app.state.db.add_many([t.model_dump() for t in req.tasks], source, str(req.extraction_id), req.chat_id or "")
        except DuplicateBatchError:
            raise HTTPException(status_code=409, detail="这批任务已保存过") from None
        logger.info("saved %d tasks", len(saved))
        return {"saved": len(saved), "tasks": saved}

    @app.get("/api/tasks")
    def list_tasks(status: Optional[str] = Query(None, description="按状态筛选")):
        if status is not None and status not in STATUSES:
            raise HTTPException(status_code=400, detail=f"状态必须是 {STATUSES} 之一")
        return {"tasks": app.state.db.list(status)}

    @app.patch("/api/tasks/{task_id}")
    def patch_task(task_id: int, patch: TaskPatch):
        fields = patch.model_dump(exclude_none=True)
        try:
            updated = app.state.db.update(task_id, fields)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=safe_error(e, settings.coze_api_token)) from None
        if updated is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return updated

    @app.delete("/api/tasks/{task_id}")
    def delete_task(task_id: int):
        if not app.state.db.delete(task_id):
            raise HTTPException(status_code=404, detail="任务不存在")
        return {"deleted": task_id}

    static_dir = BASE_DIR / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
