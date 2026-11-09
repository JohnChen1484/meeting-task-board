"""读取 .env 配置。令牌只在后端内存中使用，不会返回给前端。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def parse_project_id(value) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    if type(value) is int and value >= 0:
        return value
    raise ValueError("COZE_PROJECT_ID 必须是数字形式的非负整数")


def validate_deploy_url(value: str) -> str:
    base = value.strip().rstrip("/")
    parts = urlsplit(base)
    if (parts.scheme != "https" or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment):
        raise ValueError("COZE_DEPLOY_URL 必须是无账号、查询参数或片段的 HTTPS 基地址")
    if parts.path.endswith("/stream_run"):
        raise ValueError("COZE_DEPLOY_URL 不要包含 /stream_run，程序会自动追加")
    return base


@dataclass
class Settings:
    ai_provider: str = "mock"
    coze_api_token: str = ""
    coze_bot_id: str = ""
    coze_deploy_url: str = ""
    coze_project_id: int | None = None
    coze_api_base: str = "https://api.coze.cn"
    coze_chat_path: str = "/v3/chat"
    coze_retrieve_path: str = "/v3/chat/retrieve"
    coze_messages_path: str = "/v3/chat/message/list"
    coze_timeout_seconds: float = 90.0
    db_path: str = str(BASE_DIR / "data" / "tasks.db")

    @classmethod
    def from_env(cls) -> "Settings":
        db_path = os.getenv("DB_PATH", "data/tasks.db")
        if db_path != ":memory:" and not os.path.isabs(db_path):
            db_path = str(BASE_DIR / db_path)
        return cls(
            ai_provider=os.getenv("AI_PROVIDER", "mock").strip().lower(),
            coze_api_token=os.getenv("COZE_API_TOKEN", "").strip(),
            coze_bot_id=os.getenv("COZE_BOT_ID", "").strip(),
            coze_deploy_url=os.getenv("COZE_DEPLOY_URL", "").strip().rstrip("/"),
            coze_project_id=parse_project_id(os.getenv("COZE_PROJECT_ID", "").strip()),
            coze_api_base=os.getenv("COZE_API_BASE", "https://api.coze.cn").rstrip("/"),
            coze_chat_path=os.getenv("COZE_CHAT_PATH", "/v3/chat").strip(),
            coze_retrieve_path=os.getenv("COZE_RETRIEVE_PATH", "/v3/chat/retrieve").strip(),
            coze_messages_path=os.getenv("COZE_MESSAGES_PATH", "/v3/chat/message/list").strip(),
            coze_timeout_seconds=float(os.getenv("COZE_TIMEOUT_SECONDS", "90")),
            db_path=db_path,
        )
