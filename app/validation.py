"""保存和修改任务共用的字段规则。"""
import re
from datetime import date

from pydantic import BaseModel, field_validator

UNKNOWN = "未明确"


def normalize_owner(value):
    if not isinstance(value, str):
        raise ValueError("负责人必须是文本")
    return value.strip() or UNKNOWN


def validate_priority(value):
    if not isinstance(value, str) or value.strip() not in ("高", "中", "低", UNKNOWN):
        raise ValueError("优先级只能是高、中、低、未明确")
    return value.strip()


def validate_due_date(value):
    message = "截止日期必须是有效的 YYYY-MM-DD 日期或未明确"
    if not isinstance(value, str):
        raise ValueError(message)
    value = value.strip()
    if value == UNKNOWN:
        return value
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError(message)
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(message) from None
    return value


class TaskRules(BaseModel):
    @field_validator("owner", mode="before", check_fields=False)
    @classmethod
    def normalize_owner(cls, value):
        return normalize_owner(value)

    @field_validator("priority", mode="before", check_fields=False)
    @classmethod
    def validate_priority(cls, value):
        return validate_priority(value)

    @field_validator("due_date", mode="before", check_fields=False)
    @classmethod
    def validate_due_date(cls, value):
        return validate_due_date(value)
