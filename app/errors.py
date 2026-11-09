"""对外错误和日志统一脱敏、限制长度。"""


def safe_error(message: object, token: str = "") -> str:
    text = str(message)
    if token:
        text = text.replace(token, "[已脱敏]")
    return text[:300]
