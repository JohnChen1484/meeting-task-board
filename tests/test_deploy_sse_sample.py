"""真实部署 SSE 样本的离线回归；HTTP 传输完全由 MockTransport 提供。"""
from collections import Counter
from pathlib import Path

import httpx
import pytest

from app.ai_client import (
    CozeDeployClient, extract_deploy_answer, iter_sse_events, parse_answer,
)
from app.config import Settings


SAMPLE = Path(__file__).parent / "fixtures" / "deploy_sse_sample.txt"
EXPECTED_ANSWER = '''```markdown
## 会议纪要
**会议主题：** 紫杉项目周会
**会议时间：** 2026-10-29
**参会人员：** 未明确

### 核心决定
1. 王一鸣需在明天前提交蓝鲸接口文档

### 关键讨论要点
- 本次会议明确了蓝鲸接口文档的提交责任人和时限要求
```

```markdown
## 待办任务清单
| 序号 | 任务事项 | 负责人 | 截止日期 | 优先级 |
| --- | --- | --- | --- | --- |
| 1 | 提交蓝鲸接口文档 | 王一鸣 | 明天前（推算：2026-10-30） | 未明确 |
```

```markdown
## 待确认事项
| 序号 | 任务 | 缺失字段 | 原文线索 |
| --- | --- | --- | --- |
| 1 | 提交蓝鲸接口文档 | 优先级 | 原文无相关信息 |
```

```json
[
  {
    "task": "提交蓝鲸接口文档",
    "owner": "王一鸣",
    "due_date": "2026-10-30",
    "due_date_raw": "明天前",
    "priority": "未明确",
    "priority_basis": "无"
  }
]
```'''
EXPECTED_TASKS = [{
    "task": "提交蓝鲸接口文档", "owner": "王一鸣",
    "due_date": "2026-10-30", "due_date_raw": "明天前",
    "priority": "未明确", "priority_basis": "无",
}]


def test_recorded_event_contract_and_incremental_answer():
    events = list(iter_sse_events(SAMPLE.read_text(encoding="utf-8").splitlines()))
    assert Counter(name for name, _ in events) == {"message": 310}
    assert [data["type"] for _, data in events] == ["message_start"] + ["answer"] * 308 + ["message_end"]
    assert [data["sequence_id"] for _, data in events] == list(range(1, 311))
    answers = [data for _, data in events if data["type"] == "answer"]
    assert [data["finish"] for data in answers] == [False] * 307 + [True]
    assert extract_deploy_answer(answers[-1]) == ""
    answer = "".join(extract_deploy_answer(data) for data in answers)
    # 完整文本比较同时防止字段选错、漏段、把增量误判为累积全文或重复拼接。
    assert answer == EXPECTED_ANSWER
    result = parse_answer(answer)
    assert result.tasks == EXPECTED_TASKS
    assert result.warnings == []
    end = events[-1][1]
    assert end["content"]["answer"] is None
    assert end["content"]["message_end"] == {
        "code": "0", "message": "",
        "token_cost": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "time_cost_ms": 6468,
    }
    assert end["msg_id"] and end["reply_id"] and end["session_id"] and end["log_id"]
    assert all(data["content"][key] is None for _, data in events
               for key in ("tool_request", "tool_response"))


@pytest.mark.parametrize("chunk_size", [3, 140903])
def test_recorded_sse_through_client_without_network(chunk_size):
    raw = SAMPLE.read_bytes()

    class RecordedStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            for offset in range(0, len(raw), chunk_size):
                yield raw[offset:offset + chunk_size]

        def close(self):
            self.closed = True

    stream = RecordedStream()
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    settings = Settings(ai_provider="coze_deploy", coze_api_token="SYNTHETIC-OFFLINE",
                        coze_deploy_url="https://deploy.example.invalid", coze_project_id=123)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = CozeDeployClient(settings, http=http)
        result = client.extract("离线回放真实 SSE 样本")
    assert len(requests) == 1
    assert stream.closed
    assert result.raw_answer == EXPECTED_ANSWER
    assert result.tasks == EXPECTED_TASKS
    assert result.warnings == []
    assert result.session_id.startswith("mtb-")
