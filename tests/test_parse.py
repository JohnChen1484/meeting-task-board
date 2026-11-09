"""测试从智能体回复中提取 JSON 的逻辑（不联网）。"""
import pytest

from app.ai_client import AIParseError, parse_answer

NORMAL = """## 会议纪要
**会议主题：** 项目例会

## 待办任务清单
| 序号 | 任务事项 | 负责人 |

```json
[
  {"task": "完成首轮客户回访", "owner": "孙悦", "due_date": "2026-11-06",
   "due_date_raw": "11月6日前", "priority": "未明确", "priority_basis": "无"},
  {"task": "更新报价单模板", "owner": "未明确", "due_date": "未明确",
   "due_date_raw": "未明确", "priority": "未明确", "priority_basis": "无"}
]
```"""


def test_normal_answer():
    r = parse_answer(NORMAL)
    assert len(r.tasks) == 2
    assert r.tasks[0]["owner"] == "孙悦"
    assert r.tasks[1]["owner"] == "未明确"
    assert r.summary.startswith("## 会议纪要")
    assert "```json" not in r.summary


def test_empty_array_for_non_meeting():
    r = parse_answer("该内容不像会议记录，请问是否仍需整理？\n\n```json\n[]\n```")
    assert r.tasks == []
    assert "不像会议记录" in r.summary


def test_no_json_block_raises():
    with pytest.raises(AIParseError, match="没有找到"):
        parse_answer("## 会议纪要\n只有文字，没有代码块")


def test_invalid_json_raises():
    with pytest.raises(AIParseError, match="格式错误"):
        parse_answer('```json\n[{"task": "a",]\n```')


def test_not_array_raises():
    with pytest.raises(AIParseError, match="不是数组"):
        parse_answer('```json\n{"task": "a"}\n```')


def test_takes_last_block_and_fills_missing_fields():
    answer = '示例：\n```json\n[{"task": "旧的"}]\n```\n正式结果：\n```json\n[{"task": "新的", "owner": ""}]\n```'
    r = parse_answer(answer)
    assert [t["task"] for t in r.tasks] == ["新的"]
    assert r.tasks[0]["owner"] == "未明确"
    assert r.tasks[0]["priority"] == "未明确"


def test_empty_answer_raises():
    with pytest.raises(AIParseError):
        parse_answer("   ")
