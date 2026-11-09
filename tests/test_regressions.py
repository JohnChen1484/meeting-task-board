"""审查回归：仅临时/内存数据库，无真实 AI 或业务数据。"""
from concurrent.futures import ThreadPoolExecutor
import logging
import sqlite3
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest

from app.ai_client import AIParseError, ExtractResult, MockClient
from app.config import Settings
from app.db import TaskDB
from app.main import create_app
from tests.test_api import FakeAI, TASKS


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Settings(db_path=str(tmp_path/'audit.db')), FakeAI()))


def save(client, tasks=None, batch=None):
    return client.post('/api/tasks', json={'extraction_id': batch or str(uuid4()), 'tasks': tasks if tasks is not None else TASKS})


@pytest.mark.parametrize('text', ['据报道', '新闻项目例会', '中午吃啥', '普通会议', '无行动项'])
def test_mock_always_demo_without_marker(text, monkeypatch):
    monkeypatch.setattr('app.ai_client.time.sleep', lambda _: None)
    result = MockClient().extract(text)
    assert len(result.tasks) == 3
    assert result.summary.startswith('⚠️ 演示数据，与输入内容无关')
    rows = [line for line in result.summary.splitlines() if line.startswith(('| 1 |', '| 2 |', '| 3 |'))]
    assert len(rows) == 3
    for row, task in zip(rows, result.tasks):
        assert [v.strip() for v in row.split('|')[2:-1]] == [task[k] for k in ('task','owner','due_date','priority')]


def test_mock_non_meeting_marker(monkeypatch):
    monkeypatch.setattr('app.ai_client.time.sleep', lambda _: None)
    result = MockClient().extract('前文【演示：非会议】后文')
    assert result.tasks == []
    assert result.summary.startswith('⚠️ 演示数据，与输入内容无关')


@pytest.mark.parametrize('payload', [{}, {'meeting_text':''}, {'meeting_text':' \n '}, {'meeting_text':None}, {'meeting_text':'x'*20001}])
def test_invalid_extract_does_not_call_ai(client, payload):
    assert client.post('/api/extract', json=payload).status_code == 422
    assert client.app.state.ai.calls == 0
    assert client.app.state.last_call is None
    assert client.get('/api/tasks').json()['tasks'] == []


def test_extract_uuid_and_health_history(client):
    assert client.get('/api/health').json()['last_call'] is None
    assert client.get('/api/health').json()['ai_configured'] is True
    client.app.state.ai = FakeAI(result=ExtractResult('纪要', TASKS))
    first = client.post('/api/extract', json={'meeting_text':'原文'}).json()
    second = client.post('/api/extract', json={'meeting_text':'原文'}).json()
    assert str(UUID(first['extraction_id'])) == first['extraction_id']
    assert first['extraction_id'] != second['extraction_id']
    call = client.get('/api/health').json()['last_call']
    assert call['success'] is True and call['error'] is None
    from datetime import datetime
    assert datetime.fromisoformat(call['time']).tzinfo is not None
    assert client.get('/api/tasks').json()['tasks'] == []
    client.app.state.ai = FakeAI(error=AIParseError('错误 JSON'))
    assert client.post('/api/extract', json={'meeting_text':'x'}).status_code == 502
    call = client.get('/api/health').json()['last_call']
    assert call['success'] is False and '错误 JSON' in call['error']


def test_coze_configured_is_not_verified(tmp_path):
    c = TestClient(create_app(Settings(ai_provider='coze', coze_api_token='FAKE',coze_bot_id='bot', db_path=str(tmp_path/'h.db')), FakeAI()))
    assert c.get('/api/health').json()['ai_configured'] is True
    assert c.get('/api/health').json()['last_call'] is None
    c.app.state.settings.coze_api_token = ''
    assert c.get('/api/health').json()['ai_configured'] is False


@pytest.mark.parametrize('batch', [None, '', 'not-uuid', 123])
def test_batch_id_required(client, batch):
    data={'tasks': TASKS}
    if batch is not None:
        data['extraction_id'] = batch
    r = client.post('/api/tasks', json=data)
    assert r.status_code == 422 and 'extraction_id' in r.json()['detail']
    assert client.get('/api/tasks').json()['tasks'] == []


def test_duplicate_and_reopen(client):
    batch = str(uuid4())
    assert save(client, batch=batch).status_code == 200
    other = TestClient(create_app(Settings(db_path=client.app.state.db.path), FakeAI()))
    duplicate = save(other, batch=batch)
    assert duplicate.status_code == 409 and duplicate.json()['detail'] == '这批任务已保存过'
    assert len(other.get('/api/tasks').json()['tasks']) == 2
    # UUID 的等价写法也不能绕过去重。
    assert save(other, batch=batch.upper()).status_code == 409


@pytest.mark.parametrize('memory', [False, True])
def test_concurrent_duplicate_is_atomic(tmp_path, memory):
    path = ':memory:' if memory else str(tmp_path/'race.db')
    app = create_app(Settings(db_path=path), FakeAI())
    c1 = TestClient(app)
    c2 = TestClient(app if memory else create_app(Settings(db_path=path), FakeAI()))
    batch = str(uuid4())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda c: save(c, batch=batch).status_code, [c1,c2]))
    assert sorted(results) == [200,409]
    assert len(c1.get('/api/tasks').json()['tasks']) == 2


def test_db_batch_and_tasks_rollback(tmp_path):
    db = TaskDB(str(tmp_path/'rollback.db'))
    batch = str(uuid4())
    with pytest.raises(sqlite3.IntegrityError):
        db.add_many([TASKS[0], dict(TASKS[1], task=None)], '测试', batch)
    assert db.list() == []
    assert len(db.add_many(TASKS, '测试', batch)) == 2


@pytest.mark.parametrize('tasks,code', [([],400),([{'task':' '}],422),([{'owner':'x'}],422),([{'task':'ok'},{'task':''}],422)])
def test_invalid_batch_can_retry_without_partial_save(client, tasks, code):
    batch = str(uuid4())
    assert save(client,tasks,batch).status_code == code
    assert client.get('/api/tasks').json()['tasks'] == []
    assert save(client,batch=batch).status_code == 200


@pytest.mark.parametrize('field,value', [('priority','随便写'),('priority',''),('priority',None),('due_date','2026-02-29'),('due_date','2026-99-99'),('due_date','2026-1-01'),('due_date','2026-01-01T00:00:00'),('due_date',''),('due_date',None),('owner',None)])
def test_shared_validation_rejects_save_and_patch(client, field, value):
    saved = save(client).json()['tasks']
    r = save(client, [dict(TASKS[0], **{field:value})])
    p = client.patch(f"/api/tasks/{saved[0]['id']}", json={field:value})
    assert r.status_code == p.status_code == 422
    assert any('\u4e00' <= ch <= '\u9fff' for ch in r.json()['detail'])
    assert len(client.get('/api/tasks').json()['tasks']) == 2


@pytest.mark.parametrize('date', ['2028-02-29','2026-11-02','未明确'])
@pytest.mark.parametrize('priority', ['高','中','低','未明确'])
def test_shared_validation_accepts(client, date, priority):
    t = save(client, [{'task':'审计', 'owner':'  张三  ', 'due_date':date, 'due_date_raw':'明天/这两天均允许', 'priority':priority}]).json()['tasks'][0]
    assert t['owner'] == '张三' and t['due_date'] == date and t['priority'] == priority
    p = client.patch(f"/api/tasks/{t['id']}", json={'owner':'  李四  ','due_date':date,'priority':priority})
    assert p.status_code == 200 and p.json()['owner'] == '李四'


@pytest.mark.parametrize('owner', ['', ' \n ', ' 未明确 '])
def test_unknown_owner_normalized_in_save_and_patch(client, owner):
    t = save(client, [{'task':'缺少日期、负责人', 'owner':owner}]).json()['tasks'][0]
    assert t['owner'] == '未明确' and t['status'] == '待分配'
    assert t['due_date'] == t['priority'] == '未明确'
    t = client.patch(f"/api/tasks/{t['id']}", json={'owner':'张三'}).json()
    assert t['status'] == '待开始'
    t = client.patch(f"/api/tasks/{t['id']}", json={'owner':owner}).json()
    assert t['owner'] == '未明确' and t['status'] == '待分配'


@pytest.mark.parametrize('state', ['进行中','已完成'])
def test_clear_owner_keeps_active_or_done_state(client, state):
    tid = save(client).json()['tasks'][0]['id']
    client.patch(f'/api/tasks/{tid}', json={'status':state})
    t = client.patch(f'/api/tasks/{tid}', json={'owner':''}).json()
    assert t['owner'] == '未明确' and t['status'] == state


@pytest.mark.parametrize('patch', [{'status':'待开始'}, {'status':'待开始','owner':''}, {'status':'待开始','owner':'未明确'}])
def test_unassigned_cannot_start(client, patch):
    tid = save(client,[{'task':'待分配任务'}]).json()['tasks'][0]['id']
    r = client.patch(f'/api/tasks/{tid}', json=patch)
    assert r.status_code == 422 and r.json()['detail'] == '请先指定负责人'
    assert client.app.state.db.get(tid)['status'] == '待分配'


def test_explicit_owner_and_status(client):
    tid = save(client,[{'task':'任务'}]).json()['tasks'][0]['id']
    assert client.patch(f'/api/tasks/{tid}',json={'owner':'李四','status':'进行中'}).json()['status'] == '进行中'
    assert client.patch(f'/api/tasks/{tid}',json={'owner':'未明确','status':'待开始'}).status_code == 422
    assert client.app.state.db.get(tid)['owner'] == '李四'


def test_management_and_restart(client):
    tid = save(client,[{'task':'人工修改任务','owner':'李四','due_date':'2026-11-02','priority':'低'}]).json()['tasks'][0]['id']
    assert client.get('/').status_code == 200
    assert client.patch(f'/api/tasks/{tid}',json={'status':'进行中','due_date':'2026-11-03','priority':'高'}).status_code == 200
    restarted = TestClient(create_app(Settings(db_path=client.app.state.db.path),FakeAI()))
    t = restarted.get('/api/tasks',params={'status':'进行中'}).json()['tasks'][0]
    assert t['task'] == '人工修改任务' and t['due_date'] == '2026-11-03' and t['priority'] == '高'
    assert restarted.delete(f'/api/tasks/{tid}').status_code == 200
    assert restarted.delete(f'/api/tasks/{tid}').status_code == 404
    assert restarted.patch(f'/api/tasks/{tid}',json={'owner':'李四'}).status_code == 404


def test_memory_settings_and_crud(monkeypatch):
    monkeypatch.setenv('DB_PATH', ':memory:')
    assert Settings.from_env().db_path == ':memory:'
    c = TestClient(create_app(Settings(db_path=':memory:'), FakeAI()))
    assert c.get('/api/tasks').json()['tasks'] == []
    tid = save(c).json()['tasks'][0]['id']
    assert c.patch(f'/api/tasks/{tid}',json={'priority':'低'}).status_code == 200
    assert c.delete(f'/api/tasks/{tid}').status_code == 200
    c.app.state.db.close()
