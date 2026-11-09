import json
import logging
import sqlite3
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app.ai_client import CozeClient, parse_answer
from app.config import Settings
from app.db import TaskDB
from app.main import TaskIn, create_app


def parse(tasks):
    return parse_answer('纪要\n```json\n'+json.dumps(tasks,ensure_ascii=False)+'\n```')


def test_owner_array_and_priority_annotations():
    result=parse([
        {'task':'直播','owner':[' 陈晨 ', '刘洋'],'priority':'高（推断：原文“测试环境老是挂”）','priority_basis':'已有依据'},
        {'task':'宣传','priority':'中（原文）'},
        {'task':'采购','priority':'低(原文)'},
    ])
    assert result.tasks[0]['owner']=='陈晨、刘洋'
    assert result.tasks[0]['priority']=='高'
    assert result.tasks[0]['priority_basis']=='已有依据；推断：原文“测试环境老是挂”'
    assert result.tasks[1]['priority']=='中' and result.tasks[1]['priority_basis']=='原文'
    assert result.tasks[2]['priority']=='低' and result.warnings==[]


@pytest.mark.parametrize('value',[123,True,{'a':'b'},['陈晨',123],None])
def test_bad_owner_warns_and_preserves_original(value):
    result=parse([{'task':'有效任务','owner':value},{'task':'另一任务','owner':'李四'}])
    assert result.tasks[0]['owner']=='未明确'
    assert 'owner原值：' in result.tasks[0]['priority_basis']
    assert result.tasks[1]['owner']=='李四'
    assert len(result.warnings)==1 and '第1条任务的负责人' in result.warnings[0]
    assert TaskIn(**result.tasks[0]).owner=='未明确'


@pytest.mark.parametrize('date',['11月10日','2026-02-30','明天',123,None])
def test_bad_date_downgraded_not_failed(date):
    result=parse([{'task':'日期校验','due_date':date,'due_date_raw':'原文片段'}])
    t=result.tasks[0]
    assert t['due_date']=='未明确'
    assert '原文片段' in t['due_date_raw'] and str(date if date is not None else 'null') in t['due_date_raw']
    assert '不是标准日期' in result.warnings[0]
    assert TaskIn(**t).due_date=='未明确'


def test_invalid_date_without_raw_keeps_value():
    result=parse([{'task':'任务','due_date':'11月10日'}])
    assert result.tasks[0]['due_date_raw']=='11月10日'
    assert result.warnings==["第1条任务的截止日期'11月10日'不是标准日期，已设为未明确"]


@pytest.mark.parametrize('priority',['紧急','P0',{'label':'高'},None])
def test_bad_priority_warns(priority):
    result=parse([{'task':'优先级','priority':priority,'priority_basis':'原文：紧急'}])
    assert result.tasks[0]['priority']=='未明确'
    assert '原文：紧急' in result.tasks[0]['priority_basis']
    assert 'priority原值：' in result.tasks[0]['priority_basis']
    assert len(result.warnings)==1
    TaskIn(**result.tasks[0])


def test_missing_fields_and_empty_owner_array():
    result=parse([{'task':'a'},{'task':'b','owner':[]}])
    assert result.warnings==[]
    assert result.tasks[0]['owner']==result.tasks[1]['owner']=='未明确'


def test_each_invalid_field_warns_without_dropping_task():
    result=parse([{'task':123,'owner':False,'due_date':[],'due_date_raw':{},'priority':1,'priority_basis':[]}])
    assert len(result.tasks)==1 and len(result.warnings)==6
    assert all(result.tasks[0][k]=='未明确' for k in ('task','owner','due_date','priority'))
    assert 'priority_basis原值：[]' in result.tasks[0]['priority_basis']
    TaskIn(**result.tasks[0])


def test_last_json_block_remains_authoritative():
    result=parse_answer('```json\n[{"task":"old"}]\n```\n```json\n[{"task":"new"}]\n```')
    assert [t['task'] for t in result.tasks]==['new']


def test_trace_warning_response_and_saved_batch(tmp_path,caplog):
    answer='纪要\n```json\n[{"task":"直播","owner":["陈晨","刘洋"],"due_date":"11月10日"}]\n```'
    def handler(req):
        if req.method=='POST':return httpx.Response(200,json={'data':{'id':'chat-trace','conversation_id':'conversation-trace','status':'completed'}})
        return httpx.Response(200,json={'data':[{'role':'assistant','type':'answer','content':answer}]})
    settings=Settings(ai_provider='coze',coze_api_token='FAKE-ONLY',coze_bot_id='bot',db_path=str(tmp_path/'trace.db'))
    ai=CozeClient(settings,http=httpx.Client(transport=httpx.MockTransport(handler)))
    c=TestClient(create_app(settings,ai))
    with caplog.at_level(logging.INFO,logger='meeting_task_board'):
        result=c.post('/api/extract',json={'meeting_text':'测试直播'}).json()
    assert result['chat_id']=='chat-trace' and result['conversation_id']=='conversation-trace'
    assert result['warnings'] and 'chat_id=chat-trace' in caplog.text and 'conversation_id=conversation-trace' in caplog.text
    assert c.get('/api/tasks').json()['tasks']==[]
    payload={'extraction_id':result['extraction_id'],'chat_id':result['chat_id'],'tasks':result['tasks']}
    assert c.post('/api/tasks',json=payload).status_code==200
    assert c.post('/api/tasks',json=payload).status_code==409
    with sqlite3.connect(settings.db_path) as db:
        assert db.execute('select chat_id from saved_batches').fetchall()==[('chat-trace',)]


def test_old_batch_schema_migration_is_idempotent(tmp_path):
    path=str(tmp_path/'legacy.db')
    batch=str(uuid4())
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE saved_batches (extraction_id TEXT PRIMARY KEY, saved_at TEXT NOT NULL)')
        db.execute('INSERT INTO saved_batches VALUES (?,?)',(batch,'2026-10-01'))
    TaskDB(path)
    TaskDB(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT extraction_id,saved_at,chat_id FROM saved_batches').fetchall()==[(batch,'2026-10-01','')]


def test_mock_has_no_coze_ids():
    result=parse([{'task':'a'}])
    assert result.chat_id is None and result.conversation_id is None


def test_failure_after_chat_created_preserves_trace(tmp_path):
    def handler(req):
        if req.method=='POST':return httpx.Response(200,json={'data':{'id':'failed-chat','conversation_id':'failed-conversation','status':'in_progress'}})
        return httpx.Response(200,json={'code':4101,'msg':'denied'})
    settings=Settings(ai_provider='coze',coze_api_token='FAKE',coze_bot_id='bot',db_path=str(tmp_path/'t.db'))
    ai=CozeClient(settings,poll_interval=0,http=httpx.Client(transport=httpx.MockTransport(handler)))
    c=TestClient(create_app(settings,ai))
    response=c.post('/api/extract',json={'meeting_text':'测试'})
    assert response.status_code==502
    assert response.json()['chat_id']=='failed-chat'
    assert response.json()['conversation_id']=='failed-conversation'
    assert c.get('/api/tasks').json()['tasks']==[]


def test_trace_redacts_synthetic_token(tmp_path,caplog):
    token='SYNTHETIC-SECRET'
    def handler(req):
        if req.method=='POST':return httpx.Response(200,json={'data':{'id':token,'conversation_id':'c-'+token,'status':'completed'}})
        return httpx.Response(200,json={'data':[{'role':'assistant','type':'answer','content':'纪要\n```json\n[]\n```'}]})
    settings=Settings(ai_provider='coze',coze_api_token=token,coze_bot_id='bot',db_path=str(tmp_path/'t.db'))
    ai=CozeClient(settings,http=httpx.Client(transport=httpx.MockTransport(handler)))
    c=TestClient(create_app(settings,ai))
    with caplog.at_level(logging.INFO,logger='meeting_task_board'):
        response=c.post('/api/extract',json={'meeting_text':'测试'})
    assert response.status_code==200 and token not in response.text and token not in caplog.text
