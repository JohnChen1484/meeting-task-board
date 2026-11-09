"""Coze 畸形响应、跨阶段截止时间、错误脱敏回归。"""
import logging
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import app.ai_client as ai_module
from app.ai_client import AIError, AITimeoutError, CozeClient, PROMPT_PREFIX
from app.config import Settings
from app.main import create_app

TOKEN = 'FAKE-SENSITIVE-TOKEN'
CHAT = {'code':0,'data':{'id':'c','conversation_id':'v','status':'completed'}}
MESSAGES = {'code':0,'data':[{'role':'assistant','type':'answer','content':'纪要\n```json\n[]\n```'}]}


def build(handler, timeout=5, interval=0):
    settings = Settings(ai_provider='coze',coze_api_token=TOKEN,coze_bot_id='test',coze_timeout_seconds=timeout,db_path=':memory:')
    coze = CozeClient(settings, poll_interval=interval, http=httpx.Client(transport=httpx.MockTransport(handler)))
    return settings, coze


BAD_CHATS = [None, [], {}, {'data':None}, {'data':[]}, {'data':{}}, {'data':{'status':'completed'}}, {'data':{'id':1,'conversation_id':'v','status':'completed'}}, {'data':{'id':'c','status':'completed'}}, {'data':{'id':'c','conversation_id':[], 'status':'completed'}}, {'data':{'id':'c','conversation_id':'v'}}, {'data':{'id':'c','conversation_id':'v','status':1}}, {'data':{'id':'','conversation_id':'v','status':'completed'}}, {'data':{'id':'c','conversation_id':'v','status':'unknown'}}, {'code':'0'}, {'code':False}]
BAD_POLLS = [{}, {'data':None}, {'data':[]}, {'data':{}}, {'data':{'status':None}}, {'data':{'status':[]}}, {'data':{'status':'failed','last_error':'bad'}}]
BAD_POLLS += [
    {'data':{'status':'completed','id':123}},
    {'data':{'status':'completed','conversation_id':[]}},
    {'data':{'status':'failed','last_error':{'msg':123}}},
]
BAD_MESSAGES = [{}, {'data':None}, {'data':{}}, {'data':[None]}, {'data':[{}]}, {'data':[{'role':'assistant','type':'answer','content':{}}]}, {'data':[{'role':1,'type':'answer','content':'x'}]}, {'data':[{'role':'assistant','type':1,'content':'x'}]}]


@pytest.mark.parametrize('stage,bad', [('start',b) for b in BAD_CHATS]+[('poll',b) for b in BAD_POLLS]+[('messages',b) for b in BAD_MESSAGES])
def test_malformed_upstream_returns_502(stage, bad):
    def handler(req):
        if req.url.path == '/v3/chat':
            if stage == 'start':return httpx.Response(200,content=json.dumps(bad))
            if stage == 'poll':return httpx.Response(200,json={'data':dict(CHAT['data'],status='in_progress')})
            return httpx.Response(200,json=CHAT)
        return httpx.Response(200,content=json.dumps(bad))
    settings, coze = build(handler)
    with pytest.raises(AIError):coze.extract('原文')
    client = TestClient(create_app(settings, coze), raise_server_exceptions=False)
    r = client.post('/api/extract',json={'meeting_text':'原文'})
    assert r.status_code == 502
    assert len(r.json()['detail']) <= 300
    assert client.get('/api/health').json()['last_call']['success'] is False
    assert client.get('/api/tasks').json()['tasks'] == []


@pytest.mark.parametrize('response', [httpx.Response(500), httpx.Response(200,text='not JSON')])
def test_http_and_json_errors(response):
    _, coze = build(lambda req: response)
    with pytest.raises(AIError):coze.extract('原文')


@pytest.mark.parametrize('status', ['failed','requires_action','canceled'])
def test_initial_terminal_failure(status):
    _, coze=build(lambda req:httpx.Response(200,json={'data':dict(CHAT['data'],status=status)}))
    with pytest.raises(AIError):coze.extract('原文')


def test_original_text_is_sent():
    text='张三负责修复登录，2026-11-02前完成。'
    def handler(req):
        if req.method=='POST':
            import json
            payload=json.loads(req.content)
            assert payload['additional_messages'][0]['content']==PROMPT_PREFIX+text
            assert payload['stream'] is False and payload['auto_save_history'] is True
            return httpx.Response(200,json=CHAT)
        return httpx.Response(200,json=MESSAGES)
    assert build(handler)[1].extract(text).tasks==[]


class Clock:
    def __init__(self):self.now=0;self.sleeps=[]
    def monotonic(self):return self.now
    def sleep(self, seconds):self.sleeps.append(seconds);self.now+=seconds


@pytest.mark.parametrize('slow_stage', ['start','poll','messages'])
def test_deadline_in_every_stage(monkeypatch, slow_stage):
    clock=Clock()
    monkeypatch.setattr(ai_module,'time',clock)
    seen=[]
    def handler(req):
        stage={'/v3/chat':'start','/v3/chat/retrieve':'poll','/v3/chat/message/list':'messages'}[req.url.path]
        seen.append(stage)
        assert all(t<=5 and t>0 for t in req.extensions['timeout'].values())
        if stage==slow_stage:clock.now+=6
        if stage=='start':return httpx.Response(200,json={'data':dict(CHAT['data'],status='in_progress' if slow_stage=='poll' else 'completed')})
        if stage=='poll':return httpx.Response(200,json={'data':{'status':'completed'}})
        return httpx.Response(200,json=MESSAGES)
    settings, coze=build(handler)
    c=TestClient(create_app(settings,coze))
    assert c.post('/api/extract',json={'meeting_text':'原文'}).status_code==504
    assert seen[-1]==slow_stage


def test_per_request_budget_decreases_and_caps_at_20(monkeypatch):
    clock=Clock()
    monkeypatch.setattr(ai_module,'time',clock)
    budgets=[]
    def handler(req):
        budgets.append(req.extensions['timeout']['read'])
        if req.method=='POST':
            clock.now+=18
            return httpx.Response(200,json={'data':dict(CHAT['data'],status='in_progress')})
        if req.url.path.endswith('retrieve'):
            clock.now+=4
            return httpx.Response(200,json={'data':{'status':'completed'}})
        return httpx.Response(200,json=MESSAGES)
    assert build(handler,timeout=30,interval=1)[1].extract('原文').tasks==[]
    assert budgets==[20,11,7]


def test_sleep_cannot_exceed_deadline(monkeypatch):
    clock=Clock()
    monkeypatch.setattr(ai_module,'time',clock)
    seen=[]
    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200,json={'data':dict(CHAT['data'],status='in_progress')})
    with pytest.raises(AITimeoutError):build(handler,timeout=0.25,interval=10)[1].extract('原文')
    assert clock.sleeps==[0.25] and seen==['/v3/chat']


def test_no_request_when_budget_exhausted(monkeypatch):
    clock=Clock()
    monkeypatch.setattr(ai_module,'time',clock)
    def handler(req):pytest.fail('不应发起请求')
    with pytest.raises(AITimeoutError):build(handler,timeout=0)[1].extract('原文')


@pytest.mark.parametrize('exception', [httpx.ReadTimeout('timeout'),httpx.ConnectError('network')])
def test_transport_failures(exception):
    def handler(req):raise exception
    _, coze=build(handler)
    expected=AITimeoutError if isinstance(exception,httpx.TimeoutException) else AIError
    with pytest.raises(expected):coze.extract('原文')


@pytest.mark.parametrize('stage', ['start','poll','messages','failed'])
def test_error_redaction_in_response_health_and_file_log(stage,tmp_path,caplog):
    message='泄漏 '+TOKEN+' '+('长错误'*160)+TOKEN
    def handler(req):
        if req.method=='POST' and stage!='start':
            return httpx.Response(200,json={'data':dict(CHAT['data'],status='in_progress' if stage in ('poll','failed') else 'completed')})
        if stage=='failed':return httpx.Response(200,json={'data':{'status':'failed','last_error':{'msg':message}}})
        return httpx.Response(200,json={'code':4101,'msg':message})
    settings,coze=build(handler)
    log_path=tmp_path/'errors.log'
    handler_file=logging.FileHandler(log_path,encoding='utf-8')
    root=logging.getLogger('meeting_task_board')
    root.addHandler(handler_file)
    try:
        with caplog.at_level(logging.INFO,logger='meeting_task_board'):
            client=TestClient(create_app(settings,coze))
            r=client.post('/api/extract',json={'meeting_text':'原文'})
        handler_file.flush()
        assert r.status_code==502 and TOKEN not in r.text
        assert len(r.json()['detail'])<=300
        assert TOKEN not in client.get('/api/health').text
        assert TOKEN not in caplog.text and TOKEN not in log_path.read_text()
        assert all(len(record.getMessage())<=300 for record in caplog.records if record.name.startswith('meeting_task_board'))
    finally:
        root.removeHandler(handler_file)
        handler_file.close()


def test_unexpected_client_error_also_sanitized(caplog):
    class Broken:
        def extract(self,text):raise RuntimeError(TOKEN*100)
    settings=Settings(coze_api_token=TOKEN,db_path=':memory:')
    c=TestClient(create_app(settings,Broken()))
    r=c.post('/api/extract',json={'meeting_text':'x'})
    assert r.status_code==502 and TOKEN not in r.text and TOKEN not in caplog.text
    assert len(r.json()['detail'])<=300


def test_validation_errors_do_not_echo_sensitive_input():
    settings=Settings(coze_api_token=TOKEN,db_path=':memory:')
    c=TestClient(create_app(settings))
    r=c.post('/api/tasks',json={'extraction_id':TOKEN,'tasks':[{'task':'x','priority':TOKEN}]})
    assert r.status_code==422 and TOKEN not in r.text


@pytest.mark.parametrize('slow_stage', ['start','poll','messages'])
def test_blocked_request_cannot_hold_caller_past_deadline(slow_stage):
    from threading import Event
    from time import monotonic
    release = Event()
    finished = Event()
    def handler(req):
        stage = {'/v3/chat':'start','/v3/chat/retrieve':'poll','/v3/chat/message/list':'messages'}[req.url.path]
        if stage == slow_stage:
            try:
                release.wait(2)
            finally:
                finished.set()
        if stage == 'start':
            return httpx.Response(200,json={'data':dict(CHAT['data'],status='in_progress' if slow_stage=='poll' else 'completed')})
        if stage == 'poll':return httpx.Response(200,json={'data':{'status':'completed'}})
        return httpx.Response(200,json=MESSAGES)
    _, coze = build(handler,timeout=0.05)
    started = monotonic()
    try:
        with pytest.raises(AITimeoutError):coze.extract('原文')
        assert monotonic() - started < 0.5
    finally:
        release.set()
        assert finished.wait(1)
