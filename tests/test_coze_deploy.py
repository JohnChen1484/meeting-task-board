import json
import logging
from threading import Event
from time import monotonic

import httpx
import pytest
from fastapi.testclient import TestClient

from app.ai_client import (AIError, AIParseError, AITimeoutError, CozeDeployClient,
                           PROMPT_PREFIX, build_client, extract_deploy_answer, field_structure)
from app.config import Settings, parse_project_id
from app.main import create_app

ANSWER='纪要\n```json\n[{"task":"蓝鲸接口文档","owner":["陈晨","刘洋"],"due_date":"11月10日","priority":"高（原文）"}]\n```'
TOKEN='SYNTHETIC-DEPLOY-TOKEN'


def event(kind,content=None,name='message'):
    return ('event: '+name+'\ndata: '+json.dumps({'type':kind,'content':content},ensure_ascii=False)+'\n\n').encode()


class ByteStream(httpx.SyncByteStream):
    def __init__(self,chunks):self.chunks=chunks;self.closed=False
    def __iter__(self):yield from self.chunks
    def close(self):self.closed=True


def settings(**overrides):
    return Settings(**dict({'ai_provider':'coze_deploy','coze_api_token':TOKEN,'coze_deploy_url':'https://deploy.example.invalid','coze_project_id':123,'db_path':':memory:'},**overrides))


def client_for(chunks,status=200,timeout=2):
    requests=[]
    stream=ByteStream(chunks)
    def handler(req):
        requests.append(req)
        return httpx.Response(status,headers={'content-type':'text/event-stream'},stream=stream)
    client=CozeDeployClient(settings(coze_timeout_seconds=timeout),http=httpx.Client(transport=httpx.MockTransport(handler)))
    return client,requests,stream


def test_multisegment_sse_request_normalization_and_session_uniqueness():
    requests=[]
    wire=event('message_start')+event('tool_request',{'tool_request':{'secret_tool_value':'忽略'}})
    wire+=event('answer',{'answer':ANSWER[:25]})+event('answer',{'text':ANSWER[25:60]})+event('answer',ANSWER[60:])+event('message_end')
    def handler(req):
        requests.append(req)
        # 刻意切开中文 UTF-8 字符和 SSE 行。
        return httpx.Response(200,stream=ByteStream([wire[i:i+3] for i in range(0,len(wire),3)]))
    c=CozeDeployClient(settings(),http=httpx.Client(transport=httpx.MockTransport(handler)))
    first=c.extract('会议原文')
    second=c.extract('另一会议原文')
    assert first.session_id!=second.session_id
    assert first.session_id.startswith('mtb-') and len(first.session_id)==36
    assert first.tasks[0]['owner']=='陈晨、刘洋'
    assert first.tasks[0]['due_date']=='未明确' and first.tasks[0]['priority']=='高' and first.warnings
    assert first.chat_id is None and first.conversation_id is None
    for req,text in zip(requests,['会议原文','另一会议原文']):
        assert str(req.url)=='https://deploy.example.invalid/stream_run' and req.method=='POST'
        assert req.headers['authorization']=='Bearer '+TOKEN
        assert req.headers['content-type']=='application/json'
        body=json.loads(req.content)
        assert type(body['project_id']) is int and body['project_id']==123
        assert body['type']=='query' and body['session_id'].startswith('mtb-')
        assert body['content']['query']['prompt']==[{'type':'text','content':{'text':PROMPT_PREFIX+text}}]
    assert json.loads(requests[0].content)['session_id']!=json.loads(requests[1].content)['session_id']


@pytest.mark.parametrize('content,expected',[({'answer':'a','text':'b'},'a'),({'answer':5,'text':'b'},'b'),({'text':'c'},'c'),('d','d'),({'answer':''},'')])
def test_central_answer_extraction(content,expected):
    assert extract_deploy_answer({'type':'answer','content':content})==expected


def test_unknown_answer_logs_only_keys_and_returns_warning(caplog):
    c,_,_=client_for([event('answer',{'delta':{'private':'NEVER-LOG-VALUE'}})+event('answer',{'text':ANSWER})+event('message_end')])
    with caplog.at_level(logging.WARNING,logger='meeting_task_board'):
        result=c.extract('x')
    assert 'NEVER-LOG-VALUE' not in caplog.text
    assert 'delta' in caplog.text and 'private' in caplog.text
    assert result.warnings and any('未识别' in w for w in result.warnings)
    assert field_structure({'a':'private','b':[{'c':123}]})=={'a':None,'b':[{'c':None}]}


def test_message_end_closes_stream_without_consuming_tail():
    def chunks():
        yield event('answer',ANSWER)+event('message_end')
        pytest.fail('message_end 后不能继续读取')
    c,_,stream=client_for(chunks())
    assert c.extract('x').tasks and stream.closed


@pytest.mark.parametrize('tail',[b'',b'event: message_end\n\n'])
def test_eof_and_named_message_end(tail):
    c,_,stream=client_for([event('answer',ANSWER)+tail])
    assert c.extract('x').tasks and stream.closed


def test_multiline_json_comments_and_unterminated_final_event():
    raw=b': keepalive\r\nevent: message\r\ndata: {"type":"answer",\r\ndata: "content":"ok"}\r\n\r\n'
    c,_,_=client_for([raw])
    assert c.stream_run('full text').answer=='ok'
    c,_,_=client_for([b'event: message\ndata: {"type":"answer","content":"eof"}'])
    assert c.stream_run('text').answer=='eof'


@pytest.mark.parametrize('wire',[event('error',{'error':TOKEN+' '+('很长'*300)}), b'data: not-json\n\n', b'data: []\n\n', b'data: {"content":"no type"}\n\n'])
def test_error_or_malformed_event_is_aierror(wire):
    c,_,stream=client_for([wire])
    with pytest.raises(AIError) as caught:c.extract('x')
    assert TOKEN not in str(caught.value) and len(str(caught.value))<=300
    assert caught.value.session_id.startswith('mtb-') and stream.closed


@pytest.mark.parametrize('status',[401,403,500])
def test_http_errors(status):
    c,requests,_=client_for([],status=status)
    with pytest.raises(AIError) as caught:c.extract('x')
    assert len(requests)==1
    if status in (401,403):assert str(caught.value)=='部署服务的 API Token 无效或无权限'
    else:assert 'HTTP 500' in str(caught.value)


def test_read_timeout_is_aitimeout():
    def chunks():raise httpx.ReadTimeout('private '+TOKEN);yield
    c,_,stream=client_for(chunks())
    with pytest.raises(AITimeoutError):c.extract('x')
    assert stream.closed


@pytest.mark.parametrize('phase',['headers','stream'])
def test_overall_timeout_bounds_blocked_request(phase):
    release,finished=Event(),Event()
    def chunks():
        try:
            if phase=='stream':release.wait(2)
            yield event('answer',ANSWER)
        finally:finished.set()
    def handler(req):
        if phase=='headers':
            release.wait(2)
            finished.set()
        return httpx.Response(200,stream=ByteStream(chunks()))
    c=CozeDeployClient(settings(coze_timeout_seconds=0.04),http=httpx.Client(transport=httpx.MockTransport(handler)))
    start=monotonic()
    try:
        with pytest.raises(AITimeoutError):c.extract('x')
        assert monotonic()-start<0.5
    finally:
        release.set()
        assert finished.wait(1)


def test_unknown_answer_does_not_fallback():
    c,_,_=client_for([event('answer',{'new_unknown_field':'不猜测'})])
    with pytest.raises(AIParseError):c.extract('x')


def test_health_api_error_and_preview_save(tmp_path):
    c,_,_=client_for([event('answer',ANSWER)+event('message_end')])
    app=create_app(settings(db_path=str(tmp_path/'deploy.db')),c)
    api=TestClient(app)
    health=api.get('/api/health').json()
    assert health['ai_provider']=='coze_deploy' and health['ai_configured'] is True
    assert TOKEN not in json.dumps(health)
    result=api.post('/api/extract',json={'meeting_text':'合成输入'}).json()
    assert result['session_id'].startswith('mtb-') and result['warnings']
    assert api.get('/api/tasks').json()['tasks']==[]
    payload={'extraction_id':result['extraction_id'],'tasks':result['tasks']}
    assert api.post('/api/tasks',json=payload).status_code==200
    assert api.post('/api/tasks',json=payload).status_code==409
    c,_,_=client_for([],status=401)
    app.state.ai=c
    assert api.post('/api/extract',json={'meeting_text':'x'}).status_code==502
    assert len(api.get('/api/tasks').json()['tasks'])==1
    app.state.settings.coze_project_id=None
    assert api.get('/api/health').json()['ai_configured'] is False


@pytest.mark.parametrize('value',['abc','1.5',True,-1])
def test_invalid_project_id(value):
    with pytest.raises(ValueError):parse_project_id(value)


def test_config_and_provider_factory(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER','coze_deploy')
    monkeypatch.setenv('COZE_API_TOKEN',TOKEN)
    monkeypatch.setenv('COZE_PROJECT_ID','12345678901234567890')
    monkeypatch.setenv('COZE_DEPLOY_URL','https://deploy.example.invalid/')
    cfg=Settings.from_env()
    assert cfg.coze_project_id==12345678901234567890
    c=build_client(cfg)
    assert isinstance(c,CozeDeployClient)
    c._http.close()


@pytest.mark.parametrize('overrides',[{'coze_api_token':''},{'coze_project_id':None},{'coze_deploy_url':''},{'coze_timeout_seconds':float('nan')},{'coze_deploy_url':'https://deploy.example.invalid/stream_run'}])
def test_deploy_configuration_required(overrides):
    with pytest.raises(ValueError):CozeDeployClient(settings(**overrides))
