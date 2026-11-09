import json
from pathlib import Path

import httpx
import pytest

import app.ai_client as ai
from scripts import probe_deploy, eval_coze
from tests.test_coze_deploy import event, ByteStream, TOKEN, ANSWER
from tests.test_eval_coze import fixture_tasks


def mock_factory(monkeypatch,wire,status=200):
    original=ai.CozeDeployClient
    requests=[]
    configs=[]
    def handler(req):
        requests.append(req)
        return httpx.Response(status,stream=ByteStream([wire]))
    def factory(cfg):
        configs.append(cfg)
        return original(cfg,http=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(ai,'CozeDeployClient',factory)
    return requests,configs


def config(monkeypatch):
    monkeypatch.setenv('COZE_API_TOKEN',TOKEN)
    monkeypatch.setenv('COZE_DEPLOY_URL','https://deploy.example.invalid')
    monkeypatch.setenv('COZE_PROJECT_ID','123')


def test_probe_default_never_calls_or_prompts(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(ai,'CozeDeployClient',lambda *_:pytest.fail('禁止真实客户端'))
    monkeypatch.setattr(probe_deploy.getpass,'getpass',lambda _:pytest.fail('不能询问令牌'))
    assert probe_deploy.main(['--output-dir',str(tmp_path/'unused')])==0
    assert not (tmp_path/'unused').exists()
    assert '未发起调用' in capsys.readouterr().out


def test_probe_raw_order_counts_keys_and_preview(tmp_path,monkeypatch,capsys):
    config(monkeypatch)
    wire=event('message_start',{'id':'PRIVATE-START'})
    wire+=event('tool_request',{'tool_request':{'name':'PRIVATE-TOOL'}})
    wire+=event('answer',{'answer':ANSWER[:35]})+event('answer',{'text':ANSWER[35:]})
    wire+=event('message_end')
    requests,_=mock_factory(monkeypatch,wire)
    assert probe_deploy.main(['--live','--output-dir',str(tmp_path)])==0
    assert len(requests)==1
    body=json.loads(requests[0].content)
    assert body['content']['query']['prompt'][0]['content']['text']==probe_deploy.PROBE_TEXT
    raw=next(tmp_path.glob('probe_raw_????????_????.txt')).read_text()
    assert raw==wire.decode() and TOKEN not in raw
    output=capsys.readouterr()
    assert 'HTTP 状态码：200' in output.out and '"answer": 2' in output.out
    assert 'tool_request' in output.out and 'name' in output.out
    assert 'PRIVATE-START' not in output.out and 'PRIVATE-TOOL' not in output.out
    assert ANSWER[:20] in output.out and TOKEN not in output.out+output.err


def test_probe_parameter_overrides_and_hidden_token(tmp_path,monkeypatch):
    monkeypatch.setenv('COZE_DEPLOY_URL','invalid-env')
    monkeypatch.setenv('COZE_PROJECT_ID','invalid-env')
    monkeypatch.setenv('COZE_API_TOKEN','')
    monkeypatch.setattr(probe_deploy.getpass,'getpass',lambda _:TOKEN)
    requests,cfgs=mock_factory(monkeypatch,event('message_end'))
    assert probe_deploy.main(['--live','--url','https://override.example.invalid','--project-id','456','--output-dir',str(tmp_path)])==0
    assert len(requests)==1 and json.loads(requests[0].content)['project_id']==456
    assert cfgs[0].coze_deploy_url=='https://override.example.invalid'


@pytest.mark.parametrize('escaped',[False,True])
def test_probe_blocks_token_before_writing(tmp_path,monkeypatch,capsys,escaped):
    config(monkeypatch)
    wire=event('answer',{'answer':TOKEN})
    if escaped:wire=wire.replace(TOKEN.encode(),b'\\u0053'+TOKEN[1:].encode())
    requests,_=mock_factory(monkeypatch,wire)
    assert probe_deploy.main(['--live','--output-dir',str(tmp_path)])==1
    assert len(requests)==1
    raw=next(tmp_path.glob('*.txt')).read_text()
    assert TOKEN not in raw and '未保存' in raw and 'data:' not in raw
    output=capsys.readouterr()
    assert TOKEN not in output.out+output.err


def test_probe_unknown_answer_fields_are_recorded_without_value_on_terminal(tmp_path,monkeypatch,capsys):
    config(monkeypatch)
    wire=event('answer',{'unconfirmed':{'delta':'PRIVATE-UNKNOWN'}})+event('message_end')
    mock_factory(monkeypatch,wire)
    assert probe_deploy.main(['--live','--output-dir',str(tmp_path)])==0
    output=capsys.readouterr()
    assert 'unconfirmed' in output.out and 'delta' in output.out
    assert 'PRIVATE-UNKNOWN' not in output.out+output.err
    assert 'PRIVATE-UNKNOWN' in next(tmp_path.glob('*.txt')).read_text()


def test_probe_error_keeps_partial_answer_and_calls_once(tmp_path,monkeypatch,capsys):
    config(monkeypatch)
    mock_factory(monkeypatch,event('answer',{'text':'partial-answer'})+event('error',{'message':'failed'}))
    assert probe_deploy.main(['--live','--output-dir',str(tmp_path)])==1
    output=capsys.readouterr()
    assert 'partial-answer' in output.out and '"error": 1' in output.out


def test_probe_401_shows_status_and_token_hint(tmp_path,monkeypatch,capsys):
    config(monkeypatch)
    requests,_=mock_factory(monkeypatch,b'',status=401)
    assert probe_deploy.main(['--live','--output-dir',str(tmp_path)])==1
    output=capsys.readouterr()
    assert 'HTTP 状态码：401' in output.out and 'API Token 无效或无权限' in output.err
    assert len(requests)==1


def test_probe_report_collision_prevents_call(tmp_path,monkeypatch):
    config(monkeypatch)
    requests,_=mock_factory(monkeypatch,event('message_end'))
    args=['--live','--output-dir',str(tmp_path)]
    assert probe_deploy.main(args)==0
    assert probe_deploy.main(args)==2
    assert len(requests)==1


@pytest.mark.parametrize('flag',[True,False])
def test_eval_deploy_uses_existing_client_with_mock_sse(tmp_path,monkeypatch,flag):
    config(monkeypatch)
    monkeypatch.setenv('AI_PROVIDER','mock' if flag else 'coze_deploy')
    monkeypatch.setenv('COZE_BOT_ID','')
    original=ai.CozeDeployClient
    requests=[]
    sleeps=[]
    replies=fixture_tasks()
    def handler(req):
        index=len(requests)
        requests.append(req)
        answer='纪要\n```json\n'+json.dumps(replies[index],ensure_ascii=False)+'\n```'
        return httpx.Response(200,stream=ByteStream([event('answer',{'text':answer})+event('message_end')]))
    monkeypatch.setattr(ai,'CozeDeployClient',lambda cfg:original(cfg,http=httpx.Client(transport=httpx.MockTransport(handler))))
    monkeypatch.setattr(eval_coze.time,'sleep',sleeps.append)
    args=['--live','--output-dir',str(tmp_path)]
    if flag:args+=['--provider','coze_deploy']
    assert eval_coze.main(args)==0
    assert len(requests)==8 and sleeps==[2]*7
    assert len({json.loads(r.content)['session_id'] for r in requests})==8
    report=next(tmp_path.glob('*.md')).read_text()
    assert '31/31（100.0%）' in report and 'coze_deploy' in report and 'session_id：mtb-' in report


def test_eval_rejects_live_mock(tmp_path,monkeypatch):
    monkeypatch.setenv('AI_PROVIDER','mock')
    monkeypatch.setattr(ai,'MockClient',lambda *_:pytest.fail('不得构建 Mock'))
    assert eval_coze.main(['--live','--output-dir',str(tmp_path)])==2
