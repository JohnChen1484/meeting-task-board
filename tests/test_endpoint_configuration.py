"""仅使用模拟传输验证配置路由，不验证或猜测新版部署协议。"""
import httpx
import pytest

from app.ai_client import CozeClient, PROMPT_PREFIX
from app.config import Settings
from scripts import eval_coze
from tests.test_eval_coze import OfflineAI


def test_configurable_base_and_all_three_paths(monkeypatch):
    values={
        'COZE_API_BASE':'https://deployment.example.invalid/prefix/',
        'COZE_CHAT_PATH':'/configured/chat',
        'COZE_RETRIEVE_PATH':'/configured/status',
        'COZE_MESSAGES_PATH':'/configured/messages',
    }
    for key,value in values.items():monkeypatch.setenv(key,value)
    settings=Settings.from_env()
    settings.coze_api_token='SYNTHETIC'
    settings.coze_bot_id='fake-bot'
    captured=[]
    def handler(req):
        captured.append(req)
        assert req.url.host=='deployment.example.invalid'
        if req.method=='POST':
            import json
            body=json.loads(req.content)
            assert body['stream'] is False and body['auto_save_history'] is True
            assert body['bot_id']=='fake-bot'
            assert body['additional_messages'][0]['content']==PROMPT_PREFIX+'测试记录'
            return httpx.Response(200,json={'data':{'id':'c','conversation_id':'v','status':'in_progress'}})
        if req.url.path.endswith('/status'):
            assert req.url.params['chat_id']=='c'
            return httpx.Response(200,json={'data':{'status':'completed'}})
        return httpx.Response(200,json={'data':[{'role':'assistant','type':'answer','content':'纪要\n```json\n[]\n```'}]})
    client=CozeClient(settings,poll_interval=0,http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.extract('测试记录').tasks==[]
    assert [r.url.path for r in captured]==['/prefix/configured/chat','/prefix/configured/status','/prefix/configured/messages']


@pytest.mark.parametrize('path',['https://other.example.invalid/chat','//other.example.invalid/chat','chat','/chat#fragment'])
def test_endpoint_paths_cannot_override_base(path):
    with pytest.raises(ValueError,match='接口路径'):
        CozeClient(Settings(coze_api_token='fake',coze_bot_id='fake',coze_chat_path=path))


def test_eval_accepts_configured_https_base_and_paths_without_real_network(tmp_path,monkeypatch):
    monkeypatch.setenv('AI_PROVIDER','coze')
    monkeypatch.setenv('COZE_API_BASE','https://deployment.example.invalid/prefix')
    monkeypatch.setenv('COZE_CHAT_PATH','/configured/chat')
    monkeypatch.setenv('COZE_RETRIEVE_PATH','/configured/status')
    monkeypatch.setenv('COZE_MESSAGES_PATH','/configured/messages')
    monkeypatch.setenv('COZE_BOT_ID','fake')
    monkeypatch.setenv('COZE_API_TOKEN','SYNTHETIC')
    captured=[]
    def factory(settings):
        captured.append(settings)
        return OfflineAI()
    monkeypatch.setattr('app.ai_client.CozeClient',factory)
    monkeypatch.setattr(eval_coze.time,'sleep',lambda _:None)
    assert eval_coze.main(['--live','--output-dir',str(tmp_path)])==0
    settings=captured[0]
    assert settings.coze_api_base=='https://deployment.example.invalid/prefix'
    assert (settings.coze_chat_path,settings.coze_retrieve_path,settings.coze_messages_path)==('/configured/chat','/configured/status','/configured/messages')
