from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ai_client import ExtractResult, parse_answer
from scripts import eval_coze as evaluator

CASES=Path(__file__).parent/'cases/eval_cases.json'


@pytest.fixture(autouse=True)
def eval_provider(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'coze')


def load_cases():return json.loads(CASES.read_text())


def fixture_tasks():
    return [
        [{'task':'完成客户回访','owner':'孙悦','due_date':'2026-11-06'}, {'task':'更新报价单','owner':'未明确'}, {'task':'服务器续费','owner':'老周'}],
        [{'task':'竞品报告','owner':'李娜','due_date':'2026-11-04'}, {'task':'活动方案','owner':'市场部','due_date':'2026-10-31','priority':'高（原文）'}, {'task':'客户反馈','owner':'赵磊'}],
        [{'task':'直播保障','owner':['陈晨','刘洋']}, {'task':'宣传物料','owner':'设计组、市场组'}, {'task':'嘉宾邀请','owner':'未明确'}],
        [{'task':'新版App上线'}, {'task':'接口文档更新'}, {'task':'客服话术','owner':'周婷'}], [], [],
        [{'task':'设计稿','owner':'小王'}, {'task':'测试环境','owner':'阿杰'}, {'task':'财务预算催办'}],
        [{'task':'蓝鲸接口文档','owner':'王一鸣','due_date':'2026-10-30'}, {'task':'采购清单','owner':'何静','due_date':'2026-11-06'}],
    ]


class OfflineAI:
    def __init__(self,responses=None):
        self.responses=responses if responses is not None else fixture_tasks()
        self.calls=[]
        self.closed=False
        self._http=SimpleNamespace(close=self.close)
    def close(self):self.closed=True
    def extract(self,text):
        index=len(self.calls)
        self.calls.append(text)
        data=self.responses[index]
        if isinstance(data,Exception):raise data
        result=parse_answer('纪要\n```json\n'+json.dumps(data,ensure_ascii=False)+'\n```')
        result.chat_id=f'offline-chat-{index+1}'
        result.conversation_id=f'offline-conversation-{index+1}'
        return result


def test_all_eight_cases_offline():
    client=OfflineAI()
    sleeps=[]
    results=evaluator.run_cases(load_cases(),client,sleeps.append)
    assert len(client.calls)==8 and sleeps==[2]*7
    assert sum(len(r['checks']) for r in results)==31
    assert all(c.passed for r in results for c in r['checks'])
    report=evaluator.render_report(results,'离线模拟响应自测（非真实 Coze）')
    assert '31/31（100.0%）' in report and '8/8（100.0%）' in report
    assert 'offline-chat-8' in report and '蓝鲸' in report


@pytest.mark.parametrize('op',['equals','not_equals','contains_all','exists'])
def test_no_match_is_failure_not_vacuous_pass(op):
    rule={'description':'选择器','op':op,'keywords':['不存在'],'field':'owner','value':['陈晨'] if op=='contains_all' else '李四'}
    assert evaluator.evaluate_check([{'task':'真实任务','owner':'李四'}],rule).passed is False


def test_matching_keywords_are_or_and_all_matches_checked():
    rule={'description':'续费/服务器','op':'equals','keywords':['续费','服务器'],'field':'owner','value':'老周'}
    assert evaluator.evaluate_check([{'task':'服务器更新','owner':'老周'}],rule).passed
    assert not evaluator.evaluate_check([{'task':'服务器更新','owner':'老周'},{'task':'续费','owner':'错误人'}],rule).passed


def test_all_equals_requires_at_least_one_task():
    rule={'description':'所有优先级','op':'all_equals','field':'priority','value':'未明确'}
    assert not evaluator.evaluate_check([],rule).passed
    assert evaluator.evaluate_check([{'task':'a','priority':'未明确'}],rule).passed
    assert not evaluator.evaluate_check([{'task':'a','priority':'高'}],rule).passed


def test_not_equals_and_contains_all():
    rule={'description':'活动不作截止','op':'not_equals','field':'due_date','keywords':['直播'],'value':'2026-11-15'}
    assert not evaluator.evaluate_check([{'task':'直播','due_date':'2026-11-15'}],rule).passed
    assert not evaluator.evaluate_check([{'task':'直播'}],rule).passed
    assert evaluator.evaluate_check([{'task':'直播','due_date':'未明确'}],rule).passed
    both={'description':'多人','op':'contains_all','field':'owner','keywords':['直播'],'value':['陈晨','刘洋']}
    assert evaluator.evaluate_check([{'task':'直播','owner':'陈晨、刘洋'}],both).passed
    assert not evaluator.evaluate_check([{'task':'直播','owner':'陈晨'}],both).passed


def test_errors_fail_case_and_do_not_retry():
    cases=load_cases()[:2]
    client=OfflineAI([RuntimeError('权限失败'),fixture_tasks()[1]])
    sleeps=[]
    results=evaluator.run_cases(cases,client,sleeps.append)
    assert len(client.calls)==2 and sleeps==[2]
    assert results[0]['status']=='调用失败' and all(not c.passed for c in results[0]['checks'])
    assert all(c.passed for c in results[1]['checks'])
    assert '失败' in evaluator.render_report(results,'离线')


def test_limit_checked_before_any_calls():
    client=OfflineAI()
    cases=load_cases()+[dict(load_cases()[0],id=i) for i in (9,10,11)]
    with pytest.raises(ValueError,match='1～10'):evaluator.run_cases(cases,client)
    assert client.calls==[]


def test_ten_calls_max_with_two_second_intervals():
    cases=load_cases()+[dict(load_cases()[0],id=i) for i in (9,10)]
    client=OfflineAI(fixture_tasks()+[fixture_tasks()[0]]*2)
    sleeps=[]
    evaluator.run_cases(cases,client,sleeps.append)
    assert len(client.calls)==10 and sleeps==[2]*9


@pytest.mark.parametrize('change', ['duplicate_id','bad_op','empty_checks','empty_keywords','bad_value','bad_field'])
def test_bad_case_configuration_rejected(change):
    cases=load_cases()
    if change=='duplicate_id':cases[1]['id']=cases[0]['id']
    elif change=='empty_checks':cases[0]['checks']=[]
    else:
        rule=cases[0]['checks'][0]
        rule[{'bad_op':'op','empty_keywords':'keywords','bad_value':'value','bad_field':'field'}[change]]={'bad_op':'execute','empty_keywords':[],'bad_value':None,'bad_field':'secret'}[change]
    with pytest.raises(ValueError):evaluator.validate_cases(cases)


def test_dry_run_never_calls_or_prompts(tmp_path,monkeypatch):
    def bomb(*a,**k):pytest.fail('默认离线不能创建 CozeClient/索取令牌')
    monkeypatch.setattr('app.ai_client.CozeClient',bomb)
    monkeypatch.setattr(evaluator.getpass,'getpass',bomb)
    original_get=evaluator.os.environ.get
    def env_get(key,*args):
        if key=='COZE_API_TOKEN':pytest.fail('离线不读取令牌')
        return original_get(key,*args)
    monkeypatch.setattr(evaluator.os.environ,'get',env_get)
    assert evaluator.main(['--output-dir',str(tmp_path)])==0
    report=next(tmp_path.glob('eval_????????_????.md')).read_text()
    assert '未发起调用' in report and '未执行' in report and '100.0%' not in report


@pytest.mark.parametrize('from_env',[True,False])
def test_live_branch_only_uses_fake_client_and_redacts_report(tmp_path,monkeypatch,from_env,capsys):
    token='SYNTHETIC-TOKEN-ONLY'
    client=OfflineAI()
    # 测试 live 开关的控制流，工厂完全替换，不会发起任何网络请求。
    client.responses[0]=RuntimeError(token+' upstream error')
    captured=[]
    def factory(settings):
        captured.append(settings)
        return client
    monkeypatch.setattr('app.ai_client.CozeClient',factory)
    monkeypatch.setenv('COZE_BOT_ID','synthetic-bot')
    monkeypatch.setenv('COZE_API_TOKEN',token if from_env else '')
    if from_env:
        monkeypatch.setattr(evaluator.getpass,'getpass',lambda _:pytest.fail('有环境令牌时不能提示'))
    else:monkeypatch.setattr(evaluator.getpass,'getpass',lambda _:token)
    sleeps=[]
    monkeypatch.setattr(evaluator.time,'sleep',sleeps.append)
    assert evaluator.main(['--live','--output-dir',str(tmp_path)])==1
    assert captured[0].coze_api_token==token and client.closed and sleeps==[2]*7
    output=capsys.readouterr()
    assert token not in output.out+output.err
    report=next(tmp_path.glob('*.md')).read_text()
    assert token not in report and '[已脱敏]' in report


def test_report_redacts_long_token_before_truncating_and_escapes_markdown():
    secret='S'*400
    result=evaluator.run_cases(load_cases()[:1],OfflineAI([RuntimeError(secret+'oops')]),lambda _:None)
    report=evaluator.render_report(result,'离线',secret)
    assert 'S'*20 not in report and '[已脱敏]' in report
    tasks=[{'task':'<script>|危险\n内容','owner':'x','due_date':'未明确'}]
    result[0].update(tasks=tasks,chat_id=secret,warnings=[secret],status='已执行')
    report=evaluator.render_report(result,'离线',secret)
    assert '<script>' not in report and '&lt;script&gt;&#124;' in report
    assert secret not in report


def test_missing_bot_rejected_without_prompt_or_network(tmp_path,monkeypatch):
    monkeypatch.setenv('COZE_BOT_ID','')
    monkeypatch.setattr(evaluator.getpass,'getpass',lambda _:pytest.fail('不应询问令牌'))
    assert evaluator.main(['--live','--output-dir',str(tmp_path)])==2


def test_report_collision_does_not_invoke_client(tmp_path,monkeypatch):
    assert evaluator.main(['--output-dir',str(tmp_path)])==0
    monkeypatch.setattr('app.ai_client.CozeClient',lambda *_:pytest.fail('报告冲突不应创建客户端'))
    assert evaluator.main(['--live','--output-dir',str(tmp_path)])==2


def test_getpass_without_secure_terminal_aborts(tmp_path,monkeypatch):
    import warnings
    monkeypatch.setenv('COZE_BOT_ID','synthetic-bot')
    monkeypatch.setenv('COZE_API_TOKEN','')
    def unsafe_prompt(_):
        warnings.warn('无法关闭终端回显',evaluator.getpass.GetPassWarning)
        pytest.fail('不得回退到可见输入')
    monkeypatch.setattr(evaluator.getpass,'getpass',unsafe_prompt)
    monkeypatch.setattr('app.ai_client.CozeClient',lambda *_:pytest.fail('不得创建真实客户端'))
    assert evaluator.main(['--live','--output-dir',str(tmp_path)])==2


@pytest.mark.parametrize('key,value',[('COZE_API_BASE','http://example.invalid'),('COZE_TIMEOUT_SECONDS','nan'),('COZE_TIMEOUT_SECONDS','0')])
def test_live_config_rejected_before_token_read(tmp_path,monkeypatch,key,value):
    monkeypatch.setenv('COZE_BOT_ID','synthetic-bot')
    monkeypatch.setenv(key,value)
    monkeypatch.setattr(evaluator.getpass,'getpass',lambda _:pytest.fail('配置错误时不得索取令牌'))
    monkeypatch.setattr('app.ai_client.CozeClient',lambda *_:pytest.fail('不得创建客户端'))
    assert evaluator.main(['--live','--output-dir',str(tmp_path)])==2
