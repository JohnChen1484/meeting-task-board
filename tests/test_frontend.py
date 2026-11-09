"""执行真实页面脚本，使用轻量 DOM/fetch 替身检查页面状态和请求。"""
import json
from pathlib import Path
import re

import pytest
import quickjs

HTML = (Path(__file__).resolve().parents[1]/'static/index.html').read_text()
SCRIPT = re.search(r'<script>(.*?)</script>', HTML, re.S).group(1)
BOOT = r'''
class Node {
  constructor(tag='div') { this.tag=tag; this.children=[]; this.style={}; this.value=''; this.textContent=''; this.title=''; this.className=''; this.hidden=false; this.disabled=false; this.listeners={};
    this.classList={add:()=>{},toggle:()=>{}};
  }
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(k,v){this[k]=v;}
  addEventListener(k,v){this.listeners[k]=v;}
}
const nodes={};
const document={getElementById:(id)=>nodes[id]||(nodes[id]=new Node()),createElement:(tag)=>new Node(tag),createTextNode:(v)=>String(v)};
let health={ai_provider:'mock',ai_configured:true,last_call:null};
let extracted={extraction_id:'11111111-1111-4111-8111-111111111111',summary:'纪要',tasks:[{task:'演示任务',owner:'未明确'}],has_tasks:true};
let saveCode=200;
let calls=[];
let tasksOnServer=[];
async function fetch(url, options) {
  calls.push({url,method:options.method,body:options.body ? JSON.parse(options.body):null});
  let data;
  let status=200;
  if(url==='/api/health')data=health;
  else if(url==='/api/extract') { data=extracted; status=extracted.errorCode||200; }
  else if(options.method==='POST') {status=saveCode;data=status===409?{detail:'这批任务已保存过'}:status===422?{detail:'截止日期格式不正确'}:{saved:1};}
  else data={tasks:tasksOnServer};
  return {ok:status<400,status,json:async()=>data};
}
function alert(message){globalThis.alertMessage=message;}
function setTimeout(){}
'''


@pytest.fixture
def js():
    ctx=quickjs.Context()
    ctx.eval(BOOT)
    ctx.eval(SCRIPT.rsplit('init();',1)[0])
    return ctx


def run(ctx,code):
    ctx.eval('globalThis.finished=false; globalThis.failure=null;')
    ctx.eval('(async()=>{'+code+'})().then(()=>{finished=true}).catch(e=>{failure=String(e);finished=true});')
    for _ in range(1000):
        if not ctx.execute_pending_job():break
    assert ctx.eval('finished'), '前端异步操作没有完成'
    assert ctx.eval('failure') is None


def test_warning_is_first_in_preview_and_yellow():
    assert re.search(r'id="previewBox"[^>]*>\s*<div id="mockWarning"',HTML)
    assert '当前为演示模式，任务为固定示例，与输入无关，请勿保存为真实数据' in HTML
    assert '.demo-warning { background: #fff4cc;' in HTML


@pytest.mark.parametrize('provider,visible',[('mock',True),('coze',False)])
def test_mock_banner_visibility(js,provider,visible):
    js.eval(f'health.ai_provider={json.dumps(provider)};')
    run(js,'await refreshProvider();')
    assert js.eval('nodes.mockWarning.hidden') is not visible


@pytest.mark.parametrize('call,label,css',[(None,'已配置，未验证','unverified'),({'success':True,'time':'2026-10-29T09:07:00Z','error':None},'最近调用成功','success'),({'success':False,'time':'2026-10-29T09:07:00Z','error':'上游错误'},'最近调用失败','failure')])
def test_provider_states(js,call,label,css):
    js.eval(f'health.last_call={json.dumps(call,ensure_ascii=False)};')
    run(js,'await refreshProvider();')
    assert js.eval('nodes.provider.textContent').startswith(label)
    assert js.eval('nodes.provider.className')=='badge '+css
    if css=='success':assert re.fullmatch(r'最近调用成功 \d{2}:\d{2}',js.eval('nodes.provider.textContent'))
    if css=='failure':assert js.eval('nodes.provider.title')=='上游错误'
    # 随后的成功调用必须去掉失败原因。
    js.eval("health.last_call={success:true,time:'2026-10-29T09:07:00Z'};")
    run(js,'await refreshProvider();')
    assert js.eval('nodes.provider.title')==''


def test_extract_save_carries_and_clears_batch_id(js):
    js.eval("nodes.meeting=new Node();nodes.meeting.value='测试会议';")
    run(js,'await extract();')
    assert js.eval('extractionId')=='11111111-1111-4111-8111-111111111111'
    assert js.eval("calls.filter(c=>c.url==='/api/health').length")==1
    # 修改预览中的数据后，保存必须提交修改结果。
    js.eval("previewTasks[0].task='人工核对';")
    run(js,'await save();')
    payload=json.loads(js.eval("JSON.stringify(calls.find(c=>c.url==='/api/tasks'&&c.method==='POST').body)"))
    assert payload['extraction_id']=='11111111-1111-4111-8111-111111111111'
    assert payload['meeting_text']=='测试会议' and payload['tasks'][0]['task']=='人工核对'
    assert js.eval('extractionId') is None
    assert js.eval('nodes.previewBox.style.display')=='none'
    assert js.eval("calls.some(c=>c.url==='/api/tasks'&&c.method==='GET')")
    run(js,'await save();')
    assert js.eval("calls.filter(c=>c.url==='/api/tasks'&&c.method==='POST').length")==1


def test_409_refreshes_board_and_shows_message(js):
    js.eval("extractionId='batch';previewTasks=[{task:'任务'}];saveCode=409;")
    run(js,'await save();')
    assert js.eval('nodes.extractMsg.textContent')=='这批任务已保存过'
    assert js.eval('extractionId') is None
    assert js.eval("calls.some(c=>c.url==='/api/tasks'&&c.method==='GET')")


def test_validation_message_shown_directly_and_retry_retains_id(js):
    js.eval("extractionId='batch';previewTasks=[{task:'任务'}];saveCode=422;")
    run(js,'await save();')
    assert js.eval('nodes.saveMsg.textContent')=='截止日期格式不正确'
    assert js.eval('extractionId')=='batch'
    assert js.eval('nodes.btnSave.disabled') is False


def test_failed_extract_clears_old_batch_and_updates_health(js):
    js.eval("extractionId='old';nodes.meeting=new Node();nodes.meeting.value='记录';extracted={errorCode:502,detail:'上游失败'};health.last_call={success:false,error:'上游失败'};")
    run(js,'await extract();')
    assert js.eval('extractionId') is None
    assert js.eval('nodes.provider.textContent')=='最近调用失败'


def test_no_tasks_preview_hides_save_section(js):
    js.eval("nodes.meeting=new Node();nodes.meeting.value='【演示：非会议】';extracted.tasks=[];extracted.has_tasks=false;")
    run(js,'await extract();')
    assert js.eval('nodes.taskSection.style.display')=='none'
    assert js.eval('nodes.btnSave.disabled') is True


@pytest.mark.parametrize('button',['btnClear','btnDiscard'])
def test_clear_discard_invalidate_batch(js,button):
    run(js,'await init();')
    js.eval("extractionId='old';previewTasks=[{task:'任务'}];")
    run(js,f'nodes.{button}.onclick();')
    assert js.eval('extractionId') is None
    assert js.eval('previewTasks.length')==0


def test_reload_fetches_persisted_board(js):
    js.eval("tasksOnServer=[{id:1,task:'已保存任务',owner:'李四',due_date:'未明确',priority:'低',status:'待开始'}];")
    run(js,'await init();')
    assert js.eval("calls.some(c=>c.url==='/api/tasks'&&c.method==='GET')")
    assert js.eval('nodes.board.children.length')==4
    # 待开始列：标题和任务卡片。
    assert js.eval('nodes.board.children[1].children[1].children[0].children[0]')=='已保存任务'


def test_preview_warnings_and_chat_id_save(js):
    js.eval("nodes.meeting=new Node();nodes.meeting.value='会议';extracted.chat_id='trace-chat';extracted.warnings=['第2条任务的日期不合法','<script>不可执行</script>'];")
    run(js,'await extract();')
    assert js.eval('nodes.parseWarnings.hidden') is False
    assert js.eval('nodes.parseWarnings.textContent')=='第2条任务的日期不合法\n<script>不可执行</script>'
    run(js,'await save();')
    assert js.eval("calls.find(c=>c.url==='/api/tasks'&&c.method==='POST').body.chat_id")=='trace-chat'
    assert js.eval('lastChatId') is None


def test_new_preview_clears_old_warnings(js):
    js.eval("nodes.meeting=new Node();nodes.meeting.value='会议';extracted.warnings=['旧警告'];")
    run(js,'await extract();')
    js.eval('extracted.warnings=[];')
    run(js,'await extract();')
    assert js.eval('nodes.parseWarnings.hidden') is True
    assert js.eval('nodes.parseWarnings.textContent')==''
