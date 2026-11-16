"""通过 QuickJS 执行问答页实际脚本，模拟 DOM/fetch。"""
import json
from pathlib import Path
import re

import pytest
import quickjs

from tests.test_frontend import run

HTML = (Path(__file__).resolve().parents[1] / "static/rag.html").read_text()
SCRIPT = re.search(r"<script>(.*?)</script>", HTML, re.S).group(1)
EXAMPLES = re.findall(r'<button type="button" class="example">(.*?)</button>', HTML)
BOOT = r'''
let htmlWrites=[];
class Node {
  constructor(name='') {
    this.name=name; this.value=''; this._text=''; this._html=''; this.children=[];
    this.hidden=false; this.disabled=false; this.listeners={}; this.focused=false;
  }
  get textContent() { return this._text + this.children.map(child=>child.textContent).join(''); }
  set textContent(value) { this._text=String(value); this._html=''; this.children=[]; }
  get innerHTML() { return this._html; }
  set innerHTML(value) { this._html=value; this._text=''; this.children=[]; htmlWrites.push({target:this.name,value}); }
  addEventListener(k,v) { this.listeners[k]=v; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children=[]; this._text=''; this._html=''; }
  focus() { this.focused=true; this.disabledAtFocus=this.disabled; }
}
const nodes = {};
const examples = Array.from({length:5}, ()=>new Node());
const document = {
  getElementById: id => nodes[id] || (nodes[id]=new Node(id)),
  querySelectorAll: selector => examples,
  createElement: tag => new Node(tag),
};
let intervals=new Map(), nextInterval=1, clearedIntervals=[];
function setInterval(callback, delay) { const id=nextInterval++; intervals.set(id,{callback,delay}); return id; }
function clearInterval(id) { intervals.delete(id); clearedIntervals.push(id); }
function tick() { Array.from(intervals.values()).forEach(timer=>timer.callback()); }
let calls=[], code=200, networkError=false, invalidJson=false, resolveFetch=null, hold=false;
let responseData={answer:'婚假 **3 天**。\n下一行', elapsed_seconds:1.25, status:'answered',
  structured:true, sources:[], rewritten_query:'', intent:'knowledge'};
async function fetch(url, options) {
  calls.push({url, method:options.method, body:JSON.parse(options.body)});
  if(hold) await new Promise(resolve=>{resolveFetch=resolve;});
  if(networkError) throw new TypeError('Failed to fetch');
  return {ok:code===200, json:async()=>{if(invalidJson)throw new Error('JSON');return responseData;}};
}
'''


@pytest.fixture
def js():
    ctx = quickjs.Context()
    ctx.eval(BOOT)
    ctx.eval(SCRIPT)
    for element_id in re.findall(r'id="([^"]+)"', HTML):
        ctx.eval("document.getElementById(" + json.dumps(element_id) + ");")
    ctx.eval("examples.forEach((b,i)=>{b.textContent=" + json.dumps(EXAMPLES, ensure_ascii=False) + "[i];});")
    return ctx


def test_formatting_escapes_html_before_bold_preserves_lines(js):
    text = '**<img src=x onerror="alert(1)">** & \'内容\'\n下一行'
    formatted = js.eval("formatAnswer(" + json.dumps(text) + ")")
    assert formatted == '<strong>&lt;img src=x onerror=&quot;alert(1)&quot;&gt;</strong> &amp; &#39;内容&#39;\n下一行'
    assert "#answer { white-space: pre-wrap;" in HTML


def test_normal_query_displays_answer_and_elapsed(js):
    js.eval("nodes.question.value='婚假有几天？';")
    run(js, "await askQuestion();")
    assert json.loads(js.eval("JSON.stringify(calls)")) == [
        {"url": "/api/rag/ask", "method": "POST", "body": {"question": "婚假有几天？"}}]
    assert js.eval("nodes.answer.innerHTML") == "婚假 <strong>3 天</strong>。\n下一行"
    assert js.eval("nodes.elapsed.textContent") == "耗时：1.25 秒"
    assert js.eval("nodes.answerPanel.hidden") is False
    assert js.eval("nodes.queryMsg.hidden") is True
    assert js.eval("nodes.btnAsk.disabled") is False
    assert js.eval("nodes.statusBadge.textContent") == "已回答"
    assert js.eval("nodes.statusBadge.hidden || !nodes.sourcesPanel.hidden || !nodes.rewrittenQuestion.hidden") is False
    assert js.eval("intervals.size") == 0


def test_busy_disables_all_buttons_and_prevents_duplicate_submission(js):
    js.eval("hold=true;nodes.question.value='问题';askQuestion();")
    assert js.eval("nodes.queryMsg.textContent") == "正在检索并核对答案，通常需要 5–25 秒… 已等待 0 秒"
    assert js.eval("nodes.btnAsk.disabled && nodes.question.disabled && examples.every(b=>b.disabled)")
    assert js.eval("intervals.size === 1 && Array.from(intervals.values())[0].delay === 1000")
    js.eval("tick();tick();")
    assert js.eval("nodes.queryMsg.textContent").endswith("已等待 2 秒")
    js.eval("askQuestion();examples[0].listeners.click();")
    assert js.eval("calls.length") == 1
    js.eval("resolveFetch();")
    while js.execute_pending_job():
        pass
    assert js.eval("!querying && !nodes.btnAsk.disabled && !nodes.question.disabled && examples.every(b=>!b.disabled)")
    assert js.eval("intervals.size === 0 && clearedIntervals.length === 1")
    js.eval("tick();")
    assert js.eval("nodes.queryMsg.hidden")


@pytest.mark.parametrize("index", range(5))
def test_example_fills_question_and_submits(js, index):
    expected = ["婚假有几天？", "上限是多少？", "报销制度里打车每公里最多报多少钱？",
                "公司有哪些假期？", "公司的股票代码是多少？"]
    assert EXAMPLES == expected
    run(js, f"examples[{index}].listeners.click();")
    assert js.eval("nodes.question.value") == expected[index]
    assert js.eval("calls[0].body.question") == expected[index]


@pytest.mark.parametrize("value", ["", " \n", "问" * 501])
def test_invalid_input_never_fetches(js, value):
    js.eval("nodes.question.value=" + json.dumps(value) + ";")
    run(js, "await askQuestion();")
    assert js.eval("calls.length") == 0 and js.eval("nodes.queryMsg.className") == "msg error"


@pytest.mark.parametrize("setup,message", [
    ("code=504;responseData={detail:'知识库查询超时，请稍后重试'};", "知识库查询超时"),
    ("networkError=true;", "无法连接服务"),
    ("invalidJson=true;", "返回内容异常"),
    ("responseData={answer:'',elapsed_seconds:1};", "返回内容异常"),
])
def test_errors_show_message_restore_controls_hide_old_answer(js, setup, message):
    js.eval("nodes.question.value='问题';" + setup)
    run(js, "await askQuestion();")
    assert message in js.eval("nodes.queryMsg.textContent")
    assert js.eval("nodes.queryMsg.className") == "msg error"
    assert js.eval("nodes.answerPanel.hidden && !nodes.btnAsk.disabled && examples.every(b=>!b.disabled)")
    assert js.eval("intervals.size === 0 && clearedIntervals.length === 1")
    js.eval("tick();")
    assert message in js.eval("nodes.queryMsg.textContent")


def test_form_prevents_navigation_and_submits(js):
    js.eval("nodes.question.value='问题';globalThis.prevented=false;")
    run(js, "nodes.askForm.listeners.submit({preventDefault:()=>{prevented=true;}});")
    assert js.eval("prevented && calls.length===1")


@pytest.mark.parametrize("status,label", [
    ("answered", "已回答"), ("partial", "部分回答"), ("not_found", "知识库未找到"),
    ("clarify", "需要补充信息"), ("unverified", "未能核实"), ("chitchat", "非知识库问题"),
    ("refused", "已拒绝"), ("error", "出错"), ("unknown", ""), ("toString", ""),
    ('<img src=x onerror="alert(1)">', ""),
])
def test_status_badges_and_clarification_focus(js, status, label):
    js.eval("nodes.question.value='问题';responseData.status=" + json.dumps(status) + ";")
    run(js, "await askQuestion();")
    assert js.eval("nodes.statusBadge.textContent") == label
    assert js.eval("nodes.statusBadge.hidden") is (not label)
    assert js.eval("nodes.statusBadge.className") == (
        "status-badge warning" if status == "unverified" else "status-badge")
    assert js.eval("nodes.question.focused") is (status == "clarify")
    if status == "clarify":
        assert js.eval("nodes.question.disabledAtFocus") is False


@pytest.mark.parametrize("has_hint", [False, True])
def test_unverified_displays_warning_once_above_answer(js, has_hint):
    hint = "（提示：以下回答中有内容未能在知识库原文中核实，请以原文为准。）"
    answer = (hint + "\n\n" if has_hint else "") + "**请核对原文**。"
    js.eval("nodes.question.value='问题';responseData.status='unverified';responseData.answer="
            + json.dumps(answer) + ";")
    run(js, "await askQuestion();")
    assert js.eval("nodes.verificationHint.hidden") is False
    assert js.eval("nodes.verificationHint.textContent") == hint
    assert js.eval("nodes.answer.innerHTML") == "<strong>请核对原文</strong>。"
    assert HTML.index('id="verificationHint"') < HTML.index('id="answer"')


def test_sources_display_document_section_status_score_and_obsolete_warning(js):
    sources = [
        {"file": "休假制度v2", "section": "第三章", "status": "现行", "score": 0.616},
        {"file": "休假制度v1", "section": "第二章", "status": "已废止", "score": None},
        {"file": "公司简介", "section": "", "status": "", "score": 0},
    ]
    js.eval("nodes.question.value='问题';responseData.sources=" + json.dumps(sources) + ";")
    run(js, "await askQuestion();")
    assert js.eval("nodes.sourcesPanel.hidden") is False
    assert json.loads(js.eval("JSON.stringify(nodes.sources.children.map(item=>item.textContent))")) == [
        "《休假制度v2》 · 第三章 · 现行 · 相似度：0.62",
        "《休假制度v1》 · 第二章 · 已废止",
        "《公司简介》 · 相似度：0.00",
    ]
    assert js.eval("nodes.sources.children[1].children[1].className") == "source-status warning"
    assert js.eval("nodes.sources.children[0].children[1].className") == "source-status"


def test_next_query_clears_sources_warning_and_rewritten_question(js):
    js.eval("nodes.question.value='问题';responseData.status='unverified';"
            "responseData.sources=[{file:'原文',score:0.5}];responseData.rewritten_query='改写问题';")
    run(js, "await askQuestion();")
    assert js.eval("!nodes.sourcesPanel.hidden && !nodes.verificationHint.hidden && !nodes.rewrittenQuestion.hidden")
    js.eval("responseData.status='unknown';responseData.sources=[];responseData.rewritten_query='';")
    run(js, "await askQuestion();")
    assert js.eval("nodes.sourcesPanel.hidden && nodes.verificationHint.hidden && nodes.rewrittenQuestion.hidden && nodes.statusBadge.hidden")
    assert js.eval("nodes.sources.children.length") == 0


@pytest.mark.parametrize("rewritten,shown", [("", False), ("问题", False), ("具体的问题", True)])
def test_rewritten_question_is_only_shown_when_different(js, rewritten, shown):
    js.eval("nodes.question.value='问题';responseData.rewritten_query=" + json.dumps(rewritten) + ";")
    run(js, "await askQuestion();")
    assert js.eval("nodes.rewrittenQuestion.hidden") is (not shown)
    assert js.eval("nodes.rewrittenQuestion.textContent") == ("系统理解的问题：" + rewritten if shown else "")


def test_all_response_content_uses_text_content_or_escaped_body(js):
    attack = '<img src=x onerror="alert(1)"> & <script>alert(2)</script>'
    response = {
        "answer": "**" + attack + "**", "elapsed_seconds": 1, "status": attack,
        "sources": [{"file": attack, "section": attack, "status": attack, "score": None}],
        "rewritten_query": attack,
    }
    js.eval("nodes.question.value='问题';responseData=" + json.dumps(response) + ";")
    run(js, "await askQuestion();")
    assert js.eval("nodes.sources.children[0].textContent") == f"《{attack}》 · {attack} · {attack}"
    assert js.eval("nodes.rewrittenQuestion.textContent") == "系统理解的问题：" + attack
    assert js.eval("nodes.statusBadge.hidden") is True
    writes = json.loads(js.eval("JSON.stringify(htmlWrites)"))
    assert len(writes) == 1 and writes[0]["target"] == "answer"
    assert "<img" not in writes[0]["value"] and "<script" not in writes[0]["value"]
    assert writes[0]["value"].startswith("<strong>&lt;img")
    assert re.findall(r"\.innerHTML\s*=\s*([^;]+);", SCRIPT) == ["formatAnswer(answer)"]


def test_error_details_are_written_as_text(js):
    attack = '<img src=x onerror="alert(1)">'
    js.eval("nodes.question.value='问题';code=502;responseData=" + json.dumps({"detail": attack}) + ";")
    run(js, "await askQuestion();")
    assert js.eval("nodes.queryMsg.textContent") == attack
    assert json.loads(js.eval("JSON.stringify(htmlWrites)")) == []
    assert js.eval("intervals.size") == 0
