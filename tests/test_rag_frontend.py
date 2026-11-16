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
class Node {
  constructor() { this.value=''; this.textContent=''; this.innerHTML=''; this.hidden=false; this.disabled=false; this.listeners={}; }
  addEventListener(k,v) { this.listeners[k]=v; }
}
const nodes = {};
const examples = Array.from({length:4}, ()=>new Node());
const document = {
  getElementById: id => nodes[id] || (nodes[id]=new Node()),
  querySelectorAll: selector => examples,
};
let calls=[], code=200, networkError=false, invalidJson=false, resolveFetch=null, hold=false;
let responseData={answer:'婚假 **3 天**。\n下一行', elapsed_seconds:1.25};
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


def test_busy_disables_all_buttons_and_prevents_duplicate_submission(js):
    js.eval("hold=true;nodes.question.value='问题';askQuestion();")
    assert js.eval("nodes.queryMsg.textContent") == "正在查询知识库…"
    assert js.eval("nodes.btnAsk.disabled && nodes.question.disabled && examples.every(b=>b.disabled)")
    js.eval("askQuestion();examples[0].listeners.click();")
    assert js.eval("calls.length") == 1
    js.eval("resolveFetch();")
    while js.execute_pending_job():
        pass
    assert js.eval("!querying && !nodes.btnAsk.disabled && !nodes.question.disabled && examples.every(b=>!b.disabled)")


@pytest.mark.parametrize("index", range(4))
def test_example_fills_question_and_submits(js, index):
    expected = ["婚假有几天？", "我累计工作满 3 年，在本公司工作满 2 年，现在每年一共有几天年假？",
                "听说青禾会议助手所有版本都永久免费，是真的吗？", "公司的股票代码是多少？"]
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


def test_form_prevents_navigation_and_submits(js):
    js.eval("nodes.question.value='问题';globalThis.prevented=false;")
    run(js, "nodes.askForm.listeners.submit({preventDefault:()=>{prevented=true;}});")
    assert js.eval("prevented && calls.length===1")
