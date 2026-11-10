# 会议任务看板（meeting-task-board）

粘贴会议记录 → 调用扣子智能体提取任务 → 人工核对 → 确认后保存 → 在看板上跟踪状态。

属于「智能办公自动化 Agent」项目的程序部分：
- **Coze**：扣子开发平台上的「会议纪要助手-API版」智能体，负责理解会议记录（规则以 [API 版智能体人设](prompts/API版智能体_人设与回复逻辑.md) 为准）
- **本程序**：负责界面、数据存储和流程控制
- AI 调用全部封装在 `app/ai_client.py`，以后要换成其他大模型服务，只改这一个文件

**第二周交付范围**：会议纪要任务看板 + 智能办公助手 V5（在扣子上新建的单智能体，仓库里保存它的人设、多轮走查用例和评测脚本，见「十一」）。知识库问答（RAG）将在第三周合并进来。

## 一、在 Mac 上运行（VS Code）

需要 Python 3.10 及以上（终端输入 `python3 --version` 查看）。

1. 用 VS Code 打开 `meeting-task-board` 文件夹，菜单「终端 → 新建终端」。
2. 创建虚拟环境并安装依赖：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. 创建配置文件：
   ```bash
   cp .env.example .env
   ```
   默认 `AI_PROVIDER=mock`（演示模式，不联网），可以先用它把页面流程跑一遍。
4. 启动：
   ```bash
   uvicorn app.main:app --reload
   ```
5. 浏览器打开 http://127.0.0.1:8000 。接口文档在 http://127.0.0.1:8000/docs 。

停止服务：在终端按 `Ctrl + C`。下次启动只需第 4 步（先执行 `source .venv/bin/activate`）。

## 二、切换到真实的扣子智能体

> `coze` 保留传统 `v3/chat + bot_id` 非流式适配。新版「扣子编程」代码项目部署后使用独立 SSE 服务，应选择 `coze_deploy`，配置见第十节。当前已知请求格式来自部署页面，answer 文本字段仍待真实探测；本轮未运行真实调用。


用 VS Code 打开 `.env`，修改为：

```
AI_PROVIDER=coze
COZE_API_TOKEN=你的个人访问令牌
COZE_BOT_ID=你的智能体ID
COZE_API_BASE=https://api.coze.cn
```

国内使用 `https://api.coze.cn`；国际版使用 `https://api.coze.com`，令牌、Bot 和空间需匹配平台。PAT 必须具备 `chat`、`getChat`、`listMessage` 权限，并授权智能体所在空间。

保存后重启服务（`Ctrl + C` 再执行第 4 步）。页面右上角先显示灰色「已配置，未验证」。完成一次提取后，显示绿色「最近调用成功 HH:MM」或红色「最近调用失败」（悬停查看原因）。调用状态保存在当前服务进程内，重启后重新验证。

> ⚠️ `.env` 里有令牌，已被 `.gitignore` 排除。不要把 `.env` 发给别人、上传到网盘或贴到任何对话里。令牌只在后端使用，不会出现在网页和接口返回中。

## 三、运行测试

```bash
python -m pytest -v
```

测试全部不联网，且 `tests/conftest.py` 会禁用 `.env` 加载，入口使用内存数据库；各用例使用临时或内存数据库：
- `tests/test_parse.py`：从智能体回复中提取 JSON（正常、空数组、无代码块、格式错误等）
- `tests/test_api.py`：接口行为（预览不写库、出错不写库、按负责人自动设置状态、令牌不外泄等）
- `tests/test_coze_client.py` / `tests/test_coze_robustness.py`：模拟扣子调用、畸形响应、整体超时、错误脱敏
- `tests/test_regressions.py`：Mock 约定、字段与状态规则、批次去重、事务回滚、并发及持久化
- `tests/test_frontend.py`：通过 QuickJS 执行页面脚本，验证提示条、连接状态和保存流程（不替代真实浏览器验收）
- `tests/test_eval_multiturn.py`：多轮走查脚本的 SSE 解析、会话复用、报告命名与用例文件一致性（替身代替网络）

## 四、手动验收清单

| # | 操作 | 预期 |
|---|---|---|
| 1 | 粘贴 case 文本，点「提取任务」 | 显示加载状态，完成后出现可编辑的预览表；负责人「未明确」的格子标黄 |
| 2 | 此时刷新页面 | 看板仍为空（预览不会自动保存） |
| 3 | 点「确认保存到看板」 | 负责人明确的任务进「待开始」，「未明确」的进「待分配」 |
| 4 | 在待分配卡片点「编辑」，填负责人后保存 | 卡片自动移到「待开始」 |
| 5 | 用卡片下拉框改状态 | 卡片移动到对应列 |
| 6 | Mock 下输入含「【演示：非会议】」的文本并提取 | 提示没有待办，显示 AI 的说明，不出现任务表 |
| 7 | 把 `.env` 的令牌改错一位，重启后提取 | 页面显示错误信息（含 code），看板数据不变 |

## 五、常见问题

| 现象 | 原因与处理 |
|---|---|
| 报错 `code=4100` / `4101` | 令牌无效、过期（最长 30 天）或缺少对话相关权限；重新创建令牌并勾选对话权限、选对工作空间 |
| 报错「没有找到 ```json 代码块」 | 智能体人设里缺少「程序调用补充规则」，或发布后没有重新发布；检查人设末尾并重新发布 |
| 提取超时 | 模型较慢或网络问题；可把 `.env` 中 `COZE_TIMEOUT_SECONDS` 调大 |
| 启动时报 `COZE_API_TOKEN` 未填写 | `AI_PROVIDER=coze` 但 `.env` 里没填令牌或 bot_id |
| 端口被占用 | 换端口启动：`uvicorn app.main:app --reload --port 8001` |

运行日志在 `logs/app.log`（记录长度、状态及 Coze 的 chat_id/conversation_id，不记录会议原文和令牌）。

## 六、项目结构

```
meeting-task-board/
├── app/
│   ├── config.py       # 读取 .env
│   ├── ai_client.py    # AI 调用：扣子客户端、演示客户端、JSON 解析
│   ├── db.py           # SQLite 任务表
│   └── main.py         # FastAPI 接口 + 页面
├── static/index.html   # 前端单页（原生 JS）
├── prompts/            # 智能体人设：会议纪要 API 版（V4）、智能办公助手 V5
├── scripts/            # 探测、会议纪要评测、多轮走查（eval_multiturn.py）
├── tests/              # 自动化测试 + 评测用例（cases/、multiturn_cases.json）
├── docs/               # 第二周开发方案、第一周评测报告
├── .env.example        # 配置模板
└── requirements.txt
```

## 七、接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/extract | 提取任务预览（不写库） |
| POST | /api/tasks | 确认后批量保存，必须携带 extraction_id；已保存批次返回 409 |
| GET | /api/tasks?status= | 查询任务，可按状态筛选 |
| PATCH | /api/tasks/{id} | 修改状态 / 负责人 / 截止日期 / 优先级 |
| DELETE | /api/tasks/{id} | 删除任务 |
| GET | /api/health | 服务状态（不含令牌） |


Mock 只验证页面流程：除包含 `【演示：非会议】` 的输入返回空数组外，均返回三条固定演示任务，与原文无关。预览区会显示黄色提示条。

`POST /api/extract` 返回 UUID `extraction_id`。确认保存时将它连同 `tasks`、`meeting_text` 传给 `POST /api/tasks`；同一批次仅能保存一次。`saved_batches` 与任务写入使用同一事务；应用下次正常启动时会自动创建该表，不改写既有任务记录。

`GET /api/health` 返回 `ai_provider`、`ai_configured` 和 `last_call`；后者初始为 `null`，调用后为 `{success, time, error}`，时间使用 UTC ISO 格式，前端按本地时间显示。响应不含令牌，错误会脱敏并限制在 300 字内。


## 八、字段规范化与调用溯源

程序保持简短的 JSON 请求提示，语义提取与优先级推断遵循 `docs/API版智能体_人设与回复逻辑.md`，不在代码中复制人设。优先级允许依原文线索推断，依据保留在 `priority_basis`。

- 多人 `owner` 数组拼成 `A、B`；其他非文本字段不强制字符串化。
- `高（原文）`、`高（推断：…）` 等标注会拆为标准优先级和依据。
- 非法字段在预览中改为“未明确”，原值放入 `due_date_raw` 或 `priority_basis`，通过 `warnings` 在预览表上方提示；人工核对后仍可保存。JSON 格式或数组结构错误仍返回 502。
- 多个 JSON 代码块仍只取最后一个。
- `/api/extract` 返回 `warnings`、`chat_id`、`conversation_id`；前端保存时携带 `chat_id`，与批次一起存入 `saved_batches`。已有批次表会在应用启动时兼容添加 `chat_id` 列；测试仅使用临时数据库。

## 九、真实 Coze 提取评测

先执行默认离线预检（不联网、不索取 Token、不加载 `.env`、不连接任何数据库）：

```bash
.venv/bin/python scripts/eval_coze.py
```

读取 `tests/cases/eval_cases.json` 中 8 个用例、31 个检查项，生成 `reports/eval_YYYYMMDD_HHMM.md`。离线报告全部标记“未执行”，不虚报通过率。

仅在真实调用获得批准、平台发布和预算已准备好后，由你本机设置非敏感环境变量 `COZE_BOT_ID`、`COZE_API_BASE`（国内为 `https://api.coze.cn`），然后执行：

```bash
.venv/bin/python scripts/eval_coze.py --live
```

令牌从进程环境变量 `COZE_API_TOKEN` 读取；未设置时使用 `getpass` 隐藏输入，不在命令行传令牌。脚本始终不加载现有 `.env`，通过 `--provider` 或进程 `AI_PROVIDER` 选择 `coze` / `coze_deploy`（未设置时默认 coze），禁止使用 mock 冒充真实评测。默认单次超时 90 秒，可通过进程环境变量 `COZE_TIMEOUT_SECONDS` 调整。

脚本逐条顺序调用，每条之间间隔 2 秒，不自动重试；一次最多 10 次提取。一次提取还包含状态轮询/消息读取 HTTP 请求，10 次上限不是所有 HTTP 请求的总数。超时不意味着平台没有产生消耗。

报告包含每条输入、任务表、warnings、检查项结果、chat_id/conversation_id，最后分别统计检查项与用例通过率。调用失败时该用例的检查项全部失败。`reports/` 已被 Git 忽略。时间以 Europe/London 生成；同一分钟报告不覆盖，可用 `--output-dir /private/tmp/另一目录` 指定独立目录。`--cases` 可指定其他用例文件，超过 10 条会在调用前拒绝。

关键词数组按“或”匹配任务；多个匹配任务必须全部满足字段检查，找不到任务判失败。多人负责用 `contains_all`；`all_equals` 不会在空数组上空通过。关键词比较仅是基础回归指标，不代表完整语义准确率；固定唯一标识用例也不能单独证明调用来源，应结合 Coze 对话 ID 和平台记录。

离线单元测试：

```bash
.venv/bin/python -m pytest -v -W error tests/test_eval_coze.py tests/test_extraction_contract.py
```

用例 3 未提供会议年份，`due_date≠2026-11-15` 只能拦截一个具体错误日期；用例 8 使用固定标识，只能辅助排除原 Mock 样例。两项检查按提供的要求保留，建议后续评测增加严格的日期来源判断和动态标识。


### 接口地址配置

现有协议的三个请求都使用以下配置，不在 HTTP 调用处写死地址：

| 环境变量 | 当前默认值 |
| --- | --- |
| `COZE_API_BASE` | `https://api.coze.cn` |
| `COZE_CHAT_PATH` | `/v3/chat` |
| `COZE_RETRIEVE_PATH` | `/v3/chat/retrieve` |
| `COZE_MESSAGES_PATH` | `/v3/chat/message/list` |

最终地址为基地址加对应路径。路径以单个 `/` 开头。评测脚本同样读取这些进程环境变量，支持配置 HTTPS 部署基地址，不限定国内/国际两个域名；基地址不应包含账号、查询参数或片段。这些变量仅用于传统 `coze` 提供方，部署 SSE 服务改用 `COZE_DEPLOY_URL`。改路由不等于适配新协议，不自行尝试候选接口。


## 十、新版扣子编程部署服务（coze_deploy）

传统 `coze` 和 `mock` 均保留。新版代码项目使用：

```dotenv
AI_PROVIDER=coze_deploy
COZE_DEPLOY_URL=https://your-deployment.coze.site
COZE_PROJECT_ID=填写调用示例中的数字项目ID
COZE_API_TOKEN=填写部署页面生成的API令牌
COZE_TIMEOUT_SECONDS=90
```

`COZE_DEPLOY_URL` 取部署页接口地址的域名部分，不包含 `/stream_run`；程序只 POST `{COZE_DEPLOY_URL}/stream_run`。`COZE_PROJECT_ID` 取部署页调用示例中的 `project_id`，以 JSON 整数发送；`COZE_BOT_ID` 和传统 `COZE_API_BASE` 不参与该提供方。点击部署页 **API Token** 按钮生成令牌，此令牌不是传统 v3 接口的 PAT。

每次提取生成新的 `mtb-<UUID hex>` session_id，绝不复用；请求文本为原有短前缀加完整会议原文。SSE 按顺序拼接 `answer` 片段，收到 `message_end` 或流结束停止；`error`、HTTP 错误和整体超时明确报错，不回退 Mock。401/403 提示“部署服务的 API Token 无效或无权限”。之后仍走相同的解析、warnings、人工核对、去重和看板流程。

answer 格式的唯一适配点为 `app/ai_client.py` 中的 `extract_deploy_answer()`：依次尝试 `data.content.answer`、`data.content.text`、字符串 `data.content`。未知结构仅记录字段名层级，不记录值；不猜测更深层字段。部署流没有已确认的 chat_id/conversation_id，因此不会伪造这两个 ID；响应、日志及评测报告使用新生成的 `session_id` 辅助对照，原 v3 调用 ID 逻辑不变。

### 先探测，再评测

以下默认命令不联网、不索取令牌，也不读取 `.env`：

```bash
.venv/bin/python scripts/probe_deploy.py
```

真实调用由你自行执行。脚本不会加载 `.env`，先在本机终端设置非敏感配置，再确认发送一次测试请求：

```bash
export COZE_DEPLOY_URL=https://your-deployment.coze.site
export COZE_PROJECT_ID=123456
.venv/bin/python scripts/probe_deploy.py --live
```

项目 ID/地址也可用 `--url` 和 `--project-id` 指定，命令行优先。`COZE_API_TOKEN` 从进程环境读取；未设置则 getpass 隐藏输入，无法安全隐藏输入时停止，不回退到回显。不要将真实令牌直接写进命令行参数或聊天。

探测只发送一次固定合成会议文本，不自动重试。接收的原始 SSE 行按顺序写入 `reports/probe_raw_YYYYMMDD_HHMM.txt`（行终止符统一为 LF）。为了写入前确认没有令牌，先在内存收集并检查原始行、解码事件及拼接回答，检测到令牌则不保存原始内容，只留下阻止写入的说明。终端打印 HTTP 状态、各 data.type 数量、每类第一条 content 的键名结构，以及当前函数拼接回答的前 300 字；结构中的 null 表示省略叶子值。

未知 answer 字段时，探测仍保留原始 SSE，拼接回答可能为空；请根据键名结构与报告确认字段后，仅修改集中提取函数。原始 SSE 可能包含合成回答和工具内容，分享前自行检查；不要把带真实业务内容的报告公开。

确认提取格式后，再由你运行 8 个用例的真实评测：

```bash
.venv/bin/python scripts/eval_coze.py --provider coze_deploy --live
```

或通过进程 `AI_PROVIDER=coze_deploy` 选择提供方。评测继续使用相同的 8 个用例、31 个检查项，顺序调用、间隔 2 秒、最多 10 次，不自动重试。每次调用都会有不同 session_id。探测和评测都不创建/访问数据库，reports/ 已被 Git 忽略；同一分钟的同名报告不覆盖，可用 `--output-dir` 指定其他目录。

## 十一、智能办公助手 V5（第二周）

在扣子上新建的单智能体，负责五项功能：文档总结、会议纪要拆解、周报生成、待办梳理、文本优化。完整方案见 [docs/智能办公Agent_V5_开发方案.docx](docs/智能办公Agent_V5_开发方案.docx)。

| 文件 | 作用 |
| --- | --- |
| `prompts/office_assistant_v5.md` | 人设全文（当前 V5.1），作为扣子编程项目 config 的 sp；{插件} 占位符替换为真实工具名（对应关系见下方“插件与调用逻辑”） |
| `tests/multiturn_cases.json` | 18 组多轮走查用例（上下文记忆 M、多轮任务 T、指令识别 I、异常 E）；素材 B 需自行填入一份约 800 字的文档 |
| `scripts/eval_multiturn.py` | 用例自动走查脚本，同一用例共用 session_id，报告名精确到秒 |
| `docs/办公助手V5_走查记录.md` | 两轮走查的逐条结果、问题类型、修改动作和版本记录 |
| `docs/walkthrough/` | 走查原始报告和 Trace 导出报告（证据） |

会议纪要看板本身不受影响：它仍调用原来的会议纪要智能体，每次提取新建 session_id。

### 多轮走查（仅扣子编程智能体路径需要；低代码智能体在“预览与调试”中手动走查）

```bash
# 在 .env 中填写 OFFICE_AGENT_URL / OFFICE_AGENT_TOKEN / OFFICE_PROJECT_ID 后：
set -a && source .env && set +a
.venv/bin/python scripts/eval_multiturn.py --probe      # 2 轮记忆探测：确认服务端按 session_id 保留上下文
.venv/bin/python scripts/eval_multiturn.py --warmup     # 预热后跑全部用例
.venv/bin/python scripts/eval_multiturn.py --only M1,T1 # 改人设后先跑失败的用例
```

报告写入 `reports/multiturn_YYYYMMDD_HHMMSS.md`（reports/ 已被 Git 忽略）。关键词检查只是初筛：PASS 也要抽查，CHECK 必须人工判定。脚本不读取 `.env` 文件本身、不打印令牌，HTTP 错误只显示状态码。

### 第二周交付说明

**部署**：扣子编程项目「智能办公助手」（项目 ID 7706150881112244239），已部署，发布渠道为 API（官方默认渠道）。模型 doubao-seed-2-0-pro-260215，temperature 0，关闭深度思考。多轮记忆沿用模板的 checkpointer，平台把请求中的 session_id 作为 thread_id，同一会话共享上下文、不同会话隔离（M1–M5、E5 验证）。

**插件与调用逻辑**：工具由模型按人设规则自主决定是否调用；所有工具失败重试 1 次，仍失败返回【读取失败】/【导出失败】开头的说明，模型据此如实告知用户，不猜测内容。

| 类别 | 工具 | 实现 | 何时调用（人设规定） | 实测证据（Trace 报告序号） |
| --- | --- | --- | --- | --- |
| 信息提取 | read_url | 链接读取技能 | 用户给出链接时 | E4 #088：读取失败后如实说明，未编造总结 |
| 信息提取 | parse_file | 内容处理技能 | 用户上传文件或图片时 | 未纳入脚本走查（脚本只发文本） |
| 文本处理 | count_chars | 纯代码 | 有字数要求时，每次输出前统计最终正文 | T3 #075 调用 2 次（统计→压缩→复核） |
| 文本处理 | get_calendar | 纯代码，北京时间 | 换算相对日期（传会议日期）、确定周报周期（不传） | M1 #051、周报 #055 等 |
| 格式转换 | export_document | 文档生成 + 对象存储技能 | 用户要求导出 Word / PDF 时，先确认内容 | T1 #069：返回 .docx 下载链接 |

**走查结果**：18 组多轮用例（上下文记忆、多轮任务、指令识别、异常各一组）。V5.0 首轮通过 7 条；按问题修改人设并新增日历工具后，V5.1 回归通过 13 条、部分通过 3 条、无未通过，M2/M3 待补素材 B。Trace 显示 V5.0 整理纪要时从不调用日期工具、靠模型自己推算，这正是日期错误的根因；V5.1 改为必须查日历后日期全部正确。逐条结果和遗留问题见 [走查记录](docs/办公助手V5_走查记录.md)。

