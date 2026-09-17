# Agent Enterprise

一个可以**跑在本机**的检索增强 Agent：左侧管理对话，右侧问答；知识库来自你自己指定的文件夹，
模型可以在多家厂商之间切换。带 TypeScript 写的网页界面、多模型接入、MCP 协议支持。

## 三步跑起来

```bash
git clone https://github.com/N10086/agent-enterprise.git
cd agent-enterprise
python run.bat          # Windows；macOS / Linux 用 ./run.sh
```

启动器会自动建虚拟环境、装依赖、打开浏览器到 <http://127.0.0.1:8760>。
首次安装会下载 PyTorch 与嵌入模型（约几百 MB 到 2GB，视平台而定），之后启动只要几秒。

不想用启动器就手动来三行：

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows；macOS/Linux 是 source .venv/bin/activate
pip install -r requirements.txt
python main.py ui
```

**不需要 Node**——前端构建产物已经提交在 `webui/dist/`，只有改前端时才需要 `npm run build`。

### 模型怎么配

网页界面打开后点左下角「更多」→ **API 配置**，选厂商、填 API Key 即可（Key 只存在浏览器本地，不落盘）。

想在命令行或启动前配好，就复制一份 `.env`：

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-xxxx
OPENAI_API_BASE=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

## 数据放在哪里（重新 clone 不会丢）

**代码和数据是分开的**：对话历史、知识库索引、默认工作区文件夹都放在操作系统的应用数据目录里，
不在项目目录内。

| 系统 | 数据目录 |
|---|---|
| Windows | `%APPDATA%\agent-enterprise` |
| macOS | `~/Library/Application Support/agent-enterprise` |
| Linux | `~/.local/share/agent-enterprise` |

```
<数据目录>/
    workspace.json          当前工作区文件夹 + 知识库排除名单
    conversations/*.json    所有对话（含工具轨迹）
    kbsession/              该文件夹的 FAISS 索引
    documents/              默认工作区文件夹
```

所以**删掉项目目录、重新 clone 一份再跑，之前的对话和知识库照样在**。
想换位置（比如放到移动硬盘或云盘同步目录）：

```bash
AGENT_DATA_DIR=D:\my-agent-data python main.py ui
```

从早期版本（数据放在项目里 `public/appdata`）升级上来时，首次启动会自动把旧数据复制过去，
旧目录保持原样不动。

## 它是怎么工作的

```
用户问题
   │
   ▼
supervisor ──► researcher ⇄ tool_executor ──► tool_result_reader ──► grade_documents
                   │                （LLM 自己决定调哪个工具）            │
                   │                                       相关 ──► analyst ──► reviewer ──► END
                   └──────────────── 不相关 ──► supervisor（换查询重检）◄──┘
```

- **researcher** 把问题和工具一起交给模型，由它用原生 `tool_calls` 决定调哪个工具、或者不调。
  工具说明里写清了各自的适用时机，模型自己判断——没有关键词路由。
- **tool_executor / tool_result_reader** 执行工具并给结果打标签（`[Tool: rag_search] 检索到的文档片段：…`），
  避免裸值混进上下文让模型分不清来源。
- **grade_documents** 是检索质检（adaptive-RAG / CRAG 的 grading 步）：检索片段与问题无关时，
  不硬着头皮作答，而是换查询重检一轮。是否触发由检索片段的最高相似度决定——相似度**只决定要不要质检**，
  不用来丢弃证据。
- **analyst / reviewer** 生成答案并做格式与关键字审核，未通过会带着修正指令回到 supervisor。

这套结构对应 LangGraph 官方 [Agentic RAG 教程](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
的形状（`generate_query_or_respond → retrieve → grade_documents → generate/rewrite`）。

## 检索链路

`rag_search` 工具内部跑的是完整链路，而不是单查询直查：

```
问题 ─► multi-query 改写 ─► HyDE 假设答案 ─► 多路 FAISS 检索 ─► RRF 融合 top-5 ─► 带出处返回
```

- 切分：语义切分，块长 400~800 字符
- 向量库：本地 FAISS（`IndexFlatIP` + L2 归一化，内积即余弦相似度），零服务、单文件索引
- 融合：RRF（只看名次，不比较不同查询的相似度绝对值）

## 知识库 = 你自己的文件夹

网页界面里的「知识库」不是项目自带的语料，而是**你指定的本机文件夹**：

```
<你指定的文件夹>/          ← 工作区：里面的文档就是知识库
<数据目录>/                ← 应用自己的数据（对话、索引、默认工作区），不往你的文件夹里写东西
```

支持的格式：`.pdf` / `.docx` / `.pptx` / `.md` / `.txt` / `.rst` / `.csv` / `.json`。
其中 docx/pptx 用标准库解 zip+XML，PDF 用 pypdf——不需要装一堆解析库。

「移出知识库」只是把文件记进排除名单、不再参与检索，**不会删除你磁盘上的原文件**。

## 多模型

`app/llm.py` 里是一张 provider 表，任何 OpenAI 兼容端点都能接：

| 厂商 | 端点 | 环境变量 |
|---|---|---|
| DeepSeek | api.deepseek.com | `DEEPSEEK_API_KEY` |
| 通义千问 Qwen | dashscope 兼容模式 | `DASHSCOPE_API_KEY` |
| 智谱 GLM | open.bigmodel.cn | `ZHIPUAI_API_KEY` |
| Kimi | api.moonshot.cn | `MOONSHOT_API_KEY` |
| OpenAI | api.openai.com | `OPENAI_DIRECT_API_KEY` |
| 本地 Ollama | 127.0.0.1:11434 | 随意填 |
| 自定义 | 界面上填 | — |

切换模型靠一个 contextvar，**不改任何节点代码**：图里的 `llm` 是 `ActiveLLM` 替身，
每次调用都转发给当前生效的模型；RAG 链路内部的改写 / HyDE 也跟着一起切，
不会出现"图用 Qwen、检索用 DeepSeek"的错配。API Key 存在浏览器本地，不落盘。

## 安装（手动方式）

```bash
pip install -r requirements.txt
cp .env.example .env        # 填 OPENAI_API_KEY / OPENAI_API_BASE / MODEL_NAME
```

`.env` 走的是 OpenAI 兼容协议，所以填任意一家的地址与 key 都能直接跑。

## 运行

```bash
python run.bat                    # Windows 一键：建 venv + 装依赖 + 启动
./run.sh                          # macOS / Linux
python main.py ui                 # 已经装好依赖时，直接启动
python main.py ui --port 9000     # 换端口
python main.py ask --question "明朝开国皇帝是谁"
python main.py demo
```

界面上：

- 左栏是会话列表（新对话 / 切换 / 删除），左下角「更多」以 880×720 的独立窗口弹出设置
- 设置窗口两个页签：**API 配置**（厂商、模型、Key）和**知识库**（文件夹选择、导入、逐文件移出/纳入）
- 提问时能实时看到 Agent 的执行过程：走到哪个节点、调了什么工具、检索到哪几段、token 用量
- 对话落盘，**关掉页面甚至重启服务后历史仍在**，连工具轨迹都能完整回放

前端是 TypeScript + esbuild，构建产物已提交，所以**运行界面不需要 Node**。
要改前端再装一次即可：

```bash
cd webui && npm install && npm run build     # 另有 npm run typecheck
```

## MCP 支持

两边都通：

- **服务端**：`python mcp_server.py` 把 calculator / get_current_time / web_search / rag_search
  以 MCP 协议暴露出去，Claude Desktop、Cursor 等客户端可直接调用；`--transport streamable-http` 可远程复用。
- **客户端**：`app/mcp_client.py` 让本项目的 Agent 也能调用**外部** MCP server 的工具。
  设好 `MCP_SERVERS` 即可（JSON）：

```json
{
  "some-server": {"command": "python", "args": ["path/to/server.py"]},
  "remote": {"url": "http://127.0.0.1:8000/mcp"}
}
```

MCP 会话是异步长连接、图是同步跑的，所以客户端在后台线程里维持会话，工具调用通过
`run_coroutine_threadsafe` 同步等结果——连接只建一次，不是每次调用都重启子进程。

## 目录结构

```
app/
  graph.py         LangGraph 编排（supervisor / researcher / 工具 / 质检 / analyst / reviewer）
  llm.py           多模型接入 + 当前生效模型
  rag_chain.py     检索链路（multi-query + HyDE + FAISS + RRF）
  tools.py         工具定义与注册表（内置 + MCP）
  mcp_client.py    MCP 客户端
  documents.py     文档解析（pdf/docx/pptx/md/txt…）
  workspace.py     工作区文件夹 + 对话落盘
  paths.py         数据目录（代码与数据分离）
  session.py       知识库：扫描文件夹、建索引、门控 rag_search
  knowledge.py     向量库接入层
  state.py         图状态
  runner.py        跑图 / 流式事件
  web_search.py    多引擎联网搜索
webui/             TypeScript 前端（主窗口 + 设置窗口），dist/ 是已构建产物
serve_ui.py        本机 HTTP 服务（标准库，无 Web 框架）
mcp_server.py      MCP 服务端
run.bat / run.sh   一键启动器
```

## 说明

- 网页服务默认只监听 `127.0.0.1`；换 `--host` 会提示风险（界面上会填 API Key）。
- 联网搜索走多个引擎并合并结果，若某个引擎失败会自动换下一个。
- 删掉数据目录即可重置应用（不会动你自己的文档文件夹）。
- 只想跑界面、暂时不用本地知识库的话，可以不装 `sentence-transformers`：
  问答照常，导入文档时会提示补装依赖。
