# Agentic Search Qwen

以 **Qwen** 为决策核心的企业级 Agentic Search 演示项目：不再沿用传统 RAG「一次召回 + 拼接上下文」的固定流程，而是把**检索策略交给大模型**——由模型自主决定调用哪个数据源、用什么参数、是否需要继续检索，多轮迭代直到信息足够后生成答案。

## 核心特性

- **Agent 决策循环**：每轮输出结构化工具调用，结果回填上下文后再决策，最多 `MAX_SEARCH_ROUNDS = 20` 轮
- **6 类数据源 / 13 个工具**：SQLite 结构化数据、Chroma 向量库、Whoosh 全文索引、代码仓库、模拟企业系统、历史日志
- **混合检索**：向量检索管语义召回，BM25 管精确关键词，SQL 管聚合与条件筛选，按意图分工
- **可观测**：前端可展开查看每轮决策、工具入参、原始响应；日志同时写文本与 JSONL，并注册为 `log_search` 工具供 Agent 自省
- **故障降级**：大模型接口异常时自动切换到本地规则检索，保证服务可用
- **可评测**：内置工具选择评测（24 条）与问答准确率评测（10 条）

## 架构

```text
┌──────────────────┐      ┌──────────────────┐
│ Streamlit (app)  │      │ CLI (main.py)    │
└────────┬─────────┘      └────────┬─────────┘
         └───────────┬─────────────┘
                     ▼
        AgenticSearchAgent (core/agent.py)
          │  决策循环（≤20 轮）
          ▼
        Qwen / DashScope（OpenAI 兼容协议）
          │  tool_calls
          ▼
 ┌──────────────────────────────────────────────────────────┐
 │ SQLite │ Chroma │ Whoosh │ 代码仓库 │ 企业系统 SDK │ 日志 │
 └──────────────────────────────────────────────────────────┘
```

## 目录结构

```text
.
├─ app.py                  Streamlit Web 界面（聊天式问答 + Trace 查看）
├─ main.py                 CLI 入口，启动时调用 seed_all_large() 初始化数据
├─ config.py               路径常量、.env 加载、模型配置
├─ seed_data.py            小规模示例数据
├─ seed_data_large.py      大规模演示数据（员工/项目/合同/产品/文档/代码）
├─ eval_agent.py           工具选择与检索轮数评测（24 条用例）
├─ eval_accuracy.py        问答准确率评测（10 条用例）
├─ core/
│  ├─ agent.py             Agent 主循环、工具定义、Prompt、降级逻辑
│  ├─ llm.py               Qwen 客户端封装（OpenAI SDK + DashScope base_url）
│  └─ logger.py            单例日志：文本 + JSONL 双通道
├─ engines/
│  ├─ database.py          SQLite 连接、建表、查询与入数据
│  ├─ vector_db.py         Chroma 持久化向量库（余弦距离）
│  ├─ keyword_search.py    Whoosh 全文索引 + jieba 分词（BM25）
│  ├─ code_search.py       代码仓库文件检索与读取
│  └─ enterprise_sdk.py    模拟 HR / 财务 / 项目 / Wiki 系统
├─ data/                   运行期数据（已 gitignore，可随时重建）
│  ├─ db/enterprise.db     SQLite 业务库
│  ├─ vector_db/           Chroma 持久化目录
│  ├─ keyword_index/       Whoosh 索引
│  ├─ code_repo/           生成的示例代码仓库
│  ├─ logs/                文本日志与 JSONL 日志
│  └─ eval/                评测结果
├─ requirements.txt
└─ .env                    密钥配置（已 gitignore）
```

## 工具清单

| 工具 | 作用 |
|---|---|
| `database_search` | 在指定表中做模糊查询 |
| `database_sql` | 执行只读 SQL |
| `database_schema` | 查看表结构与数据量 |
| `vector_search` | 向量语义检索（公司信息/技术文档/会议纪要） |
| `vector_info` | 查看向量集合信息 |
| `keyword_search` | BM25 关键词检索（政策/技术文章） |
| `keyword_info` | 查看关键词索引信息 |
| `code_search` | 代码内容检索 |
| `code_read` | 读取指定代码文件 |
| `code_list` | 列出代码仓库文件 |
| `enterprise_call` | 调用模拟企业系统（hr / finance / project / wiki） |
| `enterprise_systems` | 查看可用系统与动作 |
| `log_search` | 检索历史检索日志 |

## 快速开始

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥
#    在项目根目录创建 .env，内容见下节

# 4. 初始化演示数据（首次或数据被清空时执行）
python seed_data_large.py

# 5. 启动 Web 界面
python -m streamlit run app.py

# 或使用命令行交互
python main.py
```

启动后访问 <http://localhost:8501>。

### .env 配置

```ini
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.6-plus
```

`DASHSCOPE_API_KEY` 必填；另外两项可省略，使用 `config.py` 中的默认值。

## 评测

```powershell
# 工具选择 / 检索轮数：24 条用例，含 5 条跨源问题
python eval_agent.py --dry-run          # 只校验用例定义
python eval_agent.py --limit 5
python eval_agent.py                    # 全量（耗时较长）

# 问答准确率：10 条用例，标准答案部分由 SQL 动态计算
python eval_accuracy.py --dry-run
python eval_accuracy.py
python eval_accuracy.py --ids A01,A08
```

结果保存到 `data/eval/*.json`，包含每题答案、实际工具调用序列、轮数与耗时。

## 数据说明

- 所有业务数据均由 `seed_data_large.py` **程序生成**，不依赖外部系统
- `data/` 目录不入库，删除后重新执行 `python seed_data_large.py` 即可重建
- 向量检索使用 Chroma 默认嵌入模型 `all-MiniLM-L6-v2`，首次运行会下载约 83MB 模型到 `~/.cache/chroma`
- `insert_data()` 是纯插入、无幂等保护，请勿重复执行 seed，否则数据会叠加

## 生产环境需改造为
改造顺序建议：① SQLite → PostgreSQL/MySQL；② Chroma 本地模式 → 托管向量库（如阿里云 DashVector、Milvus）；③ Whoosh → OpenSearch/ES；④ 把 Agent 抽成 FastAPI 接口，Streamlit 只做前端；⑤ 加请求队列与限流（单次查询可能 2~3 分钟）。

## 已知限制

- SQLite、Whoosh、Chroma 均为本地文件模式，仅支持单进程运行，不可多副本横向扩展
- 默认嵌入模型偏英文，中文语义检索效果有限；如需更好效果应替换为中文/多语言嵌入模型并重建索引
- 大模型单次查询最长可能跑满 20 轮，耗时与 token 消耗较高，需要按场景调整轮数上限
