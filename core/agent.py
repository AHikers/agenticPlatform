# Agent 核心模块：多轮决策循环 + 13 个工具 + 本地降级兜底
# 关键设计：
#   - 工具定义用 OpenAI Function Calling 的 JSON Schema 格式，直接喂给 LLM
#   - messages 按 OpenAI 工具协议拼接：assistant(tool_calls) → tool(结果) → 下一轮 assistant
#   - 当 LLM 报错/不可用时，自动降级到 _local_search（规则式意图识别 + 本地引擎直查）
from __future__ import annotations
import json
from datetime import datetime, timedelta
from typing import List, Dict
from core.llm import LLMClient
from core.logger import Logger
from engines.database import DatabaseEngine
from engines.vector_db import VectorDBEngine
from engines.keyword_search import KeywordSearchEngine
from engines.code_search import CodeSearchEngine
from engines.enterprise_sdk import EnterpriseSDK
from config import MAX_SEARCH_ROUNDS


# System Prompt 模板（运行时用 _build_system_prompt 替换日期占位符）
# 包含 3 段：角色与工作方式 / 意图路由规则 / 关键词提取规则
# 意图路由规则是给模型的"必看速查表"，防止它选错数据源
SYSTEM_PROMPT_TEMPLATE = """你是一个智能企业搜索助手。你拥有多个搜索工具来帮助用户找到信息。

【当前日期: {current_date}】
今天是 {current_date}，{weekday}。你必须根据当前日期来理解用户问题中的时间词（如"本周"、"本月"、"今天"、"最近"等）。

例如：
- "本周" = {week_start} 至 {week_end}
- "本月" = {month_start} 至 {month_end}
- "今天" = {current_date}
- "最近" = 最近7天（{recent_start} 至 {current_date}）

你的工作方式是：
1. 分析用户的问题，推理需要什么信息来回答
2. 决定使用哪个搜索工具，以及用什么参数搜索
3. 获取搜索结果后，分析结果是否足以回答问题
4. 如果信息不足，继续推理下一步应该搜索什么
5. 当你认为已经收集到足够信息时，给出最终答案

重要原则：
- 每次搜索前要明确推理：为什么选这个工具？期望找到什么？
- 搜索结果可能不完整，需要交叉验证
- 优先搜索最可能包含答案的数据源
- 如果一个搜索没有找到有用信息，尝试换一个工具或换关键词
- 最终回答要基于搜索到的实际数据，不要编造信息

【查询意图路由规则 - 必须严格遵守】
在调用任何工具之前，必须先判断用户问题的核心意图，选择最匹配的工具和数据源：

- 支付/收款/交易/流水/账单/对账 → 优先调用 enterprise_call(finance, get_payments) 或 enterprise_call(finance, get_transactions)
  参数示例: enterprise_call(system="finance", action="get_payments", params={{"type": "收入", "start_date": "2026-01-01", "end_date": "2026-01-31"}})
  注意: type 参数只能填 "收入" 或 "支出"（不要填"收款""付款"等）；日期格式必须是 YYYY-MM-DD
- 预算/费用/报销/发票 → 优先调用 enterprise_call(finance, get_budget/get_invoices/get_expenses)
- 员工信息/人事/组织架构/请假 → 优先调用 enterprise_call(hr, ...) 或 database_search(table=employees)
- 项目进度/里程碑 → 优先调用 enterprise_call(project, ...) 或 database_search(table=projects)
- 合同/客户 → 优先调用 database_search(table=contracts)
- 产品/价格 → 优先调用 database_search(table=products)
- 公司政策/制度/流程 → 优先调用 enterprise_call(wiki, search_policies) 或 keyword_search
- 技术文档/架构/规范 → 优先调用 enterprise_call(wiki, search_tech_docs) 或 keyword_search 或 code_search
- 模糊/概念性查询 → 优先使用 vector_search
- 精确关键词匹配 → 使用 keyword_search
- 需要精确筛选/聚合/排序 → 使用 database_sql

【关键词提取规则 - 必须严格遵守】
- 提取用户问题中的核心业务关键词，而不是泛词
- ❌ 错误示例：查询"本周的支付数据情况"→ 关键词"数据"（太泛，会匹配到"数据分析师"等无关内容）
- ✅ 正确示例：查询"本周的支付数据情况"→ 关键词"支付"或直接调用财务系统
- ❌ 错误示例：查询"项目进度如何"→ 关键词"如何"（无意义）
- ✅ 正确示例：查询"项目进度如何"→ 关键词"项目"或直接调用项目系统
- 如果关键词太泛（如"数据"、"信息"、"情况"），应该结合上下文提取更具体的词

你可以使用的搜索工具：
1. database_search - 在本地SQLite数据库中搜索结构化数据（员工、部门、项目、合同、产品）
2. database_sql - 执行SQL查询获取精确数据
3. database_schema - 查看数据库表结构
4. vector_search - 在向量数据库中进行语义搜索（适合模糊/概念性查询）
5. vector_info - 查看向量数据库集合信息
6. keyword_search - 在关键词索引中进行全文搜索（适合精确关键词匹配）
7. keyword_info - 查看关键词索引信息
8. code_search - 在代码仓库中搜索代码
9. code_read - 读取代码文件内容
10. code_list - 列出代码仓库文件
11. enterprise_call - 调用企业内部系统（HR/财务/项目/Wiki）
12. enterprise_systems - 查看可用的企业系统
13. log_search - 搜索历史日志

当你已经收集到足够的信息来回答用户问题时，请直接给出最终答案，不要再调用工具。"""


class AgenticSearchAgent:
    """Agent 主循环：把 LLM 的工具调用决策映射到本地引擎的统一入口。

    典型用法：
        agent = AgenticSearchAgent()
        answer = agent.search("公司里薪资最高的员工是谁？")
        # 或带完整 Trace（前端展示 / 评测用）
        trace  = agent.search_with_trace("技术架构评审改了什么？")
    """

    def __init__(self):
        # 所有引擎在构造函数中一次性初始化（SQLite / Chroma / Whoosh 都是本地文件模式，打开连接成本低）
        self.llm = LLMClient()
        self.logger = Logger()                  # 单例 Logger，全局共用
        self.db = DatabaseEngine()              # SQLite 结构化数据
        self.vector_db = VectorDBEngine()       # 向量语义检索
        self.keyword_search = KeywordSearchEngine()  # BM25 全文检索
        self.code_search = CodeSearchEngine()   # 代码仓库检索
        self.enterprise_sdk = EnterpriseSDK()   # 模拟企业系统 API（HR/财务/项目/Wiki）

        # 13 个工具的 JSON Schema，直接作为 chat.completions.create(tools=...) 的入参
        self.tools = self._define_tools()

    def _define_tools(self) -> list[dict]:
        """返回 OpenAI Function Calling 格式的工具描述列表。

        共 13 个工具，分 6 类：
          SQL 结构化    database_search / database_sql / database_schema
          向量语义      vector_search / vector_info
          关键词全文    keyword_search / keyword_info
          代码仓库      code_search / code_read / code_list
          企业系统      enterprise_call / enterprise_systems
          日志自省      log_search
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "database_search",
                    "description": "在本地SQLite数据库中搜索结构化数据。可以指定表名或搜索所有表。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"},
                            "table": {"type": "string", "description": "表名(可选): employees, departments, projects, contracts, products", "enum": ["employees", "departments", "projects", "contracts", "products"]},
                            "limit": {"type": "integer", "description": "返回结果数量限制", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "database_sql",
                    "description": "执行SQL查询获取精确数据。适用于需要精确筛选、聚合、排序的场景。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "要执行的SQL查询语句"}
                        },
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "database_schema",
                    "description": "查看数据库表结构，了解有哪些表和字段。",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vector_search",
                    "description": "在向量数据库中进行语义搜索。适合模糊查询、概念性搜索，能找到语义相近的内容。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索查询文本"},
                            "collection": {"type": "string", "description": "集合名称(可选)，不指定则搜索所有集合"},
                            "n_results": {"type": "integer", "description": "返回结果数量", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vector_info",
                    "description": "查看向量数据库中所有集合的信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "keyword_search",
                    "description": "在关键词索引中进行全文搜索。适合精确关键词匹配，支持中文分词。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"},
                            "index_name": {"type": "string", "description": "索引名称(可选)，不指定则搜索所有索引"},
                            "limit": {"type": "integer", "description": "返回结果数量", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "keyword_info",
                    "description": "查看关键词索引信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "code_search",
                    "description": "在代码仓库中搜索代码。支持按文件名、内容、符号搜索。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"},
                            "search_type": {"type": "string", "description": "搜索类型", "enum": ["filename", "content", "symbol"], "default": "content"},
                            "file_pattern": {"type": "string", "description": "文件名过滤模式(可选)"},
                            "limit": {"type": "integer", "description": "返回结果数量", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "code_read",
                    "description": "读取代码仓库中指定文件的内容。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "文件路径"},
                            "start_line": {"type": "integer", "description": "起始行号", "default": 1},
                            "end_line": {"type": "integer", "description": "结束行号(可选)"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "code_list",
                    "description": "列出代码仓库中的文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "description": "目录路径(可选)"},
                            "pattern": {"type": "string", "description": "文件名过滤模式(可选)"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "enterprise_call",
                    "description": "调用企业内部系统API。支持HR系统、财务系统、项目管理系统、企业知识库。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system": {"type": "string", "description": "系统名称", "enum": ["hr", "finance", "project", "wiki"]},
                            "action": {"type": "string", "description": "操作名称"},
                            "params": {"type": "object", "description": "操作参数(可选)"}
                        },
                        "required": ["system", "action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "enterprise_systems",
                    "description": "查看所有可用的企业内部系统及其操作。",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "log_search",
                    "description": "搜索历史操作日志。可以查看之前的搜索记录和决策过程。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string", "description": "搜索关键词(可选)"},
                            "level": {"type": "string", "description": "日志级别过滤(可选)", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                            "engine": {"type": "string", "description": "搜索引擎过滤(可选)"},
                            "limit": {"type": "integer", "description": "返回结果数量", "default": 20}
                        }
                    }
                }
            },
        ]

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """根据 tool_name 把 arguments 路由到对应引擎的方法，返回 JSON 字符串。

        所有分支共享一条 try/except：任何异常都不往上抛，而是包装成
        {"error": "..."} 返回给 LLM。这样即使某个工具炸了，Agent 还能
        继续换下一个工具，不会整条链路中断。
        """
        try:
            if tool_name == "database_search":
                return self.db.search(
                    query=arguments.get("query", ""),
                    table=arguments.get("table"),
                    limit=arguments.get("limit", 10)
                )
            elif tool_name == "database_sql":
                return self.db.execute_sql(arguments.get("sql", ""))
            elif tool_name == "database_schema":
                return self.db.get_schema()
            elif tool_name == "vector_search":
                return self.vector_db.search(
                    query=arguments.get("query", ""),
                    collection_name=arguments.get("collection"),
                    n_results=arguments.get("n_results", 5)
                )
            elif tool_name == "vector_info":
                return self.vector_db.get_all_collections_info()
            elif tool_name == "keyword_search":
                return self.keyword_search.search(
                    query=arguments.get("query", ""),
                    index_name=arguments.get("index_name"),
                    limit=arguments.get("limit", 10)
                )
            elif tool_name == "keyword_info":
                return self.keyword_search.get_all_indexes_info()
            elif tool_name == "code_search":
                return self.code_search.search(
                    query=arguments.get("query", ""),
                    search_type=arguments.get("search_type", "content"),
                    file_pattern=arguments.get("file_pattern"),
                    limit=arguments.get("limit", 10)
                )
            elif tool_name == "code_read":
                return self.code_search.read_file(
                    file_path=arguments.get("file_path", ""),
                    start_line=arguments.get("start_line", 1),
                    end_line=arguments.get("end_line")
                )
            elif tool_name == "code_list":
                return self.code_search.list_files(
                    directory=arguments.get("directory", ""),
                    pattern=arguments.get("pattern")
                )
            elif tool_name == "enterprise_call":
                return self.enterprise_sdk.call(
                    system=arguments.get("system", ""),
                    action=arguments.get("action", ""),
                    params=arguments.get("params")
                )
            elif tool_name == "enterprise_systems":
                return self.enterprise_sdk.get_available_systems()
            elif tool_name == "log_search":
                return self.logger.search_logs(
                    keyword=arguments.get("keyword"),
                    level=arguments.get("level"),
                    engine=arguments.get("engine"),
                    limit=arguments.get("limit", 20)
                )
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Tool execution error: {str(e)}"}, ensure_ascii=False)

    def _build_system_prompt(self) -> str:
        """替换 SYSTEM_PROMPT_TEMPLATE 中的时间占位符。

        目的：让模型把"本周/本月/最近"理解成真实的日期区间，而不是模糊词。
        这对财务/项目类跨时间查询至关重要——否则"本月支付情况"会变成全库扫描。
        """
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        weekday = now.strftime("%A")
        weekday_cn = {"Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三",
                      "Thursday": "星期四", "Friday": "星期五", "Saturday": "星期六", "Sunday": "星期日"}.get(weekday, weekday)

        week_start = now - timedelta(days=now.weekday())
        week_end = week_start + timedelta(days=6)
        month_start = now.replace(day=1)
        month_end = (month_start.replace(month=month_start.month + 1) if month_start.month < 12 else month_start.replace(year=month_start.year + 1, month=1)) - timedelta(days=1)
        recent_start = now - timedelta(days=7)

        return SYSTEM_PROMPT_TEMPLATE.format(
            current_date=current_date,
            weekday=weekday_cn,
            week_start=week_start.strftime("%Y-%m-%d"),
            week_end=week_end.strftime("%Y-%m-%d"),
            month_start=month_start.strftime("%Y-%m-%d"),
            month_end=month_end.strftime("%Y-%m-%d"),
            recent_start=recent_start.strftime("%Y-%m-%d"),
        )

    def search(self, question: str, verbose: bool = True) -> str:
        """多轮 Agent 主循环：每轮让 LLM 输出 tool_calls，执行后把结果回填 messages。

        循环终止有三种情况：
          1. LLM 不再返回 tool_calls → 认为信息足够，直接用其 content 作为最终答案
          2. LLM 返回的 content 看起来像 API 错误 → 降级到 _local_search（规则式本地搜索）
          3. 跑满 MAX_SEARCH_ROUNDS 轮 → 追加一条强制收敛的 user prompt，再让 LLM 生成答案
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": question}
        ]

        for round_num in range(1, MAX_SEARCH_ROUNDS + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"🔍 搜索轮次 {round_num}/{MAX_SEARCH_ROUNDS}")
                print(f"{'='*60}")

            response = self.llm.chat_with_tools(messages, self.tools)

            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    try:
                        arguments = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                    except json.JSONDecodeError:
                        arguments = {}

                    if verbose:
                        print(f"\n📋 决策: 调用 {tool_name}")
                        print(f"   参数: {json.dumps(arguments, ensure_ascii=False)}")

                    self.logger.log_decision(round_num, f"Call {tool_name}", json.dumps(arguments, ensure_ascii=False))

                    result = self._execute_tool(tool_name, arguments)

                    result_summary = result[:200] + "..." if len(result) > 200 else result
                    if verbose:
                        print(f"\n📊 搜索结果预览: {result_summary}")

                    self.logger.log_search(tool_name, json.dumps(arguments, ensure_ascii=False), result_summary, round_num)

                    messages.append({"role": "assistant", "content": None, "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tc["arguments"]}
                    }]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            else:
                final_answer = response["content"]
                if self._is_api_error(final_answer):
                    if verbose:
                        print(f"\n⚠️ API连接失败，降级到本地搜索模式")
                    return self._local_search(question, verbose)
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"✅ 最终回答 (经过 {round_num} 轮搜索)")
                    print(f"{'='*60}")
                    print(final_answer)

                self.logger.log_final_answer(question, final_answer, round_num)
                return final_answer

        final_prompt = f"你已经进行了{MAX_SEARCH_ROUNDS}轮搜索。请根据已收集的信息，给出最终回答。如果信息不足，请说明。"
        messages.append({"role": "user", "content": final_prompt})
        final_answer = self.llm.chat(messages)

        if verbose:
            print(f"\n{'='*60}")
            print(f"⚠️ 达到最大搜索轮数 ({MAX_SEARCH_ROUNDS})，强制输出回答")
            print(f"{'='*60}")
            print(final_answer)

        self.logger.log_final_answer(question, final_answer, MAX_SEARCH_ROUNDS)
        return final_answer

    def search_with_trace(self, question: str) -> dict:
        """search 的带 Trace 版本，返回完整的轮次记录字典（供 Streamlit 前端展开、eval 脚本统计）。

        返回结构：
          question / timestamp / system_prompt / api_available
          rounds[]: 每轮含 llm_raw_response / llm_text_content / tool_calls[] / is_final
          final_answer / total_rounds
        """
        trace = {
            "question": question,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": self._build_system_prompt(),
            "api_available": True,
            "rounds": [],
            "final_answer": "",
            "total_rounds": 0,
        }

        messages = [
            {"role": "system", "content": trace["system_prompt"]},
            {"role": "user", "content": question}
        ]

        for round_num in range(1, MAX_SEARCH_ROUNDS + 1):
            round_trace = {
                "round": round_num,
                "llm_raw_response": "",
                "llm_text_content": "",
                "tool_calls": [],
                "is_final": False,
            }

            response = self.llm.chat_with_tools(messages, self.tools)

            round_trace["llm_raw_response"] = json.dumps(response, ensure_ascii=False, indent=2)
            round_trace["llm_text_content"] = response.get("content", "")

            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    try:
                        arguments = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                    except json.JSONDecodeError:
                        arguments = {}

                    self.logger.log_decision(round_num, f"Call {tool_name}", json.dumps(arguments, ensure_ascii=False))

                    result = self._execute_tool(tool_name, arguments)

                    tool_trace = {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "result_preview": result[:500] + "..." if len(result) > 500 else result,
                    }
                    round_trace["tool_calls"].append(tool_trace)

                    self.logger.log_search(tool_name, json.dumps(arguments, ensure_ascii=False), result[:200], round_num)

                    messages.append({"role": "assistant", "content": None, "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tc["arguments"]}
                    }]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            else:
                final_answer = response["content"]
                round_trace["is_final"] = True
                round_trace["llm_text_content"] = final_answer

                if self._is_api_error(final_answer):
                    trace["api_available"] = False
                    final_answer = self._local_search_with_trace(question, trace)

                trace["rounds"].append(round_trace)
                trace["final_answer"] = final_answer
                trace["total_rounds"] = round_num
                self.logger.log_final_answer(question, final_answer, round_num)
                return trace

            trace["rounds"].append(round_trace)

        final_prompt = f"你已经进行了{MAX_SEARCH_ROUNDS}轮搜索。请根据已收集的信息，给出最终回答。如果信息不足，请说明。"
        messages.append({"role": "user", "content": final_prompt})
        final_answer = self.llm.chat(messages)

        trace["final_answer"] = final_answer
        trace["total_rounds"] = MAX_SEARCH_ROUNDS
        self.logger.log_final_answer(question, final_answer, MAX_SEARCH_ROUNDS)
        return trace

    def search_stream(self, question: str):
        """search 的流式版本（生成器），Streamlit 前端逐块 yield 渲染。

        yield 的 step 类型：
          init       → system_prompt 与问题
          tool_call  → 某一次工具调用的入参与结果
          final      → 最终答案
          fallback   → API 不可用时触发降级
        """
        yield {
            "step": "init",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": self._build_system_prompt(),
            "question": question,
            "api_available": True,
        }

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": question}
        ]

        for round_num in range(1, MAX_SEARCH_ROUNDS + 1):
            response = self.llm.chat_with_tools(messages, self.tools)

            if response["tool_calls"]:
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    try:
                        arguments = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                    except json.JSONDecodeError:
                        arguments = {}

                    self.logger.log_decision(round_num, f"Call {tool_name}", json.dumps(arguments, ensure_ascii=False))
                    result = self._execute_tool(tool_name, arguments)
                    self.logger.log_search(tool_name, json.dumps(arguments, ensure_ascii=False), result[:200], round_num)

                    yield {
                        "step": "tool_call",
                        "round": round_num,
                        "llm_raw_response": json.dumps(response, ensure_ascii=False, indent=2),
                        "llm_text_content": response.get("content", ""),
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result,
                    }

                    messages.append({"role": "assistant", "content": None, "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tc["arguments"]}
                    }]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            else:
                final_answer = response["content"]

                if self._is_api_error(final_answer):
                    yield {
                        "step": "fallback",
                        "round": round_num,
                        "intent": self._detect_intent(question),
                    }
                    final_answer = self._local_search_with_trace(question, {"fallback_intent": self._detect_intent(question)})

                self.logger.log_final_answer(question, final_answer, round_num)

                # 构建最终回答生成时的消息上下文摘要
                final_messages_context = self._build_messages_summary(messages)

                yield {
                    "step": "final",
                    "round": round_num,
                    "final_answer": final_answer,
                    "llm_raw_response": json.dumps(response, ensure_ascii=False, indent=2),
                    "final_messages_context": final_messages_context,
                    "final_prompt_type": "模型自主判断信息已足够，直接生成回答",
                }
                return

        final_prompt = f"你已经进行了{MAX_SEARCH_ROUNDS}轮搜索。请根据已收集的信息，给出最终回答。"
        messages.append({"role": "user", "content": final_prompt})
        final_answer = self.llm.chat(messages)
        self.logger.log_final_answer(question, final_answer, MAX_SEARCH_ROUNDS)

        final_messages_context = self._build_messages_summary(messages)

        yield {
            "step": "final",
            "round": MAX_SEARCH_ROUNDS,
            "final_answer": final_answer,
            "final_messages_context": final_messages_context,
            "final_prompt_type": f"达到最大搜索轮数({MAX_SEARCH_ROUNDS})，系统追加提示词强制生成回答",
            "force_prompt": final_prompt,
        }

    def _local_search_with_trace(self, question: str, trace: dict) -> str:
        """LLM 不可用时的降级搜索（带 trace 版本）：jieba 分词 → 规则式意图识别 → 各引擎直查。

        降级路径故意走"无 LLM"的硬编码分支，以保证在 API 挂掉时仍能返回有意义的业务数据
        （主要依赖 enterprise_sdk 里的模拟系统数据，而非 LLM 生成）。
        """
        try:
            import jieba
            keywords = list(jieba.cut_for_search(question.replace("?", "").replace("？", "")))
            stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "都", "一",
                         "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
                         "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些", "什么",
                         "怎么", "如何", "哪", "为什么", "多少", "几", "可以", "能", "应该",
                         "需要", "情况", "信息", "数据", "问题", "关于", "对于", "目前", "现在",
                         "今天", "本周", "本月", "本季度", "今年", "最近", "当前", "最新",
                         "怎样", "是否", "有没有", "能不能"}
            keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) > 1 and k.strip() not in stopwords]
        except ImportError:
            keywords = [question.replace("?", "").replace("？", "").strip()]

        if not keywords:
            keywords = [question]

        trace["fallback_keywords"] = keywords

        intent = self._detect_intent(question)
        trace["fallback_intent"] = intent

        all_results = []

        if intent == "finance":
            try:
                finance_result = self.enterprise_sdk.call("finance", "get_payments", {})
                if finance_result and "error" not in finance_result.lower():
                    all_results.append(("财务系统-支付流水", {"enterprise_finance": finance_result}))
            except Exception:
                pass
            try:
                txn_result = self.enterprise_sdk.call("finance", "get_transactions", {})
                if txn_result and "error" not in txn_result.lower():
                    all_results.append(("财务系统-交易流水", {"enterprise_finance": txn_result}))
            except Exception:
                pass
        elif intent == "hr":
            try:
                hr_result = self.enterprise_sdk.call("hr", "search_employees", {"keyword": keywords[0] if keywords else ""})
                if hr_result and "error" not in hr_result.lower():
                    all_results.append(("HR系统", {"enterprise": hr_result}))
            except Exception:
                pass
        elif intent == "project":
            try:
                proj_result = self.enterprise_sdk.call("project", "search_projects", {"keyword": keywords[0] if keywords else ""})
                if proj_result and "error" not in proj_result.lower():
                    all_results.append(("项目系统", {"enterprise_project": proj_result}))
            except Exception:
                pass

        trace["fallback_results"] = [{"label": label, "has_data": bool(res)} for label, res in all_results]

        if not all_results:
            return "抱歉，在本地数据源中没有找到与您问题相关的信息。"

        answer_parts = ["⚠️ API 连接失败，使用本地财务数据：\n"]
        for label, kw_results in all_results:
            if "enterprise_finance" in kw_results:
                try:
                    fin_data = json.loads(kw_results["enterprise_finance"])
                    if fin_data.get("data"):
                        answer_parts.append(f"**【{label}】**")
                        summary = fin_data.get("summary", {})
                        if summary:
                            answer_parts.append(f"汇总: 收入 {summary.get('total_income', 0):,} 元, 支出 {summary.get('total_expense', 0):,} 元, 净额 {summary.get('net_amount', 0):,} 元, 共 {summary.get('count', 0)} 笔")
                        for item in fin_data["data"][:5]:
                            answer_parts.append(f"- [{item.get('date', 'N/A')}] {item.get('type', 'N/A')} | {item.get('counterpart', 'N/A')} | {item.get('amount', 0):,} 元 | {item.get('description', 'N/A')} | 状态: {item.get('status', 'N/A')}")
                except Exception:
                    pass
            if "enterprise" in kw_results:
                try:
                    ent_data = json.loads(kw_results["enterprise"])
                    if ent_data.get("data"):
                        answer_parts.append(f"**【{label}】**")
                        for emp in ent_data["data"]:
                            answer_parts.append(f"- {emp.get('name', 'N/A')}: {emp.get('position', 'N/A')} ({emp.get('department', 'N/A')})")
                except Exception:
                    pass
            if "enterprise_project" in kw_results:
                try:
                    proj_data = json.loads(kw_results["enterprise_project"])
                    if proj_data.get("data"):
                        answer_parts.append(f"**【{label}】**")
                        for proj in proj_data["data"][:5]:
                            answer_parts.append(f"- {proj.get('name', 'N/A')}: 状态={proj.get('status', 'N/A')}, 进度={proj.get('progress', 'N/A')}%")
                except Exception:
                    pass

        return "\n".join(answer_parts)

    def _is_api_error(self, content: str) -> bool:
        """判断 LLM 返回的 content 是否是一条包装过的 API 错误消息。

        触发条件：content 能被解析成 JSON，且 dict 中含 "error" 字段，
        error 文本包含 connection / timeout / api / key / auth / unauthorized 等关键词。
        命中即让主循环走降级路径。
        """
        if not content:
            return False
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "error" in data:
                error_str = str(data["error"]).lower()
                return any(keyword in error_str for keyword in ["connection", "timeout", "api", "key", "auth", "unauthorized", "not found", "rate limit"])
        except json.JSONDecodeError:
            pass
        return False

    def _local_search(self, question: str, verbose: bool = True) -> str:
        """降级搜索的 verbose 版本（CLI 用）：流程同 _local_search_with_trace，但把每一步打到 stdout。"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 本地搜索模式: 搜索所有本地数据源")
            print(f"{'='*60}")

        try:
            import jieba
            keywords = list(jieba.cut_for_search(question.replace("?", "").replace("？", "")))
            stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "都", "一",
                         "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
                         "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些", "什么",
                         "怎么", "如何", "哪", "为什么", "多少", "几", "可以", "能", "应该",
                         "需要", "情况", "信息", "数据", "问题", "关于", "对于", "目前", "现在",
                         "今天", "本周", "本月", "本季度", "今年", "最近", "当前", "最新",
                         "怎样", "是否", "有没有", "能不能"}
            keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) > 1 and k.strip() not in stopwords]
        except ImportError:
            keywords = [question.replace("?", "").replace("？", "").strip()]

        if not keywords:
            keywords = [question]

        search_queries = keywords[:5]

        intent = self._detect_intent(question)

        all_results = []

        if intent == "finance":
            if verbose:
                print(f"\n[*] 检测到财务相关意图，优先搜索财务系统")
            try:
                finance_result = self.enterprise_sdk.call("finance", "get_payments", {})
                if finance_result and "error" not in finance_result.lower():
                    all_results.append(("财务系统-支付流水", {"enterprise_finance": finance_result}))
                    if verbose:
                        print(f"    ✅ 财务系统找到支付流水数据")
            except Exception as e:
                if verbose:
                    print(f"    ❌ 财务系统搜索失败: {e}")

            try:
                txn_result = self.enterprise_sdk.call("finance", "get_transactions", {})
                if txn_result and "error" not in txn_result.lower():
                    all_results.append(("财务系统-交易流水", {"enterprise_finance": txn_result}))
                    if verbose:
                        print(f"    ✅ 财务系统找到交易流水数据")
            except Exception as e:
                if verbose:
                    print(f"    ❌ 财务系统搜索失败: {e}")

        elif intent == "hr":
            if verbose:
                print(f"\n[*] 检测到HR相关意图，优先搜索HR系统")
            try:
                hr_result = self.enterprise_sdk.call("hr", "search_employees", {"keyword": keywords[0] if keywords else ""})
                if hr_result and "error" not in hr_result.lower():
                    all_results.append(("HR系统", {"enterprise": hr_result}))
                    if verbose:
                        print(f"    ✅ HR系统找到结果")
            except Exception as e:
                if verbose:
                    print(f"    ❌ HR系统搜索失败: {e}")

        elif intent == "project":
            if verbose:
                print(f"\n[*] 检测到项目相关意图，优先搜索项目系统")
            try:
                proj_result = self.enterprise_sdk.call("project", "search_projects", {"keyword": keywords[0] if keywords else ""})
                if proj_result and "error" not in proj_result.lower():
                    all_results.append(("项目系统", {"enterprise_project": proj_result}))
                    if verbose:
                        print(f"    ✅ 项目系统找到结果")
            except Exception as e:
                if verbose:
                    print(f"    ❌ 项目系统搜索失败: {e}")

        for kw in search_queries:
            if verbose:
                print(f"\n[*] 搜索关键词: {kw}")

            kw_results = {}

            try:
                db_result = self.db.search(kw)
                if db_result and db_result != "[]" and db_result != "{}":
                    kw_results["database"] = db_result
                    if verbose:
                        print(f"    ✅ SQLite数据库找到结果")
            except Exception as e:
                if verbose:
                    print(f"    ❌ SQLite搜索失败: {e}")

            try:
                vector_result = self.vector_db.search(kw, n_results=3)
                if vector_result and "error" not in vector_result.lower():
                    kw_results["vector_db"] = vector_result
                    if verbose:
                        print(f"    ✅ 向量数据库找到结果")
            except Exception as e:
                if verbose:
                    print(f"    ❌ 向量搜索失败: {e}")

            try:
                keyword_result = self.keyword_search.search(kw, limit=5)
                if keyword_result and "error" not in keyword_result.lower():
                    kw_results["keyword"] = keyword_result
                    if verbose:
                        print(f"    ✅ 关键词索引找到结果")
            except Exception as e:
                if verbose:
                    print(f"    ❌ 关键词搜索失败: {e}")

            if kw_results:
                all_results.append((kw, kw_results))

        if not all_results:
            answer = "抱歉，在本地数据源中没有找到与您问题相关的信息。"
        else:
            answer_parts = ["我通过搜索本地数据源，找到了以下相关信息：\n"]

            for kw, kw_results in all_results:
                if "enterprise_finance" in kw_results:
                    try:
                        fin_data = json.loads(kw_results["enterprise_finance"])
                        if fin_data.get("data"):
                            answer_parts.append(f"【财务系统 - {kw}】")
                            summary = fin_data.get("summary", {})
                            if summary:
                                answer_parts.append(f"  汇总: 收入 {summary.get('total_income', 0):,} 元, 支出 {summary.get('total_expense', 0):,} 元, 净额 {summary.get('net_amount', 0):,} 元, 共 {summary.get('count', 0)} 笔")
                            for item in fin_data["data"][:5]:
                                answer_parts.append(f"  - [{item.get('date', 'N/A')}] {item.get('type', 'N/A')} | {item.get('counterpart', 'N/A')} | {item.get('amount', 0):,} 元 | {item.get('description', 'N/A')} | 状态: {item.get('status', 'N/A')}")
                    except Exception:
                        pass

                if "enterprise_project" in kw_results:
                    try:
                        proj_data = json.loads(kw_results["enterprise_project"])
                        if proj_data.get("data"):
                            answer_parts.append(f"\n【项目系统 - {kw}】")
                            for proj in proj_data["data"][:5]:
                                answer_parts.append(f"  - {proj.get('name', 'N/A')}: 状态={proj.get('status', 'N/A')}, 进度={proj.get('progress', 'N/A')}%, 负责人={proj.get('lead', 'N/A')}")
                    except Exception:
                        pass

                if "enterprise" in kw_results:
                    try:
                        ent_data = json.loads(kw_results["enterprise"])
                        if ent_data.get("data"):
                            answer_parts.append(f"【企业HR系统 - 搜索关键词: {kw}】")
                            for emp in ent_data["data"]:
                                answer_parts.append(f"  - {emp.get('name', 'N/A')}: {emp.get('position', 'N/A')} ({emp.get('department', 'N/A')})")
                    except Exception:
                        pass

                if "database" in kw_results:
                    try:
                        db_data = json.loads(kw_results["database"])
                        if isinstance(db_data, dict):
                            for table, rows in db_data.items():
                                if rows and table == "employees":
                                    answer_parts.append(f"\n【数据库员工表 - 搜索关键词: {kw}】")
                                    for row in rows[:3]:
                                        answer_parts.append(f"  - {row.get('name', 'N/A')}: {row.get('position', 'N/A')} ({row.get('department', 'N/A')})")
                                elif rows and table == "departments":
                                    answer_parts.append(f"\n【数据库部门表 - 搜索关键词: {kw}】")
                                    for row in rows[:2]:
                                        answer_parts.append(f"  - {row.get('name', 'N/A')}: 负责人={row.get('manager', 'N/A')}")
                                elif rows and table == "contracts":
                                    answer_parts.append(f"\n【数据库合同表 - 搜索关键词: {kw}】")
                                    for row in rows[:3]:
                                        answer_parts.append(f"  - {row.get('client_name', 'N/A')}: 金额={row.get('amount', 0):,} 元, 状态={row.get('status', 'N/A')}")
                                elif rows and table == "projects":
                                    answer_parts.append(f"\n【数据库项目表 - 搜索关键词: {kw}】")
                                    for row in rows[:3]:
                                        answer_parts.append(f"  - {row.get('name', 'N/A')}: 状态={row.get('status', 'N/A')}, 预算={row.get('budget', 0):,} 元")
                                elif rows and table == "products":
                                    answer_parts.append(f"\n【数据库产品表 - 搜索关键词: {kw}】")
                                    for row in rows[:3]:
                                        answer_parts.append(f"  - {row.get('name', 'N/A')}: 类别={row.get('category', 'N/A')}, 价格={row.get('price', 0):,} 元")
                    except Exception:
                        pass

            if len(answer_parts) == 1:
                answer = "抱歉，在本地数据源中没有找到与您问题相关的信息。"
            else:
                answer = "\n".join(answer_parts)

        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ 本地搜索完成")
            print(f"{'='*60}")
            print(answer)

        self.logger.log_final_answer(question, answer, 0)
        return answer

    def _build_messages_summary(self, messages: list[dict]) -> list[dict]:
        """把 messages 列表压缩成适合前端展示的摘要。

        每条按 role 区分处理：
          system / user / tool → 只保留 content 与 200~300 字预览
          assistant 且含 tool_calls → 展开成 tool_info 列表（工具名+参数），预览写成"调用 N 个工具: [...]"
          assistant 纯文本 → 保留 content 与预览

        Streamlit 前端用这个摘要渲染"消息上下文"面板。
        """
        summary = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            entry = {"index": i + 1, "role": role}

            if role == "system":
                content = msg.get("content", "")
                entry["content"] = content
                entry["preview"] = content[:200] + "..." if len(content) > 200 else content
            elif role == "user":
                content = msg.get("content", "")
                entry["content"] = content
                entry["preview"] = content[:300] if len(content) > 300 else content
            elif role == "assistant":
                if msg.get("tool_calls"):
                    tool_info = []
                    tool_calls_raw = []
                    for tc in msg.get("tool_calls", []):
                        fn = tc.get("function", {})
                        tool_calls_raw.append({
                            "id": tc.get("id", ""),
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": fn.get("name", "unknown"),
                                "arguments": fn.get("arguments", "{}"),
                            }
                        })
                        try:
                            args = json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {})
                        except json.JSONDecodeError:
                            args = {}
                        tool_info.append({
                            "tool_name": fn.get("name", "unknown"),
                            "arguments": args,
                        })
                    entry["tool_calls"] = tool_info
                    entry["content"] = json.dumps({"role": role, "content": msg.get("content"), "tool_calls": tool_calls_raw}, ensure_ascii=False, indent=2)
                    entry["preview"] = f"调用 {len(tool_info)} 个工具: {[t['tool_name'] for t in tool_info]}"
                else:
                    content = msg.get("content", "")
                    entry["content"] = content
                    entry["preview"] = content[:300] if len(content) > 300 else content
            elif role == "tool":
                content = msg.get("content", "")
                entry["content"] = content
                entry["preview"] = content[:200] + "..." if len(content) > 200 else content
                entry["tool_call_id"] = msg.get("tool_call_id", "")

            summary.append(entry)
        return summary

    def _detect_intent(self, question: str) -> str:
        """基于关键词的规则式意图识别（降级路径用）。

        优先级：finance > hr > project（因为财务类关键词最具体，最容易先命中）。
        三个分支的识别结果直接决定降级搜索时走哪个 enterprise_sdk 调用。
        """
        finance_keywords = {"支付", "付款", "收款", "交易", "流水", "账单", "对账", "结算",
                           "收入", "支出", "到账", "转账", "报销", "发票", "预算", "费用",
                           "财务", "金额", "款项", "资金", "成本"}
        hr_keywords = {"员工", "人事", "组织", "架构", "请假", "入职", "离职", "考勤",
                      "薪资", "招聘", "培训", "HR", "人员"}
        project_keywords = {"项目", "进度", "里程碑", "交付", "迭代", "sprint",
                           "需求", "排期", "deadline"}

        question_lower = question.lower()
        for kw in finance_keywords:
            if kw in question_lower:
                return "finance"
        for kw in hr_keywords:
            if kw in question_lower:
                return "hr"
        for kw in project_keywords:
            if kw in question_lower:
                return "project"
        return "general"
