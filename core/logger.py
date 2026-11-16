# 允许在类/函数的类型注解中使用尚未定义的名称（如 Logger 自身返回类型）
from __future__ import annotations

import logging  # Python 标准库，提供日志记录的框架与处理器
import json     # 用于解析与生成 JSON 格式的日志条目
import os       # 处理文件路径与目录
from typing import Optional  # 类型注解用，允许值为 None
from datetime import datetime  # 生成时间戳

from config import LOGS_DIR  # 从项目配置中读取日志目录（data/logs）


class Logger:
    """单例日志工具：同时输出控制台、文本日志与 JSONL 结构化日志。

    单例通过覆写 __new__ 实现，保证全局只有一个 Logger 实例，
    从而避免在不同模块里重复创建 handler 导致日志重复。
    """

    _instance = None  # 单例的唯一实例，类级别共享

    def __new__(cls, *args, **kwargs):
        """控制实例化过程，确保全局只创建一个 Logger。"""
        if cls._instance is None:                 # 还没创建过实例
            cls._instance = super().__new__(cls)  # 调用父类 __new__ 分配对象
            cls._instance._initialized = False    # 标记尚未走 __init__
        return cls._instance                      # 每次都返回同一个对象

    def __init__(self, name: str = "agentic_search"):
        """初始化三个输出通道（控制台、文本、JSONL）。仅在首次实例化时执行一次。"""
        if self._initialized:   # 已经初始化过就跳过，避免重复添加 handler
            return
        self._initialized = True  # 打标记

        self.name = name  # 日志器名称，同时也是文件名前缀
        self.logger = logging.getLogger(name)  # 拿到底层的 logging.Logger 实例
        self.logger.setLevel(logging.DEBUG)    # 根级别设为 DEBUG，让所有级别都能被 handler 决定输出与否

        if not self.logger.handlers:  # 只有没有 handler 时才加，避免重复
            # —— 通道 1：控制台，INFO 级别以上，格式简洁 ——
            console_handler = logging.StreamHandler()  # 输出到 stderr
            console_handler.setLevel(logging.INFO)      # 控制台只显示 INFO 及更高
            console_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_format)

            # —— 通道 2：文本日志文件，DEBUG 级别，带函数名和行号 ——
            log_file = os.path.join(LOGS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")  # 按日期滚动（每天一个文件）
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)

            # —— 通道 3：JSONL 结构化日志，每行一个 JSON 对象 ——
            json_log_file = os.path.join(LOGS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d')}.json")
            self.json_handler = JsonLogHandler(json_log_file)  # 自定义 Handler，见文件末尾
            self.json_handler.setLevel(logging.DEBUG)

            # 三个 handler 全部挂上
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(self.json_handler)

    # —— 基础日志接口，透传到底层 Logger，**kwargs 会作为 extra 进入 LogRecord ——
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)

    # —— 业务语义化封装：把 Agent 的多轮行为结构化写入日志 ——

    def log_search(self, engine: str, query: str, results_summary: str, round_num: int):
        """记录一次引擎检索。"""
        self.info(
            f"[Round {round_num}] Search: engine={engine}, query={query}",
            extra={
                "search_engine": engine,          # 使用的引擎名（database_sql / vector_search 等）
                "query": query,                   # 传给引擎的参数（通常是 JSON 字符串）
                "results_summary": results_summary,  # 结果摘要，不写全文以避免日志爆炸
                "round": round_num                # 第几轮
            }
        )

    def log_decision(self, round_num: int, decision: str, reasoning: str):
        """记录 Agent 在某一轮的决策与推理。"""
        self.info(
            f"[Round {round_num}] Decision: {decision}",
            extra={
                "round": round_num,
                "decision": decision,      # 工具名，如 Call database_search / Final Answer
                "reasoning": reasoning     # LLM 的思考过程
            }
        )

    def log_final_answer(self, question: str, answer: str, total_rounds: int):
        """记录最终答案，answer 只保留前 200 字符。"""
        self.info(
            f"[Final] Answer after {total_rounds} rounds",
            extra={
                "question": question,
                "answer": answer[:200],
                "total_rounds": total_rounds
            }
        )

    # —— 让 Agent 能反查自己的历史检索日志 ——

    def search_logs(self, keyword: str = None, level: str = None,
                    engine: str = None, limit: int = 20) -> str:
        """读取最新的 JSONL 日志文件，按条件过滤后返回 JSON 字符串。

        这个方法本身会被注册为 Agent 的工具 `log_search`，使 Agent 在遇到
        连续失败时能回头检视上一轮用了什么引擎、传了什么参数。
        """
        # 找最新的 JSON 日志文件（按文件名倒序，取第一个）
        log_file = None
        for f in sorted(os.listdir(LOGS_DIR), reverse=True):
            if f.endswith(".json") and f.startswith(self.name):
                log_file = os.path.join(LOGS_DIR, f)
                break

        if not log_file or not os.path.exists(log_file):
            return json.dumps({"error": "No log file found"}, ensure_ascii=False)

        results = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:  # JSONL 格式：每行一个 JSON 对象
                try:
                    entry = json.loads(line.strip())
                    # keyword 匹配整个条目（时间 + 消息 + extra 字段）
                    if keyword and keyword.lower() not in json.dumps(entry, ensure_ascii=False).lower():
                        continue
                    if level and entry.get("level", "").upper() != level.upper():
                        continue
                    if engine and entry.get("search_engine") != engine:
                        continue
                    results.append(entry)
                except (json.JSONDecodeError, AttributeError):
                    continue  # 某行解析失败就跳过，不影响整体

        return json.dumps(results[-limit:], ensure_ascii=False, indent=2, default=str)


class JsonLogHandler(logging.Handler):
    """自定义 logging.Handler：把 LogRecord 序列化成 JSON 写入文件。

    Python 自带的 FileHandler 只能写文本格式，这里自己实现 emit()
    把每条日志拼成一行 JSON，从而形成 JSONL（JSON Lines）格式。
    """

    def __init__(self, log_file: str):
        super().__init__()
        self.log_file = log_file

    def emit(self, record):
        """被 logging 框架在有日志要写入时自动调用。"""
        try:
            # 每个 JSON 条目的基础字段
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),  # 创建时间
                "level": record.levelname,                                          # INFO / DEBUG / ...
                "message": record.getMessage(),                                     # 日志消息正文
                "module": record.module,                                            # 所在模块
                "function": record.funcName,                                        # 所在函数
                "line": record.lineno,                                              # 所在行号
            }

            # 业务日志专用字段：如果 record 上存在这些 extra 属性就追加进去
            for key in ["search_engine", "query", "results_summary", "round",
                        "decision", "reasoning", "question", "answer", "total_rounds"]:
                if hasattr(record, key):
                    log_entry[key] = getattr(record, key)

            # 追加写入 JSONL 文件，每条一行
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 日志写入失败不影响主流程，静默吞掉
