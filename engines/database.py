# -*- coding: utf-8 -*-
# SQLite 结构化数据引擎：封装建表、模糊搜索、SQL 直查、Schema 探查与批量插入。
# 被 Agent 注册为三个工具：database_search / database_sql / database_schema。
# search() 内部用 jieba 做中文分词后拼接 LIKE 条件，execute_sql() 允许 LLM 直接写 SQL。
from __future__ import annotations
import sqlite3
import json
import re
from typing import List, Dict, Optional
from config import DB_PATH

# 中文停用词集合：_extract_keywords 分词后会过滤掉这些常见词，
# 另外还包含一些 Agent 查询特有的无意义词（"情况"、"信息"、"数据"、时间词等）。
STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "怎么", "如何", "哪",
    "为什么", "多少", "几", "可以", "能", "应该", "需要", "情况", "信息", "数据",
    "问题", "关于", "对于", "目前", "现在", "今天", "本周", "本月", "本季度", "今年",
    "最近", "当前", "最新", "如何", "怎样", "是否", "有没有", "能不能",
}


def _extract_keywords(query: str) -> list[str]:
    """从 Agent 的查询文本中提取有意义的关键词。

    优先用 jieba 的 search 模式分词（适合搜索引擎场景，会输出更细粒度的词条），
    然后过滤掉停用词、单字（len<=1）。jieba 不可用时退化为正则切分。
    全部被过滤掉时兜底返回原 query，避免后续 SQL 拼接为空。
    """
    try:
        import jieba
        words = list(jieba.cut_for_search(query))
        keywords = [w.strip() for w in words if w.strip() and len(w.strip()) > 1 and w.strip() not in STOP_WORDS]
    except ImportError:
        # jieba 没装时的降级路径：按标点和空白切分
        keywords = re.split(r'[\s,，、；;？?！!。.]+', query)
        keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) > 1]
        keywords = [k for k in keywords if k not in STOP_WORDS]
    if not keywords:
        keywords = [query]  # 兜底：把整句当一个关键词
    return keywords


class DatabaseEngine:
    """SQLite 数据库封装类：连接管理 + 模糊搜索 + SQL 直查 + Schema 探查 + 批量写入。

    连接策略：每次操作前 connect()、操作后 close()，不跨方法复用连接。
    SQLite 的单文件模式在并发写时会报 "database is locked"，这种短连接策略可以缓解。
    """

    def __init__(self):
        self.db_path = DB_PATH
        self.conn = None

    def connect(self):
        """建立 SQLite 连接，设置 row_factory 让查询结果可以用列名访问。"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """关闭连接并置空，避免 close() 被重复调用时报错。"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def init_tables(self):
        """创建 5 张业务表（若不存在）。供 seed_data_large.py 和首次启动时调用。

        表结构是演示数据的核心，没有外键约束——保持简单以便 seed 时自由插入。
        """
        self.connect()
        cursor = self.conn.cursor()

        # —— 员工表 ——
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT,
                position TEXT,
                email TEXT,
                phone TEXT,
                hire_date TEXT,
                salary REAL,
                status TEXT DEFAULT 'active'
            )
        """)

        # —— 部门表 ——
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS departments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                manager TEXT,
                budget REAL,
                headcount INTEGER,
                location TEXT
            )
        """)

        # —— 项目表 ——
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT,
                lead TEXT,
                status TEXT,
                start_date TEXT,
                end_date TEXT,
                budget REAL,
                description TEXT
            )
        """)

        # —— 合同表 ——
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_name TEXT NOT NULL,
                type TEXT,
                amount REAL,
                status TEXT,
                sign_date TEXT,
                expire_date TEXT,
                responsible_person TEXT,
                description TEXT
            )
        """)

        # —— 产品表 ——
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                price REAL,
                stock INTEGER,
                description TEXT,
                launch_date TEXT
            )
        """)

        self.conn.commit()
        self.close()

    def search(self, query: str, table: str = None, limit: int = 10) -> str:
        """在指定表（或全部 5 张表）中对关键词做 LIKE 模糊匹配。

        搜索策略：先 jieba 分词 + 停用词过滤得到 keywords，
        再对每张表的每一列拼 "col LIKE ?" 条件，多个关键词 OR 连接。

        table 为 None 时会遍历全部 5 张表，返回 {表名: [行...]} 的嵌套结构；
        指定表名时直接返回行列表。返回值统一为 JSON 字符串，便于 Agent 写入 tool 消息。

        注意：这个方法**不支持排序和聚合**，适合简单的关键词检索。
        复杂查询（COUNT/SUM/ORDER BY/GROUP BY）应走 execute_sql()。
        """
        self.connect()
        try:
            cursor = self.conn.cursor()
            keywords = _extract_keywords(query)

            if not keywords:
                return json.dumps([], ensure_ascii=False, indent=2, default=str)

            # —— 指定表的场景 ——
            if table and table in ["employees", "departments", "projects", "contracts", "products"]:
                # 先查表结构拿到列名（SELECT * LIMIT 0 不返回行但会设置 cursor.description）
                sql = f"SELECT * FROM {table} LIMIT 0"
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description]

                # 对每列 × 每个关键词 拼 LIKE 条件
                conditions = []
                params = []
                for col in columns:
                    for kw in keywords:
                        conditions.append(f"{col} LIKE ?")
                        params.append(f"%{kw}%")

                where_clause = " OR ".join(conditions)
                sql = f"SELECT * FROM {table} WHERE {where_clause} LIMIT ?"
                params.append(limit)

                cursor.execute(sql, params)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
            else:
                # —— 全表扫描的场景 ——
                tables = ["employees", "departments", "projects", "contracts", "products"]
                all_results = {}
                for t in tables:
                    sql = f"SELECT * FROM {t} LIMIT 0"
                    cursor.execute(sql)
                    columns = [desc[0] for desc in cursor.description]

                    conditions = []
                    params = []
                    for col in columns:
                        for kw in keywords:
                            conditions.append(f"{col} LIKE ?")
                            params.append(f"%{kw}%")

                    where_clause = " OR ".join(conditions)
                    sql = f"SELECT * FROM {t} WHERE {where_clause} LIMIT ?"
                    params.append(limit)

                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                    table_results = [dict(row) for row in rows]
                    if table_results:
                        all_results[t] = table_results

                results = all_results

            return json.dumps(results, ensure_ascii=False, indent=2, default=str)
        finally:
            self.close()

    def execute_sql(self, sql: str) -> str:
        """直接执行 LLM 生成的 SQL 语句。返回 JSON 字符串。

        - SELECT 语句：返回结果行列表（每行是 {列名: 值} 的 dict）
        - 其他语句（INSERT/UPDATE/DELETE）：返回 {"affected_rows": N}
        - 异常：返回 {"error": "..."}，不让异常抛到外层
        """
        self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            if sql.strip().upper().startswith("SELECT"):
                # SELECT：取列名 + 行数据，拼成 dict 列表
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                results = [dict(zip(columns, row)) for row in rows]
                return json.dumps(results, ensure_ascii=False, indent=2, default=str)
            else:
                # 写操作：提交并返回受影响行数
                self.conn.commit()
                return json.dumps({"affected_rows": cursor.rowcount}, ensure_ascii=False)
        except Exception as e:
            # SQL 执行出错不抛异常，包装成 JSON 返回给 Agent
            # Agent 可以看到错误信息后改写 SQL 重试，而不是整条链路中断
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        finally:
            self.close()

    def get_schema(self) -> str:
        """查询所有表的 Schema 信息：列名、类型、行数。

        Agent 会在调用 database_sql 之前先用这个方法探查表结构，
        避免凭空猜列名导致 SQL 语法错误。
        """
        self.connect()
        try:
            cursor = self.conn.cursor()
            # sqlite_master 里存了所有表定义
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            schema_info = {}
            for table in tables:
                # PRAGMA table_info 返回 (cid, name, type, notnull, dflt_value, pk)
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
                # 顺便查表行数，让 Agent 对数据量有概念
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                schema_info[table] = {"columns": columns, "row_count": count}

            return json.dumps(schema_info, ensure_ascii=False, indent=2)
        finally:
            self.close()

    def insert_data(self, table: str, data: list[dict]) -> int:
        """批量插入数据。逐行 INSERT，返回实际插入的行数。

        注意：这个方法是**纯追加、无幂等保护**——重复调用会导致数据重复。
        seed_data_large.py 每次全量重跑时应先清空表或删除 data/ 目录。
        """
        self.connect()
        try:
            cursor = self.conn.cursor()
            if not data:
                return 0

            # 从第一条数据拿到列名，用 ? 占位符防 SQL 注入
            columns = list(data[0].keys())
            placeholders = ", ".join(["?"] * len(columns))
            col_str = ", ".join(columns)

            count = 0
            for row in data:
                values = [row.get(col) for col in columns]
                cursor.execute(f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})", values)
                count += 1

            self.conn.commit()
            return count
        finally:
            self.close()
