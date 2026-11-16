from __future__ import annotations

"""问答准确率评测：10 条用例，答案需命中全部关键事实点。

用法:
    .\\.venv\\Scripts\\python.exe eval_accuracy.py --dry-run   # 只看标准答案，不调模型
    .\\.venv\\Scripts\\python.exe eval_accuracy.py             # 跑全部 10 条
    .\\.venv\\Scripts\\python.exe eval_accuracy.py --ids A01,A07

判定: 每题包含若干"必需事实组"，每组给出若干等价写法（如 200 / 200人），
      答案中每组至少命中一个才算该题正确；准确率 = 正确题数 / 总题数。
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import DATA_DIR, DB_PATH  # noqa: E402


def query(sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def count(table: str, where: str = "", params: tuple = ()) -> str:
    sql = f"SELECT COUNT(*) FROM {table}" + (f" WHERE {where}" if where else "")
    return str(query(sql, params)[0][0])


def first_value(sql: str) -> str:
    return str(query(sql)[0][0])


# required 中每一项是"必需事实组"，组内任一写法命中即算通过；
# "fn" 表示标准答案在运行时从数据库计算，避免与数据版本耦合。
CASES = [
    {
        "id": "A01", "type": "结构化-计数",
        "question": "公司目前一共有多少名员工？",
        "fn": lambda: [[count("employees")]],
    },
    {
        "id": "A02", "type": "结构化-计数",
        "question": "技术部现在有多少名员工？",
        "fn": lambda: [[count("employees", "department = ?", ("技术部",))]],
    },
    {
        "id": "A03", "type": "结构化-计数",
        "question": "公司目前一共有多少份合同？",
        "fn": lambda: [[count("contracts")]],
    },
    {
        "id": "A04", "type": "结构化-计数",
        "question": "公司现在一共有多少种产品？",
        "fn": lambda: [[count("products")]],
    },
    {
        "id": "A05", "type": "结构化-极值",
        "question": "公司里薪资最高的员工是谁？他的薪资是多少？",
        "fn": lambda: [[first_value("SELECT name FROM employees ORDER BY salary DESC LIMIT 1")]],
    },
    {
        "id": "A06", "type": "结构化-极值",
        "question": "合同金额最高的客户是哪一家？金额是多少？",
        "fn": lambda: [[first_value("SELECT client_name FROM contracts ORDER BY amount DESC LIMIT 1")]],
    },
    {
        "id": "A07", "type": "政策-关键词",
        "question": "按照公司的年假制度，工作满5年的员工可以享受多少天年假？",
        "required": [["10天", "十天"]],
    },
    {
        "id": "A08", "type": "政策-关键词",
        "question": "报销需要在费用发生后多少天内提交？",
        "required": [["30天", "三十天"]],
    },
    {
        "id": "A09", "type": "代码-定位",
        "question": "代码仓库里 Redis 工具封装在哪个文件？",
        "required": [["redis.py"]],
    },
    {
        "id": "A10", "type": "会议-语义",
        "question": "技术架构评审会议决定把数据库从什么迁移到什么？",
        "required": [["oracle"], ["mysql"]],
    },
]


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "").replace("，", "").replace(" ", "").replace("\u3000", "")
    return text


def build_expected() -> None:
    for case in CASES:
        if "fn" in case:
            case["required"] = case["fn"]()


def check(case: dict, answer: str) -> tuple[bool, list[str]]:
    text = normalize(answer)
    missing = []
    for group in case["required"]:
        if not any(normalize(alt) in text for alt in group):
            missing.append("/".join(group))
    return (not missing), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="问答准确率评测（10 条）")
    parser.add_argument("--ids", type=str, default=None, help="逗号分隔的用例 ID，如 A01,A07")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="只打印标准答案，不调用模型")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(DB_PATH) or not query("SELECT name FROM sqlite_master WHERE type='table'"):
        print("[!] 数据库为空，请先执行: .\\.venv\\Scripts\\python.exe seed_data_large.py")
        return

    build_expected()

    cases = CASES
    if args.ids:
        wanted = {x.strip().upper() for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    print(f"用例数: {len(cases)}\n")
    if args.dry_run:
        for c in cases:
            print(f"{c['id']}  {c['question']}")
            for group in c["required"]:
                print(f"      标准答案: {' 或 '.join(group)}")
        return

    from core.agent import AgenticSearchAgent

    agent = AgenticSearchAgent()
    results = []

    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['id']} {case['question']}")
        start = time.time()
        try:
            trace = agent.search_with_trace(case["question"])
            answer = trace.get("final_answer", "") or ""
            passed, missing = check(case, answer)
            record = {
                "id": case["id"], "type": case["type"], "question": case["question"],
                "expected": case["required"], "answer": answer,
                "passed": passed, "missing": missing,
                "rounds": trace.get("total_rounds", 0),
                "latency_sec": round(time.time() - start, 2),
                "api_available": trace.get("api_available", True),
            }
        except Exception as exc:
            record = {
                "id": case["id"], "type": case["type"], "question": case["question"],
                "expected": case["required"], "answer": f"ERROR: {exc}",
                "passed": False, "missing": ["<异常>"],
                "rounds": 0, "latency_sec": round(time.time() - start, 2),
                "api_available": False,
            }

        results.append(record)
        status = "PASS" if record["passed"] else "FAIL"
        print(f"    {status} | {record['latency_sec']}s | 轮数 {record['rounds']}"
              + (f" | 缺失: {', '.join(record['missing'])}" if record["missing"] else ""))
        if not record["passed"]:
            print(f"    答案: {record['answer'][:200]}")
        time.sleep(args.pause)

    total = len(results)
    passed_n = sum(1 for r in results if r["passed"])
    summary = {
        "total": total,
        "correct": passed_n,
        "accuracy": round(passed_n / total, 3),
        "avg_latency_sec": round(sum(r["latency_sec"] for r in results) / total, 2),
        "avg_rounds": round(sum(r["rounds"] for r in results) / total, 2),
        "api_failures": sum(1 for r in results if not r["api_available"]),
    }

    print("\n" + "=" * 60)
    print(f"准确率: {passed_n}/{total} = {summary['accuracy']:.1%}")
    print(f"平均耗时: {summary['avg_latency_sec']}s | 平均轮数: {summary['avg_rounds']}")
    print(f"API 降级: {summary['api_failures']} 次")
    for ctype in sorted({r["type"] for r in results}):
        group = [r for r in results if r["type"] == ctype]
        print(f"  {ctype}: {sum(1 for r in group if r['passed'])}/{len(group)}")

    out = args.json_out or os.path.join(
        DATA_DIR, "eval", f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
