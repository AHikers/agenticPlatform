from __future__ import annotations

"""Agentic Search 评测脚本.

用法:
    .\\.venv\\Scripts\\python.exe eval_agent.py                # 跑全部 24 条
    .\\.venv\\Scripts\\python.exe eval_agent.py --limit 5      # 只跑前 5 条
    .\\.venv\\Scripts\\python.exe eval_agent.py --ids C01,S05  # 指定用例
    .\\.venv\\Scripts\\python.exe eval_agent.py --dry-run      # 只校验用例定义，不调模型

指标:
    通过率      用例期望的工具组是否被覆盖
    首选集准确率 第一个"有意义"的工具是否落在期望组内
    工具覆盖率   期望工具组被覆盖的比例
    平均轮数/调用次数/耗时
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import DATA_DIR  # noqa: E402

# 探查类工具：用于先看表结构/集合信息，不计入"首选集"判定
EXPLORATORY_TOOLS = {"database_schema", "vector_info", "keyword_info", "enterprise_systems"}

# groups: 每个元素是一组"可接受的等价工具"
# mode="all" 表示每个组都要被覆盖（跨源问题），"any" 表示覆盖任意一组即可
CASES = [
    {
        "id": "S01", "category": "结构化-部门",
        "question": "公司目前有哪些部门？各部门负责人分别是谁？",
        "groups": [["database_search", "database_sql", "enterprise_call"]], "mode": "any",
    },
    {
        "id": "S02", "category": "结构化-员工",
        "question": "技术部现在有多少人？平均薪资大概是多少？",
        "groups": [["database_search", "database_sql"]], "mode": "any",
        "keyword": "技术部",
    },
    {
        "id": "S03", "category": "结构化-合同",
        "question": "公司目前有哪些仍在执行中的合同？涉及金额最大的是哪一个？",
        "groups": [["database_search", "database_sql"]], "mode": "any",
        "keyword": "合同",
    },
    {
        "id": "S04", "category": "结构化-产品",
        "question": "我们目前有哪些产品？各自的定价是多少？",
        "groups": [["database_search", "database_sql"]], "mode": "any",
    },
    {
        "id": "S05", "category": "财务-支付",
        "question": "本周的支付数据情况如何？收入和支出各是多少？",
        "groups": [["enterprise_call"]], "mode": "any",
        "keyword": "支付",
    },
    {
        "id": "S06", "category": "财务-交易",
        "question": "本月的交易流水按收入、支出汇总分别是多少？",
        "groups": [["enterprise_call"]], "mode": "any",
    },
    {
        "id": "S07", "category": "财务-发票",
        "question": "目前有哪些待处理的发票？金额合计多少？",
        "groups": [["enterprise_call"]], "mode": "any",
    },
    {
        "id": "S08", "category": "财务-预算",
        "question": "各部门今年的预算分别是多少？哪个部门预算最高？",
        "groups": [["enterprise_call"]], "mode": "any",
        "keyword": "预算",
    },
    {
        "id": "S09", "category": "HR-请假",
        "question": "最近有哪些员工请假了？请假原因是什么？",
        "groups": [["enterprise_call"]], "mode": "any",
    },
    {
        "id": "S10", "category": "HR-组织",
        "question": "公司的组织架构是怎样的？技术部下面有哪些小组？",
        "groups": [["enterprise_call", "database_search", "database_sql"]], "mode": "any",
    },
    {
        "id": "S11", "category": "项目-进度",
        "question": "目前哪些项目存在延期风险？各自进度如何？",
        "groups": [["enterprise_call", "database_search", "database_sql"]], "mode": "any",
    },
    {
        "id": "S12", "category": "政策-关键词",
        "question": "公司的年假政策是怎么规定的？入职不满一年怎么计算？",
        "groups": [["keyword_search", "enterprise_call"]], "mode": "any",
        "keyword": "年假",
    },
    {
        "id": "S13", "category": "政策-关键词",
        "question": "员工报销流程需要哪些步骤？要准备哪些材料？",
        "groups": [["keyword_search", "enterprise_call"]], "mode": "any",
        "keyword": "报销",
    },
    {
        "id": "S14", "category": "技术-语义",
        "question": "我们系统的微服务架构采用了哪些技术选型？",
        "groups": [["vector_search", "keyword_search", "enterprise_call"]], "mode": "any",
    },
    {
        "id": "S15", "category": "技术-语义",
        "question": "系统里数据库和缓存分别用的是什么方案？",
        "groups": [["vector_search", "keyword_search", "code_search", "enterprise_call"]], "mode": "any",
    },
    {
        "id": "S16", "category": "会议-语义",
        "question": "最近一次技术评审会议讨论了哪些问题？形成了什么结论？",
        "groups": [["vector_search"]], "mode": "any",
    },
    {
        "id": "S17", "category": "代码-实现",
        "question": "代码里用户登录和权限鉴权是怎么实现的？",
        "groups": [["code_search", "code_read", "code_list"]], "mode": "any",
        "keyword": "auth",
    },
    {
        "id": "S18", "category": "代码-实现",
        "question": "项目里的 Redis 缓存是在哪个文件里配置的？",
        "groups": [["code_search", "code_read", "code_list"]], "mode": "any",
        "keyword": "redis",
    },
    {
        "id": "S19", "category": "日志-自省",
        "question": "查一下历史日志，之前有没有检索失败或者 API 报错的记录？",
        "groups": [["log_search"]], "mode": "any",
    },
    {
        "id": "C01", "category": "跨源-政策+编制",
        "question": "公司的年假政策是怎么规定的？技术部现在有多少人？",
        "groups": [["keyword_search", "enterprise_call"], ["database_search", "database_sql", "enterprise_call"]],
        "mode": "all",
    },
    {
        "id": "C02", "category": "跨源-财务+合同",
        "question": "本周支付流水里金额最大的那笔支出，对应合同的付款条款是怎么约定的？",
        "groups": [["enterprise_call"], ["database_search", "database_sql"]], "mode": "all",
    },
    {
        "id": "C03", "category": "跨源-代码+文档",
        "question": "用户登录接口的实现代码在哪个文件？技术文档里对鉴权方案是怎么说明的？",
        "groups": [["code_search", "code_read", "code_list"], ["keyword_search", "vector_search"]],
        "mode": "all",
    },
    {
        "id": "C04", "category": "跨源-产品+制度",
        "question": "介绍一下公司的产品线，再说明报销制度对差旅费用的规定。",
        "groups": [["database_search", "database_sql"], ["keyword_search", "enterprise_call"]],
        "mode": "all",
    },
    {
        "id": "C05", "category": "跨源-会议+代码",
        "question": "最近一次会议提到的系统架构问题，在代码仓库里对应哪个服务的哪个文件？",
        "groups": [["vector_search"], ["code_search", "code_read", "code_list"]],
        "mode": "all",
    },
]


def first_meaningful_tool(called: list[str]) -> str:
    for name in called:
        if name not in EXPLORATORY_TOOLS:
            return name
    return called[0] if called else ""


def evaluate(case: dict, trace: dict, elapsed: float) -> dict:
    called: list[str] = []
    for rnd in trace.get("rounds", []):
        for tc in rnd.get("tool_calls", []):
            called.append(tc.get("tool_name", ""))

    groups = case["groups"]
    covered = [any(tool in called for tool in group) for group in groups]
    passed = all(covered) if case.get("mode") == "all" else any(covered)

    first = first_meaningful_tool(called)
    hit1 = any(first in group for group in groups)

    answer = trace.get("final_answer", "") or ""
    keyword = case.get("keyword")
    keyword_hit = None if not keyword else (keyword.lower() in answer.lower())

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "passed": passed,
        "hit1": hit1,
        "recall": round(sum(covered) / len(groups), 3),
        "expected_groups": groups,
        "called_tools": called,
        "rounds": trace.get("total_rounds", 0),
        "tool_calls": len(called),
        "latency_sec": round(elapsed, 2),
        "api_available": trace.get("api_available", True),
        "keyword": keyword,
        "keyword_hit": keyword_hit,
        "answer_preview": answer[:300],
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    kw_results = [r for r in results if r["keyword_hit"] is not None]
    latencies = [r["latency_sec"] for r in results]
    return {
        "total_cases": total,
        "pass_rate": round(sum(1 for r in results if r["passed"]) / total, 3),
        "hit1_rate": round(sum(1 for r in results if r["hit1"]) / total, 3),
        "avg_recall": round(sum(r["recall"] for r in results) / total, 3),
        "avg_rounds": round(sum(r["rounds"] for r in results) / total, 2),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in results) / total, 2),
        "avg_latency_sec": round(sum(latencies) / total, 2),
        "max_latency_sec": round(max(latencies), 2),
        "api_failures": sum(1 for r in results if not r["api_available"]),
        "keyword_cases": len(kw_results),
        "keyword_hit_rate": round(sum(1 for r in kw_results if r["keyword_hit"]) / len(kw_results), 3) if kw_results else None,
    }


def print_report(results: list[dict], summary: dict) -> None:
    print("\n" + "=" * 100)
    print("用例明细")
    print("=" * 100)
    print(f"{'ID':<5}{'结果':<6}{'首选':<6}{'覆盖':<6}{'轮数':<5}{'调用':<5}{'耗时(s)':<9}{'期望工具':<44}实际工具")
    for r in results:
        expected = " | ".join("/".join(g) for g in r["expected_groups"])
        print(
            f"{r['id']:<5}"
            f"{'PASS' if r['passed'] else 'FAIL':<6}"
            f"{'OK' if r['hit1'] else 'NO':<6}"
            f"{r['recall']:<6}"
            f"{r['rounds']:<5}"
            f"{r['tool_calls']:<5}"
            f"{r['latency_sec']:<9}"
            f"{expected[:42]:<44}"
            f"{', '.join(r['called_tools'])}"
        )

    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"用例数:            {summary['total_cases']}")
    print(f"通过率:            {summary['pass_rate']:.1%}")
    print(f"首选集准确率:      {summary['hit1_rate']:.1%}")
    print(f"期望工具覆盖率:    {summary['avg_recall']:.1%}")
    print(f"平均检索轮数:      {summary['avg_rounds']}")
    print(f"平均工具调用次数:  {summary['avg_tool_calls']}")
    print(f"平均耗时:          {summary['avg_latency_sec']}s (最大 {summary['max_latency_sec']}s)")
    print(f"API 降级次数:      {summary['api_failures']}")
    if summary["keyword_hit_rate"] is not None:
        print(f"答案关键词命中率:  {summary['keyword_hit_rate']:.1%} ({summary['keyword_cases']} 条带关键词用例)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Search 评测")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条用例")
    parser.add_argument("--ids", type=str, default=None, help="逗号分隔的用例 ID，如 S05,C01")
    parser.add_argument("--dry-run", action="store_true", help="只校验用例定义，不调用模型")
    parser.add_argument("--json-out", type=str, default=None, help="结果 JSON 输出路径")
    parser.add_argument("--pause", type=float, default=1.0, help="用例之间的间隔秒数")
    args = parser.parse_args()

    cases = CASES
    if args.ids:
        wanted = {x.strip().upper() for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("[!] 没有匹配的用例")
        return

    if args.dry_run:
        print(f"用例定义校验通过，共 {len(cases)} 条：")
        for c in cases:
            assert c["groups"] and c["mode"] in {"any", "all"}, f"用例 {c['id']} 定义有误"
            print(f"  {c['id']:<5}{c['category']:<14}{c['question']}")
        return

    if not os.path.exists(os.path.join(DATA_DIR, "db", "enterprise.db")):
        print("[!] 未找到 data/db/enterprise.db，请先执行一次 .\\\\.venv\\\\Scripts\\\\python.exe main.py 初始化数据")
        return

    from core.agent import AgenticSearchAgent

    print(f"[*] 初始化 Agent，共 {len(cases)} 条用例")
    agent = AgenticSearchAgent()

    results = []
    for idx, case in enumerate(cases, 1):
        print(f"\n[{idx}/{len(cases)}] {case['id']} {case['question']}")
        start = time.time()
        try:
            trace = agent.search_with_trace(case["question"])
        except Exception as exc:  # 单条失败不影响整体评测
            print(f"    [!] 用例异常: {exc}")
            results.append({
                "id": case["id"], "category": case["category"], "question": case["question"],
                "passed": False, "hit1": False, "recall": 0.0,
                "expected_groups": case["groups"], "called_tools": [],
                "rounds": 0, "tool_calls": 0, "latency_sec": round(time.time() - start, 2),
                "api_available": False, "keyword": case.get("keyword"), "keyword_hit": None,
                "answer_preview": f"ERROR: {exc}",
            })
            continue

        record = evaluate(case, trace, time.time() - start)
        results.append(record)
        print(
            f"    {'PASS' if record['passed'] else 'FAIL'} | 轮数 {record['rounds']} "
            f"| 工具 {record['called_tools']} | {record['latency_sec']}s"
        )
        time.sleep(args.pause)

    summary = summarize(results)
    print_report(results, summary)

    out_path = args.json_out or os.path.join(
        DATA_DIR, "eval", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
