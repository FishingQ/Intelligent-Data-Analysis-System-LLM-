"""
训练数据集评测脚本
对结构化 NL2SQL 和非结构化 RAG 任务进行批量评测

用法:
    python scripts/run_eval.py --dataset 金融 --limit 10     # 评测指定数据集
    python scripts/run_eval.py --all --limit 5                # 全部数据集
    python scripts/run_eval.py --structured-only              # 仅结构化
    python scripts/run_eval.py --output report.json           # 输出评测报告
"""
import sys
import os
import json
import time
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class EvalResult:
    """单条评测结果"""
    task_id: Any
    question: str
    success: bool = False
    generated_sql: str = ""
    expected_sql: str = ""
    sql_match: bool = False
    answer_text: str = ""
    expected_answer: str = ""
    elapsed_ms: float = 0
    error: str = ""


@dataclass
class EvalReport:
    """评测报告"""
    dataset_name: str
    total: int = 0
    success: int = 0
    sql_exact_match: int = 0
    sql_partial_match: int = 0
    avg_elapsed_ms: float = 0
    results: List[EvalResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0

    @property
    def sql_accuracy(self) -> float:
        return self.sql_partial_match / self.total if self.total > 0 else 0


def normalize_sql(sql: str) -> str:
    """标准化 SQL 以便比较"""
    if not sql:
        return ""
    s = sql.strip().rstrip(";")
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*([,()=<>!])\s*', r'\1', s)
    return s.upper()


def compare_sql(generated: str, expected: str) -> str:
    """
    比较两条 SQL 的匹配程度
    返回: "exact" | "partial" | "none"
    """
    gen_norm = normalize_sql(generated)
    exp_norm = normalize_sql(expected)

    if not gen_norm or not exp_norm:
        return "none"

    if gen_norm == exp_norm:
        return "exact"

    # 提取关键字比较
    gen_tokens = set(re.findall(r'\b(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|HAVING|'
                                r'COUNT|SUM|AVG|MAX|MIN|DISTINCT|LIMIT|'
                                r'LEFT|RIGHT|INNER|OUTER|ON|AND|OR|IN|LIKE|BETWEEN|AS)\b',
                                gen_norm))
    exp_tokens = set(re.findall(r'\b(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|HAVING|'
                                r'COUNT|SUM|AVG|MAX|MIN|DISTINCT|LIMIT|'
                                r'LEFT|RIGHT|INNER|OUTER|ON|AND|OR|IN|LIKE|BETWEEN|AS)\b',
                                exp_norm))

    if gen_tokens and exp_tokens:
        overlap = len(gen_tokens & exp_tokens)
        total = len(gen_tokens | exp_tokens)
        if overlap / total >= 0.5:
            return "partial"

    # 提取 FROM 子句的表名比较
    gen_tables = set(re.findall(r'FROM\s+(\w+)', gen_norm))
    exp_tables = set(re.findall(r'FROM\s+(\w+)', exp_norm))
    if gen_tables and exp_tables and gen_tables == exp_tables:
        return "partial"

    return "none"


def run_structured_eval(dataset_name: str, limit: int = 0) -> EvalReport:
    """评测结构化 NL2SQL 数据集"""
    from backend.training.dataset_loader import DatasetLoader
    from backend.llm.client import create_llm_client
    from backend.nlp.intent_classifier import IntentClassifier
    from backend.nlp.schema_mapper import SchemaMapper
    from backend.nlp.nl2sql_generator import NL2SQLGenerator
    from backend.nlp.sql_validator import SQLValidator
    from backend.analyzer.query_executor import QueryExecutor
    from backend.data_sources.factory import DataSourceFactory
    from backend.data_sources.base import BaseAdapter
    from backend.shared.schemas import DataSourceConfig, DataSourceType

    loader = DatasetLoader(base_dir=str(PROJECT_ROOT))
    ds = loader.load_structured(dataset_name)
    if not ds:
        print(f"[ERROR] 无法加载数据集: {dataset_name}")
        return EvalReport(dataset_name=dataset_name)

    tasks = ds.tasks[:limit] if limit > 0 else ds.tasks
    report = EvalReport(dataset_name=dataset_name, total=len(tasks))

    # 初始化服务
    print(f"评测 {dataset_name} ({len(tasks)} 条任务)...")
    try:
        llm = create_llm_client()
        intent_clf = IntentClassifier(llm)
        schema_mapper = SchemaMapper()
        nl2sql = NL2SQLGenerator(llm)
        sql_validator = SQLValidator()
        executor = QueryExecutor()
    except Exception as e:
        print(f"[ERROR] 初始化服务失败: {e}")
        return report

    # 注册 Excel 数据源
    excel_path = ds.data_paths[0] if ds.data_paths else ""
    source_id = f"eval_{dataset_name}"
    adapter = None

    if excel_path and os.path.exists(excel_path):
        try:
            config = DataSourceConfig(
                source_type=DataSourceType.EXCEL,
                source_id=source_id,
                display_name=dataset_name,
                connection_params={"file_path": excel_path},
            )
            adapter = DataSourceFactory.create(config)
        except Exception as e:
            print(f"[WARN] 无法连接数据源: {e}")

    if not adapter:
        print(f"[ERROR] 数据源不可用: {excel_path}")
        return report

    schemas = adapter.get_schemas()

    for i, task in enumerate(tasks):
        result = EvalResult(
            task_id=task.id,
            question=task.problem,
            expected_sql=task.expected_sql,
        )

        t0 = time.time()
        try:
            question = task.problem
            intent = intent_clf.classify(question, schemas)
            mapped = schema_mapper.map_question(question, schemas)
            sql = nl2sql.generate(question, mapped, intent=intent, schemas=schemas)
            result.generated_sql = sql

            valid, msg = sql_validator.validate(sql, schemas)
            if not valid:
                result.error = f"SQL校验失败: {msg}"
                report.results.append(result)
                continue

            query_result = adapter.execute(sql)
            result.elapsed_ms = (time.time() - t0) * 1000

            if query_result.success:
                result.success = True

            # SQL 比较
            match_level = compare_sql(sql, task.expected_sql)
            if match_level == "exact":
                result.sql_match = True
                report.sql_exact_match += 1
                report.sql_partial_match += 1
            elif match_level == "partial":
                report.sql_partial_match += 1

        except Exception as e:
            result.error = str(e)
            result.elapsed_ms = (time.time() - t0) * 1000

        if result.success:
            report.success += 1

        report.results.append(result)

        # 进度
        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            print(f"  [{i+1}/{len(tasks)}] "
                  f"成功:{report.success} "
                  f"SQL准确:{report.sql_partial_match}")

    if adapter:
        adapter.close()

    report.avg_elapsed_ms = (
        sum(r.elapsed_ms for r in report.results) / len(report.results)
        if report.results else 0
    )

    return report


def run_unstructured_eval(dataset_name: str, limit: int = 0) -> EvalReport:
    """评测非结构化 RAG 数据集（向量召回 + LLM 作答）"""
    from backend.training.dataset_loader import DatasetLoader
    from backend.llm.client import create_llm_client
    from backend.rag.pipeline import RAGPipeline, load_or_build_retriever

    loader = DatasetLoader(base_dir=str(PROJECT_ROOT))
    ds = loader.load_unstructured(dataset_name)
    if not ds:
        print(f"[ERROR] 无法加载数据集: {dataset_name}")
        return EvalReport(dataset_name=dataset_name)

    tasks = ds.tasks[:limit] if limit > 0 else ds.tasks
    report = EvalReport(dataset_name=dataset_name, total=len(tasks))

    print(f"评测 {dataset_name} ({len(tasks)} 条任务)...")
    try:
        llm = create_llm_client()
        index_dir = str(PROJECT_ROOT / "data" / "rag_index" / dataset_name)
        print(f"  构建/加载向量索引: {index_dir}")
        store = load_or_build_retriever(ds.data_paths, index_dir=index_dir)
        rag = RAGPipeline(store=store, llm=llm, top_k=3)
    except Exception as e:
        print(f"[ERROR] 初始化 RAG 失败: {e}")
        return report

    for i, task in enumerate(tasks):
        result = EvalResult(
            task_id=task.id,
            question=task.question,
            expected_answer=task.expected_answer,
        )

        t0 = time.time()
        try:
            response = rag.answer(task.question)
            result.answer_text = response
            result.elapsed_ms = (time.time() - t0) * 1000

            if response and len(response) > 0:
                result.success = True
                report.success += 1

        except Exception as e:
            result.error = str(e)
            result.elapsed_ms = (time.time() - t0) * 1000

        report.results.append(result)

        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            print(f"  [{i+1}/{len(tasks)}] 成功:{report.success}")

    report.avg_elapsed_ms = (
        sum(r.elapsed_ms for r in report.results) / len(report.results)
        if report.results else 0
    )

    return report


def print_report(report: EvalReport, verbose: bool = False):
    """打印评测报告"""
    print()
    print("=" * 60)
    print(f"  评测报告: {report.dataset_name}")
    print("=" * 60)
    print(f"  任务总数:      {report.total}")
    print(f"  执行成功:      {report.success} ({report.success_rate:.1%})")
    print(f"  SQL完全匹配:   {report.sql_exact_match}")
    print(f"  SQL部分匹配:   {report.sql_partial_match} ({report.sql_accuracy:.1%})")
    print(f"  平均耗时:      {report.avg_elapsed_ms:.0f}ms")
    print("-" * 60)

    if verbose and report.results:
        print()
        print("  详细结果:")
        for r in report.results:
            status = "✅" if r.success else "❌"
            sql_info = ""
            if r.generated_sql:
                sql_info = f" | SQL: {r.generated_sql[:60]}..."
            print(f"  {status} [{r.task_id}] {r.question[:50]}...{sql_info}")
            if r.error:
                print(f"     错误: {r.error}")

    print()


def save_reports(reports: List[EvalReport], output_path: str):
    """保存评测报告为 JSON"""
    data = []
    for report in reports:
        data.append({
            "dataset": report.dataset_name,
            "total": report.total,
            "success": report.success,
            "success_rate": report.success_rate,
            "sql_exact_match": report.sql_exact_match,
            "sql_partial_match": report.sql_partial_match,
            "sql_accuracy": report.sql_accuracy,
            "avg_elapsed_ms": report.avg_elapsed_ms,
            "results": [
                {
                    "task_id": str(r.task_id),
                    "question": r.question[:100],
                    "success": r.success,
                    "generated_sql": r.generated_sql,
                    "expected_sql": r.expected_sql,
                    "sql_match": r.sql_match,
                    "elapsed_ms": r.elapsed_ms,
                    "error": r.error,
                }
                for r in report.results
            ],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"评测报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="训练数据集评测")
    parser.add_argument("--dataset", "-d", help="指定数据集名称")
    parser.add_argument("--all", "-a", action="store_true", help="评测全部数据集")
    parser.add_argument("--structured-only", action="store_true", help="仅结构化")
    parser.add_argument("--unstructured-only", action="store_true", help="仅非结构化")
    parser.add_argument("--limit", "-n", type=int, default=0, help="限制评测条数")
    parser.add_argument("--output", "-o", help="输出 JSON 报告路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    from backend.training.dataset_loader import DatasetLoader
    loader = DatasetLoader(base_dir=str(PROJECT_ROOT))

    reports = []

    if args.all:
        # 全部评测
        for name in loader.STRUCTURED_DATASETS:
            report = run_structured_eval(name, args.limit)
            print_report(report, args.verbose)
            reports.append(report)

        for name in loader.UNSTRUCTURED_DATASETS:
            report = run_unstructured_eval(name, args.limit)
            print_report(report, args.verbose)
            reports.append(report)

    elif args.dataset:
        # 指定数据集
        if args.dataset in loader.STRUCTURED_DATASETS:
            report = run_structured_eval(args.dataset, args.limit)
        elif args.dataset in loader.UNSTRUCTURED_DATASETS:
            report = run_unstructured_eval(args.dataset, args.limit)
        else:
            names = (list(loader.STRUCTURED_DATASETS.keys()) +
                     list(loader.UNSTRUCTURED_DATASETS.keys()))
            print(f"未知数据集: {args.dataset}")
            print(f"可用数据集: {', '.join(names)}")
            return
        print_report(report, args.verbose)
        reports.append(report)

    elif args.structured_only:
        for name in loader.STRUCTURED_DATASETS:
            report = run_structured_eval(name, args.limit)
            print_report(report, args.verbose)
            reports.append(report)

    elif args.unstructured_only:
        for name in loader.UNSTRUCTURED_DATASETS:
            report = run_unstructured_eval(name, args.limit)
            print_report(report, args.verbose)
            reports.append(report)

    else:
        loader.get_summary()
        print()
        print("请指定 --dataset <名称> 或 --all")
        print("示例:")
        print("  python scripts/run_eval.py --dataset 金融 --limit 5")
        print("  python scripts/run_eval.py --all --limit 10")
        return

    # 总览
    if len(reports) > 1:
        print()
        print("=" * 60)
        print("  总览")
        print("=" * 60)
        total_tasks = sum(r.total for r in reports)
        total_success = sum(r.success for r in reports)
        total_sql = sum(r.sql_partial_match for r in reports)
        print(f"  数据集: {len(reports)} 个")
        print(f"  总任务: {total_tasks}")
        print(f"  总成功: {total_success} ({total_success/total_tasks:.1%})" if total_tasks else "")
        print(f"  SQL匹配: {total_sql} ({total_sql/total_tasks:.1%})" if total_tasks else "")

    # 保存
    if args.output:
        save_reports(reports, args.output)


if __name__ == "__main__":
    main()
