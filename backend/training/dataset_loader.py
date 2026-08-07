"""
训练数据集加载器
支持加载结构化 NL2SQL 任务和非结构化 RAG 任务
"""
import json
import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class StructuredTask:
    """结构化 NL2SQL 训练任务"""
    id: int
    problem: str
    dataset: str               # "金融" | "医疗" | "通信"
    excel_path: str            # Excel 文件路径
    expected_sql: str = ""     # 样例答案 SQL


@dataclass
class UnstructuredTask:
    """非结构化 RAG 训练任务"""
    id: str
    question: str
    dataset: str               # "multi_step_retrieval" | "table_query" | "table_domain_ops"
    data_path: str             # Markdown 数据文件路径
    context_id: str = ""       # 关联的数据段落 ID
    expected_answer: str = ""


@dataclass
class TrainingDataset:
    """训练数据集"""
    name: str
    dataset_type: str          # "structured" | "unstructured"
    data_paths: List[str] = field(default_factory=list)    # 数据文件路径
    tasks: List[Any] = field(default_factory=list)          # StructuredTask | UnstructuredTask
    task_count: int = 0


class DatasetLoader:
    """
    训练数据集加载器

    用法:
        loader = DatasetLoader()
        datasets = loader.load_all()

        # 按名称获取
        finance = loader.load_structured("金融")
        # 获取所有非结构化任务
        unstructured = loader.load_unstructured("multi_step_retrieval")
    """

    # ================================================================
    # 数据集注册表
    # ================================================================

    STRUCTURED_DATASETS = {
        "金融": {
            "excel": "结构化数据/金融/financial_asset_management.xlsx",
            "tasks": "结构化数据/金融/merged_problems1_task.json",
            "answers": "结构化数据/金融/样例答案.json",
        },
        "医疗": {
            "excel": "结构化数据/医疗/Healthcare_Analytics_Competition.xlsx",
            "tasks": "结构化数据/医疗/merged_problems2_task.json",
            "answers": "结构化数据/医疗/样例答案.json",
        },
        "通信": {
            "excel": "结构化数据/通信/telecom_operations_db.xlsx",
            "tasks": "结构化数据/通信/merged_problems3_task.json",
            "answers": "结构化数据/通信/样例答案.json",
        },
    }

    UNSTRUCTURED_DATASETS = {
        "multi_step_retrieval": {
            "data": "非结构化数据/Multi-step_Retrieval-数据.md",
            "tasks": "非结构化数据/Multi-step_Retrieval_task.json",
        },
        "table_query": {
            "data": "非结构化数据/Table_Query-数据.md",
            "tasks": "非结构化数据/Table_Query_task.json",
        },
        "table_domain_ops": {
            "data": "非结构化数据/Table_Domain-specific_Operations-数据.md",
            "tasks": "非结构化数据/Table_Domain-specific_Operations_task.json",
        },
    }

    # 公共答案文件（非结构化共用）
    UNSTRUCTURED_ANSWERS = "非结构化数据/样例答案.json"

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT

    # ---- 加载全部 ----

    def load_all(self) -> Dict[str, TrainingDataset]:
        """加载全部训练数据集"""
        datasets = {}

        for name in self.STRUCTURED_DATASETS:
            ds = self.load_structured(name)
            if ds:
                datasets[f"structured/{name}"] = ds

        for name in self.UNSTRUCTURED_DATASETS:
            ds = self.load_unstructured(name)
            if ds:
                datasets[f"unstructured/{name}"] = ds

        return datasets

    # ---- 结构化数据 ----

    def load_structured(self, name: str) -> Optional[TrainingDataset]:
        """加载结构化 NL2SQL 训练集"""
        cfg = self.STRUCTURED_DATASETS.get(name)
        if not cfg:
            logger.error(f"未知结构化数据集: {name}")
            return None

        excel_path = self.base_dir / cfg["excel"]
        task_path = self.base_dir / cfg["tasks"]
        answer_path = self.base_dir / cfg.get("answers", "")

        if not task_path.exists():
            logger.error(f"任务文件不存在: {task_path}")
            return None

        # 加载任务
        with open(task_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)

        # 加载样例答案
        answers = {}
        if answer_path.exists():
            with open(answer_path, "r", encoding="utf-8") as f:
                raw_answers = json.load(f)
            for a in raw_answers:
                answers[a.get("id")] = a.get("sql", a.get("answer", ""))

        # 构建任务列表
        tasks = []
        for t in raw_tasks:
            task = StructuredTask(
                id=t["id"],
                problem=t.get("problem", t.get("question", "")),
                dataset=name,
                excel_path=str(excel_path) if excel_path.exists() else "",
                expected_sql=answers.get(t["id"], ""),
            )
            tasks.append(task)

        logger.info(f"加载结构化数据集 [{name}]: {len(tasks)} 条任务, "
                     f"{len(answers)} 条样例答案, "
                     f"Excel: {'已找到' if excel_path.exists() else '未找到'}")

        return TrainingDataset(
            name=name,
            dataset_type="structured",
            data_paths=[str(excel_path)] if excel_path.exists() else [],
            tasks=tasks,
            task_count=len(tasks),
        )

    # ---- 非结构化数据 ----

    def load_unstructured(self, name: str) -> Optional[TrainingDataset]:
        """加载非结构化 RAG 训练集"""
        cfg = self.UNSTRUCTURED_DATASETS.get(name)
        if not cfg:
            logger.error(f"未知非结构化数据集: {name}")
            return None

        data_path = self.base_dir / cfg["data"]
        task_path = self.base_dir / cfg["tasks"]
        answer_path = self.base_dir / self.UNSTRUCTURED_ANSWERS

        if not task_path.exists():
            logger.error(f"任务文件不存在: {task_path}")
            return None

        # 加载任务
        with open(task_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)

        # 加载样例答案
        answers = {}
        if answer_path.exists():
            with open(answer_path, "r", encoding="utf-8") as f:
                raw_answers = json.load(f)
            for a in raw_answers:
                answers[a["id"]] = a.get("answer", a.get("sql", ""))

        # 解析 markdown 数据中的段落映射 (格式: === id === ...content...)
        context_map = {}
        if data_path.exists():
            context_map = self._parse_markdown_contexts(str(data_path))

        # 构建任务列表
        tasks = []
        for t in raw_tasks:
            tid = t["id"]
            task = UnstructuredTask(
                id=tid,
                question=t.get("question", t.get("problem", "")),
                dataset=name,
                data_path=str(data_path) if data_path.exists() else "",
                context_id=tid,
                expected_answer=answers.get(tid, ""),
            )
            tasks.append(task)

        logger.info(f"加载非结构化数据集 [{name}]: {len(tasks)} 条任务, "
                     f"{len(answers)} 条样例答案, "
                     f"{len(context_map)} 个数据段落")

        return TrainingDataset(
            name=name,
            dataset_type="unstructured",
            data_paths=[str(data_path)] if data_path.exists() else [],
            tasks=tasks,
            task_count=len(tasks),
        )

    # ---- 按 ID 获取上下文数据 ----

    def get_context_for_task(self, task: UnstructuredTask) -> str:
        """根据任务 ID 获取对应的数据上下文"""
        if not task.data_path or not os.path.exists(task.data_path):
            return ""

        context_map = self._parse_markdown_contexts(task.data_path)
        return context_map.get(task.context_id, "")

    @staticmethod
    def _parse_markdown_contexts(filepath: str) -> Dict[str, str]:
        """
        解析 Markdown 文件中的 === id === 分隔的数据段落

        格式:
            === 05ab28a47b924ae ===
            ```markdown
            ...table data...
            ```
            === next_id ===
            ...
        """
        contexts = {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # 按 === id === 分割
            import re
            blocks = re.split(r'\n=== (.+?) ===\n', content)

            # blocks[0] 是第一个分隔符前的内容（可能为空）
            # 之后是 id1, content1, id2, content2, ...
            for i in range(1, len(blocks), 2):
                if i + 1 <= len(blocks):
                    cid = blocks[i].strip()
                    ccontent = blocks[i + 1].strip()
                    contexts[cid] = ccontent

        except Exception as e:
            logger.warning(f"解析Markdown上下文失败 ({filepath}): {e}")

        return contexts

    # ---- 统计信息 ----

    def get_summary(self) -> str:
        """获取所有数据集的摘要"""
        lines = ["=== 训练数据集总览 ===", ""]

        # 结构化
        lines.append("## 结构化 NL2SQL 数据集")
        for name, cfg in self.STRUCTURED_DATASETS.items():
            task_path = self.base_dir / cfg["tasks"]
            excel_path = self.base_dir / cfg["excel"]
            answer_path = self.base_dir / cfg.get("answers", "")

            task_count = 0
            answer_count = 0
            if task_path.exists():
                with open(task_path, "r", encoding="utf-8") as f:
                    task_count = len(json.load(f))
            if answer_path.exists():
                with open(answer_path, "r", encoding="utf-8") as f:
                    answer_count = len(json.load(f))

            size_mb = excel_path.stat().st_size / 1024 / 1024 if excel_path.exists() else 0
            lines.append(f"  📊 {name}: {task_count}题, {answer_count}样例答案, "
                         f"Excel={size_mb:.1f}MB")

        lines.append("")
        lines.append("## 非结构化 RAG 数据集")
        for name, cfg in self.UNSTRUCTURED_DATASETS.items():
            task_path = self.base_dir / cfg["tasks"]
            data_path = self.base_dir / cfg["data"]

            task_count = 0
            if task_path.exists():
                with open(task_path, "r", encoding="utf-8") as f:
                    task_count = len(json.load(f))

            size_mb = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
            lines.append(f"  📝 {name}: {task_count}题, 数据={size_mb:.1f}MB")

        return "\n".join(lines)
