"""
Schema 映射器
将数据库表结构转化为 LLM 可理解的格式化文本，用于 Prompt 拼接
"""
from typing import List, Optional
import logging

from backend.shared.schemas import TableSchema, ColumnInfo

logger = logging.getLogger(__name__)


class SchemaMapper:
    """
    Schema 映射器

    核心职责:
    1. 将 TableSchema 格式化为 Prompt 友好的文本
    2. 自动发现表间可能的关联关系
    3. 生成字段描述、类型标注等辅助信息

    用法:
        mapper = SchemaMapper()
        text = mapper.format_for_prompt(schemas)
        joins = mapper.suggest_joins(schemas)
    """

    def format_for_prompt(
        self,
        schemas: List[TableSchema],
        max_sample_values: int = 3,
    ) -> str:
        """
        将多个表的 Schema 格式化为 LLM Prompt 可用的文本块

        Args:
            schemas: 表结构列表
            max_sample_values: 每个字段最多展示的样本值数

        Returns:
            格式化后的文本

        示例输出:
            ## 表: sales (57行)
            字段:
              - id (INTEGER) [主键]
              - dept_name (TEXT) 示例: 销售部, 技术部
              - amount (REAL) 示例: 15000.0, 23000.0
              - sale_date (DATE) 示例: 2024-01-15
        """
        if not schemas:
            return "（无可用数据表）"

        parts = []
        for table in schemas:
            lines = [f"## 表: {table.table_name}"]
            if table.description:
                lines.append(f"  描述: {table.description}")
            lines.append(f"  行数: {table.row_count}")
            lines.append("  字段:")

            for col in table.columns:
                col_line = self._format_column(col, max_sample_values)
                lines.append(f"    - {col_line}")

            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def format_compact(self, schemas: List[TableSchema]) -> str:
        """
        紧凑格式 —— 适合 token 敏感的 LLM 调用

        示例: "sales(id,dept_name,amount,sale_date) | users(id,name,role)"
        """
        def _table_str(t: TableSchema) -> str:
            cols = ", ".join(c.name for c in t.columns)
            return f"{t.table_name}({cols}) [{t.row_count}行]"

        return " | ".join(_table_str(t) for t in schemas)

    def suggest_joins(self, schemas: List[TableSchema]) -> List[dict]:
        """
        自动发现表间可能的关联关系

        策略:
        1. 外键约束 (confidence=1.0)
        2. 字段名相同 (confidence=0.7)
        3. 命名规律: xxx_id 匹配表名 (confidence=0.8)

        Returns:
            [{"left":"orders","right":"users","on":"user_id","type":"fk","confidence":1.0}, ...]
        """
        joins = []
        for i, t1 in enumerate(schemas):
            for j, t2 in enumerate(schemas):
                if i >= j:
                    continue
                for c1 in t1.columns:
                    # 策略1: 外键约束
                    if c1.is_foreign_key and c1.referenced_table == t2.table_name:
                        joins.append({
                            "left": t1.table_name,
                            "right": t2.table_name,
                            "on": c1.name,
                            "type": "fk",
                            "confidence": 1.0,
                        })
                        continue

                    for c2 in t2.columns:
                        # 策略2: 字段名相同且类型相似
                        if c1.name.lower() == c2.name.lower():
                            if self._types_compatible(c1, c2):
                                joins.append({
                                    "left": t1.table_name,
                                    "right": t2.table_name,
                                    "on": c1.name,
                                    "type": "name_match",
                                    "confidence": 0.7,
                                })

                        # 策略3: 命名约定
                        if c1.name.lower() == f"{t2.table_name.lower()}_id":
                            joins.append({
                                "left": t1.table_name,
                                "right": t2.table_name,
                                "on": c1.name,
                                "type": "naming_convention",
                                "confidence": 0.85,
                            })

        # 去重并取最高置信度
        seen = set()
        unique_joins = []
        for j in sorted(joins, key=lambda x: x["confidence"], reverse=True):
            key = tuple(sorted([j["left"], j["right"], j["on"]]))
            if key not in seen:
                seen.add(key)
                unique_joins.append(j)

        return unique_joins

    def summarize_for_display(self, schema: TableSchema) -> str:
        """为前端数据源面板生成表摘要"""
        num_cols = [c for c in schema.columns if self._is_numeric(c.data_type)]
        date_cols = [c for c in schema.columns if self._is_date(c.data_type)]
        cat_cols = [c for c in schema.columns
                    if c.name not in [nc.name for nc in num_cols]
                    and c.name not in [dc.name for dc in date_cols]]

        parts = [f"{schema.table_name}: {schema.row_count}行"]
        if num_cols:
            parts.append(f"{len(num_cols)}个数值列")
        if date_cols:
            parts.append(f"{len(date_cols)}个日期列")
        if cat_cols:
            parts.append(f"{len(cat_cols)}个分类列")
        return ", ".join(parts)

    # ---- 内部方法 ----

    def _format_column(self, col: ColumnInfo, max_samples: int) -> str:
        """格式化单个字段信息"""
        parts = [col.name, f"({col.data_type})"]

        if col.is_primary_key:
            parts.append("[主键]")
        if col.is_foreign_key and col.referenced_table:
            parts.append(f"[外键→{col.referenced_table}]")
        if not col.nullable:
            parts.append("[必填]")

        # 样本值
        if col.sample_values:
            samples = col.sample_values[:max_samples]
            sample_str = ", ".join(
                str(s) for s in samples if s is not None
            )
            if sample_str:
                parts.append(f"示例: {sample_str}")

        return " ".join(parts)

    @staticmethod
    def _is_numeric(data_type: str) -> bool:
        dt = data_type.upper()
        return any(t in dt for t in [
            "INT", "FLOAT", "DOUBLE", "REAL", "NUMERIC", "DECIMAL", "NUMBER"
        ])

    @staticmethod
    def _is_date(data_type: str) -> bool:
        dt = data_type.upper()
        return any(t in dt for t in ["DATE", "TIME", "DATETIME", "TIMESTAMP"])

    @staticmethod
    def _types_compatible(c1: ColumnInfo, c2: ColumnInfo) -> bool:
        """判断两个字段类型是否兼容 (可用于关联)"""
        # 主键关联
        if c1.is_primary_key or c2.is_primary_key:
            return True
        # 同类型
        if c1.data_type.upper() == c2.data_type.upper():
            return True
        # 整数类型兼容
        if SchemaMapper._is_numeric(c1.data_type) and SchemaMapper._is_numeric(c2.data_type):
            return True
        return False
