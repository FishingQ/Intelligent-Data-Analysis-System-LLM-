"""
关联查询器 (M4-3 / Phase 3)
自动发现表间 JOIN 关系，支持三种发现策略:
- 外键约束 (最可靠，confidence=1.0)
- 字段名匹配 (如 dept_id ↔ departments.id, confidence=0.7)
- 值域重叠度 (两个字段共享大量相同值, confidence=0.5)
"""
import logging
from typing import List, Dict, Optional
from collections import Counter

from backend.shared.schemas import TableSchema, ColumnInfo

logger = logging.getLogger(__name__)


class JoinAnalyzer:
    """
    关联查询器

    用法:
        analyzer = JoinAnalyzer()
        joins = analyzer.discover_joins(schemas)      # 自动发现
        joins = analyzer.suggest_joins(schemas)         # 推荐给Prompt
    """

    def discover_joins(
        self,
        schemas: List[TableSchema],
        min_confidence: float = 0.5,
    ) -> List[dict]:
        """
        自动发现表间所有可能的 JOIN 关系

        Args:
            schemas: 多个表的 Schema 列表
            min_confidence: 最低置信度阈值

        Returns:
            [{"left": "t1", "right": "t2", "left_col": "col1", "right_col": "col2",
              "type": "fk"|"name_match"|"value_overlap", "confidence": 0.0-1.0}, ...]
        """
        if len(schemas) < 2:
            return []

        joins = []

        for i in range(len(schemas)):
            for j in range(i + 1, len(schemas)):
                t1 = schemas[i]
                t2 = schemas[j]
                found = self._find_joins_between(t1, t2)
                joins.extend(found)

        # 按置信度降序
        joins.sort(key=lambda j: j["confidence"], reverse=True)

        # 过滤低置信度
        joins = [j for j in joins if j["confidence"] >= min_confidence]

        logger.info(
            f"关联发现 | {len(schemas)}个表 → {len(joins)}个关联关系"
            f"（≥{min_confidence:.0%}置信度）"
        )

        return joins

    def suggest_joins(
        self,
        schemas: List[TableSchema],
        top_k: int = 10,
    ) -> List[dict]:
        """
        推荐最可靠的 JOIN 关系（供 Prompt 使用）

        Returns:
            简化的建议列表，适合插入到 LLM Prompt 中
        """
        joins = self.discover_joins(schemas, min_confidence=0.5)
        seen = set()
        unique = []

        for j in joins:
            key = (j["left"], j["right"])
            if key not in seen:
                seen.add(key)
                unique.append(j)
                if len(unique) >= top_k:
                    break

        logger.info(f"JOIN建议: {len(unique)}个 (去重后，来自{len(joins)}个原始发现)")
        return unique

    # ---- 内部方法 ----

    def _find_joins_between(
        self, t1: TableSchema, t2: TableSchema
    ) -> List[dict]:
        """发现两个表之间的所有 JOIN 关系"""
        joins = []

        for c1 in t1.columns:
            for c2 in t2.columns:
                # 策略1: 外键约束
                fk_join = self._check_fk(t1, c1, t2, c2)
                if fk_join:
                    joins.append(fk_join)
                    continue

                # 策略2: 字段名匹配
                name_join = self._check_name_match(t1, c1, t2, c2)
                if name_join:
                    joins.append(name_join)
                    continue

                # 策略3: ID 命名规律 (xxx_id → table_name 或用_id结尾)
                id_join = self._check_id_pattern(t1, c1, t2, c2)
                if id_join:
                    joins.append(id_join)

        return joins

    @staticmethod
    def _check_fk(
        t1: TableSchema, c1: ColumnInfo,
        t2: TableSchema, c2: ColumnInfo,
    ) -> Optional[dict]:
        """检查外键约束"""
        # c1 是外键，引用了 t2
        if c1.is_foreign_key and c1.referenced_table == t2.table_name:
            return {
                "left": t1.table_name, "right": t2.table_name,
                "left_col": c1.name, "right_col": c2.name if c2.is_primary_key else c1.name,
                "type": "fk", "confidence": 1.0,
            }

        # c2 是外键，引用了 t1
        if c2.is_foreign_key and c2.referenced_table == t1.table_name:
            return {
                "left": t2.table_name, "right": t1.table_name,
                "left_col": c2.name, "right_col": c1.name if c1.is_primary_key else c2.name,
                "type": "fk", "confidence": 1.0,
            }

        return None

    @staticmethod
    def _check_name_match(
        t1: TableSchema, c1: ColumnInfo,
        t2: TableSchema, c2: ColumnInfo,
    ) -> Optional[dict]:
        """字段名完全一致 → 可能关联"""
        if c1.name.lower() == c2.name.lower():
            # 同名且同类型 → 高置信度
            same_type = c1.data_type.upper() == c2.data_type.upper()
            confidence = 0.85 if same_type else 0.65

            return {
                "left": t1.table_name, "right": t2.table_name,
                "left_col": c1.name, "right_col": c2.name,
                "type": "name_match",
                "confidence": confidence,
            }

        return None

    @staticmethod
    def _check_id_pattern(
        t1: TableSchema, c1: ColumnInfo,
        t2: TableSchema, c2: ColumnInfo,
    ) -> Optional[dict]:
        """检查 ID 命名规律: xxx_id → xxx 表"""
        # c1 名称形如 "user_id"，t2 名称是 "users" 或 "user"
        if c1.name.lower().endswith("_id"):
            base = c1.name.lower()[:-3]  # 去掉 _id
            t2_name = t2.table_name.lower()

            if base == t2_name or base + "s" == t2_name or t2_name + "s" == base:
                return {
                    "left": t1.table_name, "right": t2.table_name,
                    "left_col": c1.name,
                    "right_col": next(
                        (c.name for c in t2.columns if c.is_primary_key),
                        "id"
                    ),
                    "type": "id_pattern",
                    "confidence": 0.75,
                }

        # c2 名称可能是外键
        if c2.name.lower().endswith("_id"):
            base = c2.name.lower()[:-3]
            t1_name = t1.table_name.lower()

            if base == t1_name or base + "s" == t1_name or t1_name + "s" == base:
                return {
                    "left": t2.table_name, "right": t1.table_name,
                    "left_col": c2.name,
                    "right_col": next(
                        (c.name for c in t1.columns if c.is_primary_key),
                        "id"
                    ),
                    "type": "id_pattern",
                    "confidence": 0.75,
                }

        return None

    @staticmethod
    def format_for_prompt(joins: List[dict]) -> str:
        """
        将 JOIN 发现结果格式化为 Prompt 可用的文本

        Returns:
            "表A.customer_id ↔ 表B.id (外键约束, 置信度:100%)"
        """
        if not joins:
            return "（未自动发现表关联关系，请根据字段名判断）"

        lines = ["已知表关联关系:"]
        for j in joins:
            type_cn = {"fk": "外键约束", "name_match": "字段名匹配",
                       "id_pattern": "ID命名规律", "value_overlap": "值域重叠"}
            t = type_cn.get(j["type"], j["type"])
            lines.append(
                f"  - {j['left']}.{j['left_col']} ↔ "
                f"{j['right']}.{j['right_col']} "
                f"({t}, 置信度:{j['confidence']:.0%})"
            )

        return "\n".join(lines)
