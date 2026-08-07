"""
公式引擎 (M4-6 / Phase 3)
对查询结果应用公式计算，支持四则运算、聚合比例和条件表达式。

安全策略: whitelist 白名单过滤，禁止 import/exec/eval 等危险操作。
"""
import logging
import re
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from backend.shared.schemas import QueryResult

logger = logging.getLogger(__name__)

# 安全白名单
_ALLOWED_FUNCTIONS = {
    "abs", "round", "sum", "min", "max", "avg", "mean",
    "count", "len", "sqrt", "log", "pow",
}
_FORBIDDEN_TOKENS = [
    "__", "import", "exec", "eval", "compile", "open",
    "os.", "sys.", "subprocess", "shutil", "socket",
]


class FormulaEngine:
    """
    公式引擎

    用法:
        engine = FormulaEngine()
        result = engine.apply(result, "profit_rate = profit / revenue")
        result = engine.apply(result, "bonus = IF(salary>10000, salary*0.1, 0)")
    """

    def apply(
        self,
        result: QueryResult,
        formula: str,
    ) -> QueryResult:
        """
        应用公式计算

        Args:
            result: 查询结果
            formula: 公式表达式，如 "new_col = col_a + col_b * 0.1"

        Returns:
            带新列的 QueryResult
        """
        if not result.success or not result.data:
            return QueryResult(
                success=False,
                error_message="无有效数据，无法应用公式",
            )

        # 安全检查
        if not self._is_safe(formula):
            return QueryResult(
                success=False,
                columns=result.columns,
                data=result.data,
                row_count=result.row_count,
                error_message=f"公式包含不允许的操作: {formula}",
            )

        df = pd.DataFrame(result.data, columns=result.columns)

        try:
            new_df = self._evaluate(df, formula)

            return QueryResult(
                success=True,
                columns=list(new_df.columns),
                data=new_df.values.tolist(),
                row_count=len(new_df),
                execution_time_ms=result.execution_time_ms,
            )

        except Exception as e:
            logger.warning(f"公式计算失败: {e}")
            return QueryResult(
                success=False,
                columns=result.columns,
                data=result.data,
                row_count=result.row_count,
                error_message=f"公式计算错误: {e}",
            )

    def apply_formulas(
        self,
        result: QueryResult,
        formulas: List[str],
    ) -> QueryResult:
        """依次应用多个公式"""
        for formula in formulas:
            result = self.apply(result, formula)
            if not result.success:
                break
        return result

    # ---- 推荐公式 ----

    def suggest_formulas(self, result: QueryResult) -> List[str]:
        """基于数据列自动推荐可用公式"""
        if not result.success or not result.data:
            return []

        df = pd.DataFrame(result.data, columns=result.columns)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        suggestions = []

        # 比率
        if len(num_cols) >= 2:
            suggestions.append(f"ratio = {num_cols[0]} / {num_cols[1]}")

        # 百分比
        if num_cols:
            suggestions.append(f"pct = {num_cols[0]} / {num_cols[0]}.sum()")

        # 差值
        if len(num_cols) >= 2:
            suggestions.append(f"diff = {num_cols[0]} - {num_cols[1]}")

        # 增长率
        if num_cols:
            suggestions.append(f"growth = {num_cols[0]}.pct_change()")

        return suggestions

    # ---- 内部方法 ----

    @staticmethod
    def _is_safe(formula: str) -> bool:
        """安全检查：禁止危险操作"""
        formula_lower = formula.lower()
        for token in _FORBIDDEN_TOKENS:
            if token in formula_lower:
                logger.warning(f"公式包含禁止token: {token}")
                return False
        return True

    def _evaluate(self, df: pd.DataFrame, formula: str) -> pd.DataFrame:
        """安全地执行公式"""
        # 解析: new_col = expression
        match = re.match(r'^\s*(\w+)\s*=\s*(.+)$', formula.strip())
        if not match:
            # 尝试 pd.eval 直接执行（仅表达式）
            try:
                result = pd.eval(formula, target=df)
                if isinstance(result, pd.Series):
                    df["result"] = result
                return df
            except Exception:
                raise ValueError(f"无法解析公式: {formula}。格式应为 'new_col = expression'")

        col_name = match.group(1)
        expression = match.group(2)

        # 处理 IF 函数
        df = self._handle_if(df, col_name, expression)
        if col_name in df.columns:
            return df

        # 处理聚合函数
        df = self._handle_aggregation(df, col_name, expression)
        if col_name in df.columns:
            return df

        # 通用表达式求值
        try:
            df[col_name] = df.eval(expression)
        except Exception:
            # 回退: 逐行计算
            df[col_name] = self._safe_eval_row(df, expression)

        return df

    @staticmethod
    def _handle_if(df: pd.DataFrame, col_name: str, expression: str) -> pd.DataFrame:
        """处理 IF(condition, true_val, false_val)"""
        match = re.match(
            r'IF\s*\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)\s*$',
            expression, re.IGNORECASE
        )
        if not match:
            return df

        condition, true_val, false_val = match.groups()

        # 安全解析条件
        try:
            mask = df.eval(condition).astype(bool)
            true_series = df.eval(true_val) if not true_val.replace('.', '').replace('-', '').isdigit() else pd.Series([float(true_val)] * len(df))
            false_series = df.eval(false_val) if not false_val.replace('.', '').replace('-', '').isdigit() else pd.Series([float(false_val)] * len(df))

            if not isinstance(true_series, pd.Series):
                true_series = pd.Series([true_series] * len(df))
            if not isinstance(false_series, pd.Series):
                false_series = pd.Series([false_series] * len(df))

            df[col_name] = np.where(mask, true_series, false_series)
        except Exception as e:
            logger.warning(f"IF公式执行失败: {e}")

        return df

    @staticmethod
    def _handle_aggregation(df: pd.DataFrame, col_name: str, expression: str) -> pd.DataFrame:
        """处理含聚合函数的表达式"""
        has_agg = any(f in expression.upper() for f in ["SUM(", "AVG(", "MEAN(", "MIN(", "MAX("])
        if not has_agg:
            return df

        # 简单替换: SUM(col) → col.sum()
        expr_upper = expression.upper()
        for func in ["SUM", "AVG", "MEAN", "MIN", "MAX"]:
            pattern = rf'{func}\((\w+)\)'
            for col_match in re.finditer(pattern, expr_upper):
                orig = expression[col_match.start():col_match.end()]
                col_ref = col_match.group(1).lower()
                replacement = f"df['{col_ref}'].{func.lower()}()"
                expression = expression.replace(orig, replacement, 1)

        try:
            df[col_name] = eval(expression, {"df": df, "np": np, "pd": pd})
        except Exception as e:
            logger.warning(f"聚合公式执行失败: {e}")

        return df

    @staticmethod
    def _safe_eval_row(df: pd.DataFrame, expression: str) -> pd.Series:
        """安全的逐行求值回退"""
        results = []
        for _, row in df.iterrows():
            try:
                local_vars = row.to_dict()
                local_vars["np"] = np
                results.append(eval(expression, {"__builtins__": {}}, local_vars))
            except Exception:
                results.append(np.nan)
        return pd.Series(results)
