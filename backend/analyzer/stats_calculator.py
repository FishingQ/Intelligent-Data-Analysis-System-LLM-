"""
统计计算器 (M4-2)
对查询结果进行二次统计分析：描述性统计、相关性、同比环比、分布分析
"""
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from backend.shared.schemas import QueryResult

logger = logging.getLogger(__name__)


class StatsCalculator:
    """
    统计计算器

    对 QueryResult 进行二次统计加工，输出结构化指标。

    用法:
        calc = StatsCalculator()
        summary = calc.describe(result)       # 描述性统计
        corr = calc.correlation(result)        # 相关性矩阵
        yoy = calc.yoy_mom(result, ...)        # 同比/环比
    """

    def describe(self, result: QueryResult) -> Dict[str, Any]:
        """
        描述性统计

        Returns:
            {
              "numeric_columns": {"col_name": {"mean": ..., "std": ..., "min": ..., ...}},
              "categorical_columns": {"col_name": {"unique": ..., "top": ..., "freq": ...}},
              "row_count": int,
              "column_count": int,
            }
        """
        df = self._to_dataframe(result)
        if df is None:
            return {"error": "无有效数据"}

        stats: Dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": {},
            "categorical_columns": {},
        }

        # 数值列统计
        num_df = df.select_dtypes(include=[np.number])
        for col in num_df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            stats["numeric_columns"][col] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": self._to_native(series.min()),
                "q25": round(float(series.quantile(0.25)), 4),
                "median": round(float(series.median()), 4),
                "q75": round(float(series.quantile(0.75)), 4),
                "max": self._to_native(series.max()),
                "sum": round(float(series.sum()), 2),
                "missing": int(df[col].isna().sum()),
            }

        # 分类列统计（≤20个唯一值才统计）
        for col in df.columns:
            if col in num_df.columns:
                continue
            series = df[col].dropna()
            unique = series.nunique()
            if unique <= 20 and unique > 0:
                value_counts = series.value_counts()
                stats["categorical_columns"][col] = {
                    "unique": int(unique),
                    "top": str(value_counts.index[0]) if len(value_counts) > 0 else "",
                    "top_freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "distribution": {
                        str(k): int(v) for k, v in value_counts.head(10).items()
                    },
                }

        return stats

    def correlation(self, result: QueryResult) -> Dict[str, Any]:
        """
        数值列相关性分析

        Returns:
            {
              "correlation_matrix": [[1.0, 0.8, ...], ...],
              "columns": ["col1", "col2", ...],
              "strong_pairs": [{"col1": ..., "col2": ..., "coefficient": ...}, ...]
            }
        """
        df = self._to_dataframe(result)
        if df is None:
            return {"error": "无有效数据"}

        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return {"error": "至少需要2个数值列", "columns": list(num_df.columns)}

        corr_matrix = num_df.corr()

        # 找出强相关对 (|r| > 0.5)
        strong_pairs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.5:
                    strong_pairs.append({
                        "col1": cols[i],
                        "col2": cols[j],
                        "coefficient": round(float(r), 4),
                    })
        strong_pairs.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

        return {
            "correlation_matrix": corr_matrix.values.tolist(),
            "columns": cols,
            "strong_pairs": strong_pairs[:10],
        }

    def yoy_mom(
        self,
        result: QueryResult,
        date_col: str,
        value_col: str,
    ) -> Dict[str, Any]:
        """
        同比(YoY) / 环比(MoM) 计算

        Args:
            result: 查询结果
            date_col: 日期列名
            value_col: 数值列名

        Returns:
            {
              "data": [{"date": ..., "value": ..., "mom": ..., "yoy": ...}, ...],
              "latest_mom": "+5.2%",
              "latest_yoy": "+12.1%",
            }
        """
        df = self._to_dataframe(result)
        if df is None:
            return {"error": "无有效数据"}

        if date_col not in df.columns or value_col not in df.columns:
            return {"error": f"列 '{date_col}' 或 '{value_col}' 不存在"}

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])
        df = df.sort_values(date_col)

        df["mom"] = df[value_col].pct_change(periods=1)
        df["yoy"] = df[value_col].pct_change(periods=12)

        latest_mom = df["mom"].iloc[-1] if len(df) > 1 and not pd.isna(df["mom"].iloc[-1]) else None
        latest_yoy = df["yoy"].iloc[-1] if len(df) > 12 and not pd.isna(df["yoy"].iloc[-1]) else None

        recency = []
        for _, row in df.iterrows():
            rec = {
                "date": str(row[date_col])[:10],
                "value": self._to_native(row[value_col]),
            }
            if not pd.isna(row.get("mom")):
                rec["mom"] = f"{row['mom']:+.2%}"
            if not pd.isna(row.get("yoy")):
                rec["yoy"] = f"{row['yoy']:+.2%}"
            recency.append(rec)

        return {
            "data": recency[-24:],  # 最近24条
            "latest_mom": f"{latest_mom:+.2%}" if latest_mom is not None else "N/A",
            "latest_yoy": f"{latest_yoy:+.2%}" if latest_yoy is not None else "N/A",
        }

    def summary(self, result: QueryResult) -> str:
        """
        生成可读的统计摘要文本

        Returns:
            "共100行×5列，数值列3个(销售额/利润/数量)，分类列1个(部门: 5类)，日期列1个"
        """
        df = self._to_dataframe(result)
        if df is None:
            return "无有效数据"

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols and df[c].nunique() <= 20]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        parts = [f"共{len(df)}行×{len(df.columns)}列"]

        if num_cols:
            parts.append(f"数值列{len(num_cols)}个({', '.join(num_cols[:5])})")
        if cat_cols:
            parts.append(f"分类列{len(cat_cols)}个({', '.join(cat_cols[:5])})")
        if date_cols:
            parts.append(f"日期列{len(date_cols)}个({', '.join(date_cols[:3])})")

        return "，".join(parts)

    # ---- 内部方法 ----

    @staticmethod
    def _to_dataframe(result: QueryResult) -> Optional[pd.DataFrame]:
        if not result or not result.success or not result.data:
            return None
        return pd.DataFrame(result.data, columns=result.columns)

    @staticmethod
    def _to_native(val: Any) -> Any:
        """numpy类型转为Python原生类型"""
        try:
            if hasattr(val, 'item'):
                return val.item()
        except Exception:
            pass
        return val
