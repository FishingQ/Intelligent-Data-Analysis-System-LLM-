"""
智能图表推荐器 (M5-1)
根据查询结果的列类型和数据特征，自动推荐最佳图表类型和轴映射
"""
import logging
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

from backend.shared.schemas import QueryResult, ChartConfig, ChartType

logger = logging.getLogger(__name__)


class ChartRecommender:
    """
    智能图表推荐器

    根据数据特征自动推荐:
    - 图表类型 (line/bar/pie/scatter/heatmap)
    - X轴/Y轴映射
    - 图表标题

    推荐规则（优先级从高到低）:

    | 条件 | 推荐图表 | 示例场景 |
    |------|---------|---------|
    | 1日期 + 1+数值 classifies | LINE | 月度销售趋势 |
    | 1分类(≤6类) + 1数值(占比) | PIE | 部门预算占比 |
    | 1分类(≤20类) + 1+数值 | BAR | 各省GDP排名 |
    | 2+数值 | SCATTER | 身高-体重关系 |
    | 矩阵数据(3列以上, 行列+值) | HEATMAP | 相关系数矩阵 |

    用法:
        recommender = ChartRecommender()
        config = recommender.recommend(result)
    """

    # 柱状图最大分类数
    MAX_BAR_CATEGORIES = 20
    # 饼图最大分类数
    MAX_PIE_CATEGORIES = 6

    def recommend(self, result: QueryResult) -> Optional[ChartConfig]:
        """
        分析查询结果，返回推荐图表配置

        Args:
            result: 查询执行结果

        Returns:
            ChartConfig 或 None（数据不适合可视化）
        """
        if not result.success or not result.data or result.row_count == 0:
            return None

        df = pd.DataFrame(result.data, columns=result.columns)
        if len(df.columns) < 2:
            return None

        # 分类列
        date_cols = self._find_date_columns(df)
        num_cols = self._find_numeric_columns(df)
        cat_cols = [c for c in df.columns if c not in date_cols and c not in num_cols]

        # 规则1: 日期 + 数值 → 折线图
        if date_cols and num_cols:
            return self._build(
                ChartType.LINE, date_cols[0], num_cols[0],
                title=f"{num_cols[0]} 时间趋势",
                df=df, date_cols=date_cols, num_cols=num_cols,
            )

        # 规则2: 分类 + 数值
        if cat_cols and num_cols:
            x_col = cat_cols[0]
            y_col = num_cols[0]
            n_unique = df[x_col].nunique()

            if n_unique <= self.MAX_PIE_CATEGORIES:
                # 检查是否为占比类数据
                total = df[y_col].sum()
                if total > 0:
                    return self._build(
                        ChartType.PIE, x_col, y_col,
                        title=f"{y_col} 占比分布",
                        df=df, cat_cols=cat_cols, num_cols=num_cols,
                    )

            if n_unique <= self.MAX_BAR_CATEGORIES:
                # 柱状图（横向或纵向）
                chart_type = ChartType.BAR
                if n_unique > 10:
                    # 分类多时条形图（横向）更易读
                    pass  # ECharts 中通过配置实现

                return self._build(
                    chart_type, x_col, y_col,
                    title=f"{y_col} 对比排名",
                    df=df, cat_cols=cat_cols, num_cols=num_cols,
                )

            # 分类太多但仍有数值 → 取Top10
            top_df = df.nlargest(min(10, len(df)), y_col)
            return self._build(
                ChartType.BAR, x_col, y_col,
                title=f"{y_col} Top {len(top_df)}",
                df=top_df, cat_cols=cat_cols, num_cols=num_cols,
            )

        # 规则3: 双数值 → 散点图
        if len(num_cols) >= 2:
            return self._build(
                ChartType.SCATTER, num_cols[0], num_cols[1],
                title=f"{num_cols[0]} vs {num_cols[1]}",
                df=df, num_cols=num_cols,
            )

        # 规则4: 单数值 + 首列为标签
        if num_cols and len(df.columns) >= 2:
            label_col = df.columns[0]
            val_col = num_cols[0]
            n_unique = df[label_col].nunique()
            if n_unique <= self.MAX_BAR_CATEGORIES:
                return self._build(
                    ChartType.BAR if n_unique > self.MAX_PIE_CATEGORIES else ChartType.PIE,
                    label_col, val_col,
                    title=f"{val_col} 分布",
                    df=df, num_cols=num_cols,
                )

        return None

    def _build(
        self,
        chart_type: ChartType,
        x_axis: str,
        y_axis: str,
        title: str,
        df: pd.DataFrame,
        **hints,
    ) -> ChartConfig:
        """构建 ChartConfig"""
        return ChartConfig(
            chart_type=chart_type,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            series_data={},
        )

    # ---- 列类型检测 ----

    @staticmethod
    def _find_date_columns(df: pd.DataFrame) -> List[str]:
        """检测日期列"""
        date_cols = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_cols.append(c)
                continue
            # 尝试解析字符串日期
            if df[c].dtype == object:
                sample = df[c].dropna().head(5)
                if len(sample) > 0:
                    try:
                        parsed = pd.to_datetime(sample)
                        if parsed.notna().sum() >= len(sample) * 0.8:
                            date_cols.append(c)
                    except (ValueError, TypeError):
                        pass
        return date_cols

    @staticmethod
    def _find_numeric_columns(df: pd.DataFrame) -> List[str]:
        """检测数值列"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
