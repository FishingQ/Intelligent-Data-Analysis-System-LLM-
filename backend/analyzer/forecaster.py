"""
时序预测器 (M4-4 / Phase 3)
对查询结果的时间序列数据进行趋势预测:
- Prophet (主方案): Facebook Prophet，自动检测趋势+季节+变点
- 移动平均 (回退方案): 简单移动平均 + 指数平滑，零依赖

自动检测日期列和数值列，生成预测数据和可视化图表。
"""
import logging
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.shared.schemas import (
    QueryResult, ForecastPoint, ForecastResult, ChartConfig, ChartType,
)

logger = logging.getLogger(__name__)


class Forecaster:
    """
    时序预测器

    双引擎:
    1. Prophet → 自动趋势+季节分解（需安装 prophet）
    2. 移动平均 → 零依赖回退（简单移动平均 + 线性趋势外推）

    用法:
        forecaster = Forecaster()
        result = forecaster.forecast(query_result)
        # 自动检测日期列和数值列
        result = forecaster.forecast(query_result, date_col="日期", value_col="销售额", periods=12)
    """

    DEFAULT_PERIODS = 12          # 默认预测期数
    MIN_HISTORICAL_POINTS = 6     # 最少历史数据点

    def forecast(
        self,
        result: QueryResult,
        date_col: str = "",
        value_col: str = "",
        periods: int = 0,
    ) -> ForecastResult:
        """
        执行时序预测

        Args:
            result: 查询执行结果
            date_col: 日期列名（留空自动检测）
            value_col: 数值列名（留空自动检测）
            periods: 预测期数（留空使用默认值）

        Returns:
            ForecastResult
        """
        if not result.success or not result.data or result.row_count == 0:
            return ForecastResult(trend_direction="flat")

        df = pd.DataFrame(result.data, columns=result.columns)

        # 自动检测列
        if not date_col:
            date_col = self._find_date_column(df)
        if not value_col:
            value_col = self._find_value_column(df, date_col)

        if not date_col or not value_col:
            return ForecastResult(
                trend_direction="flat",
                chart=self._empty_chart("缺少日期列或数值列，无法预测"),
            )

        if periods <= 0:
            periods = min(self.DEFAULT_PERIODS, len(df) // 2) or self.DEFAULT_PERIODS

        # 清洗数据
        df_clean = self._prepare_data(df, date_col, value_col)
        if len(df_clean) < self.MIN_HISTORICAL_POINTS:
            return ForecastResult(
                trend_direction="flat",
                chart=self._empty_chart(f"历史数据不足（需要≥{self.MIN_HISTORICAL_POINTS}个点，当前{len(df_clean)}个）"),
            )

        logger.info(
            f"时序预测 | 历史: {len(df_clean)}点 | "
            f"预测: {periods}期 | 日期: {date_col} | 数值: {value_col}"
        )

        # 尝试 Prophet，失败回退移动平均
        try:
            return self._forecast_prophet(df_clean, date_col, value_col, periods)
        except Exception as e:
            logger.warning(f"Prophet 预测失败 ({e})，使用移动平均回退方案")
            return self._forecast_moving_average(df_clean, date_col, value_col, periods)

    # ---- Prophet 方案 ----

    def _forecast_prophet(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        periods: int,
    ) -> ForecastResult:
        """使用 Facebook Prophet 进行预测"""
        from prophet import Prophet

        # Prophet 要求列名为 ds / y
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df[date_col]),
            "y": df[value_col].astype(float),
        }).dropna()

        model = Prophet(
            growth="linear",
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.fit(prophet_df)

        # 推断时间频率
        freq = self._infer_freq(prophet_df["ds"])
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        # 构建结果
        historical = []
        forecast_points = []

        for _, row in forecast.iterrows():
            date_str = row["ds"].strftime("%Y-%m-%d")
            is_hist = row["ds"] <= prophet_df["ds"].max()

            point = ForecastPoint(
                date=date_str,
                value=round(float(row["yhat"]), 2),
                lower_bound=round(float(row["yhat_lower"]), 2),
                upper_bound=round(float(row["yhat_upper"]), 2),
                is_historical=bool(is_hist),
            )

            if is_hist:
                historical.append(point)
            else:
                forecast_points.append(point)

        # 趋势方向
        trend_direction = self._calc_trend(forecast_points)

        # 构建图表
        chart = self._build_forecast_chart(
            historical[-50:], forecast_points, value_col
        )

        return ForecastResult(
            historical=historical[-50:],
            forecast=forecast_points,
            trend_direction=trend_direction,
            trend_strength=self._calc_trend_strength(forecast_points),
            chart=chart,
        )

    # ---- 移动平均回退方案 ----

    def _forecast_moving_average(
        self,
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        periods: int,
    ) -> ForecastResult:
        """移动平均 + 线性趋势外推（零依赖回退）"""
        series = pd.to_numeric(df[value_col], errors="coerce").dropna()
        dates = pd.to_datetime(df[date_col]).iloc[:len(series)]

        if len(series) < 2:
            return ForecastResult(trend_direction="flat")

        values = series.values
        window = max(2, min(5, len(values) // 3))

        # 移动平均平滑
        smoothed = pd.Series(values).rolling(window=window, center=True).mean()
        smoothed.iloc[:window//2] = values[:window//2]
        smoothed.iloc[-window//2:] = values[-window//2:]
        smoothed = smoothed.ffill().bfill().values

        # 线性趋势外推
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, smoothed, 1)
        trend_line = slope * x + intercept

        # 残差标准差
        residuals = smoothed - trend_line
        residual_std = np.std(residuals) if len(residuals) > 1 else 0

        # 构建历史点
        freq_days = self._infer_freq_days(dates)
        historical = []
        for i in range(len(values)):
            historical.append(ForecastPoint(
                date=dates.iloc[i].strftime("%Y-%m-%d") if i < len(dates) else "",
                value=round(float(values[i]), 2),
                lower_bound=round(float(values[i]) - residual_std, 2),
                upper_bound=round(float(values[i]) + residual_std, 2),
                is_historical=True,
            ))

        # 构建预测点
        last_date = dates.iloc[-1] if len(dates) > 0 else datetime.now()
        forecast_points = []
        for i in range(1, periods + 1):
            future_x = len(values) + i - 1
            pred_val = slope * future_x + intercept
            ci = 1.645 * residual_std * np.sqrt(1 + i / len(values))

            future_date = last_date + timedelta(days=freq_days * i)
            forecast_points.append(ForecastPoint(
                date=future_date.strftime("%Y-%m-%d"),
                value=round(float(pred_val), 2),
                lower_bound=round(float(pred_val - ci), 2),
                upper_bound=round(float(pred_val + ci), 2),
                is_historical=False,
            ))

        trend_direction = self._calc_trend(forecast_points)
        chart = self._build_forecast_chart(
            historical[-50:], forecast_points, value_col
        )

        return ForecastResult(
            historical=historical[-50:],
            forecast=forecast_points,
            trend_direction=trend_direction,
            trend_strength=self._calc_trend_strength(forecast_points),
            chart=chart,
        )

    # ---- 图表构建 ----

    def _build_forecast_chart(
        self,
        historical: List[ForecastPoint],
        forecast: List[ForecastPoint],
        value_col: str,
    ) -> ChartConfig:
        """构建 ECharts 预测折线图（历史实线 + 预测虚线 + 置信区间）"""
        hist_dates = [p.date for p in historical]
        hist_vals = [p.value for p in historical]

        fc_dates = [historical[-1].date] + [p.date for p in forecast] if historical else [p.date for p in forecast]
        fc_vals = [historical[-1].value] + [p.value for p in forecast] if historical else [p.value for p in forecast]

        lower_vals = [historical[-1].value] + [p.lower_bound for p in forecast] if historical else [p.lower_bound for p in forecast]
        upper_vals = [historical[-1].value] + [p.upper_bound for p in forecast] if historical else [p.upper_bound for p in forecast]

        option = {
            "title": {
                "text": f"{value_col} 趋势预测",
                "left": "center",
                "textStyle": {"fontSize": 16},
            },
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0, "data": ["历史数据", "预测值", "置信区间"]},
            "xAxis": {"type": "category", "data": hist_dates + fc_dates[1:], "boundaryGap": False},
            "yAxis": {"type": "value", "name": value_col},
            "color": ["#5470c6", "#ee6666", "#91cc75"],
            "series": [
                {
                    "name": "历史数据",
                    "type": "line",
                    "data": hist_vals + [None] * (len(fc_dates) - 1),
                    "smooth": True,
                    "symbolSize": 4,
                },
                {
                    "name": "预测值",
                    "type": "line",
                    "data": [None] * (len(hist_vals) - 1) + fc_vals,
                    "smooth": True,
                    "lineStyle": {"type": "dashed", "width": 2},
                    "symbolSize": 6,
                },
                {
                    "name": "置信区间",
                    "type": "line",
                    "data": [None] * (len(hist_vals) - 1) + upper_vals,
                    "smooth": True,
                    "lineStyle": {"type": "dotted", "width": 1, "opacity": 0},
                    "symbol": "none",
                    "areaStyle": {"color": "rgba(238, 102, 102, 0.1)"},
                    "stack": "confidence",
                },
                {
                    "name": "置信区间下界",
                    "type": "line",
                    "data": [None] * (len(hist_vals) - 1) + lower_vals,
                    "smooth": True,
                    "lineStyle": {"type": "dotted", "width": 1, "opacity": 0},
                    "symbol": "none",
                    "areaStyle": {"color": "#fff"},
                },
            ],
            "toolbox": {
                "feature": {"saveAsImage": {"title": "保存为图片"}},
            },
        }

        return ChartConfig(
            chart_type=ChartType.LINE,
            title=f"{value_col} 趋势预测",
            x_axis="date",
            y_axis=value_col,
            echarts_option=option,
        )

    # ---- 列检测 ----

    @staticmethod
    def _find_date_column(df: pd.DataFrame) -> str:
        """自动寻找日期列"""
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return str(c)
            if df[c].dtype == object:
                sample = df[c].dropna().head(5)
                if len(sample) > 0:
                    try:
                        parsed = pd.to_datetime(sample)
                        if parsed.notna().sum() >= len(sample) * 0.8:
                            return str(c)
                    except (ValueError, TypeError):
                        pass

        # 包含日期关键字的列
        date_keywords = ["date", "time", "日期", "时间", "年", "月", "日", "ds"]
        for c in df.columns:
            for kw in date_keywords:
                if kw in str(c).lower():
                    return str(c)

        return ""

    @staticmethod
    def _find_value_column(df: pd.DataFrame, exclude: str) -> str:
        """自动寻找数值列"""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            if c != exclude:
                return str(c)

        # 带数值关键词的列
        value_keywords = ["value", "amount", "price", "sales", "count", "值", "金额", "价格", "销量", "数量", "y"]
        for c in df.columns:
            if c == exclude:
                continue
            for kw in value_keywords:
                if kw in str(c).lower():
                    return str(c)

        # 回退：第一列非日期列
        for c in df.columns:
            if c != exclude and df[c].dtype in [np.float64, np.int64, float, int]:
                return str(c)

        return ""

    @staticmethod
    def _prepare_data(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """清洗：日期解析 + 数值转换 + 去NaN + 排序"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        return df

    @staticmethod
    def _infer_freq(dates: pd.Series) -> str:
        """推断时间频率 (Prophet 用)"""
        if len(dates) < 2:
            return "D"
        diff = dates.diff().dropna().median()
        days = diff.total_seconds() / 86400 if hasattr(diff, 'total_seconds') else 1
        if days >= 360:
            return "YE"
        elif days >= 28:
            return "ME"
        elif days >= 6:
            return "W"
        else:
            return "D"

    @staticmethod
    def _infer_freq_days(dates: pd.Series) -> int:
        """推断时间频率天数"""
        if len(dates) < 2:
            return 1
        diff = (dates.dropna().diff().dropna().dt.total_seconds() / 86400).median()
        return max(1, int(round(diff)))

    @staticmethod
    def _calc_trend(forecast_points: List[ForecastPoint]) -> str:
        """计算趋势方向"""
        if len(forecast_points) < 2:
            return "flat"
        first_val = forecast_points[0].value
        last_val = forecast_points[-1].value
        if first_val == 0:
            return "flat"
        change_pct = (last_val - first_val) / abs(first_val)
        if change_pct > 0.03:
            return "up"
        elif change_pct < -0.03:
            return "down"
        else:
            return "flat"

    @staticmethod
    def _calc_trend_strength(forecast_points: List[ForecastPoint]) -> float:
        """计算趋势强度（变化率）"""
        if len(forecast_points) < 2:
            return 0.0
        first_val = forecast_points[0].value
        last_val = forecast_points[-1].value
        if first_val == 0:
            return 0.0
        return round((last_val - first_val) / abs(first_val), 4)

    @staticmethod
    def _empty_chart(msg: str) -> ChartConfig:
        return ChartConfig(
            chart_type=ChartType.LINE,
            title=msg,
            echarts_option={
                "title": {"text": msg, "left": "center", "top": "center",
                          "textStyle": {"color": "#999", "fontSize": 14}},
                "xAxis": {"show": False},
                "yAxis": {"show": False},
                "series": [],
            },
        )
