"""
ECharts 构建器 (M5-2)
将 ChartConfig + QueryResult 转化为完整的 ECharts option JSON
支持: Line / Bar / Pie / Scatter / Heatmap
"""
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from backend.shared.schemas import ChartConfig, ChartType, QueryResult

logger = logging.getLogger(__name__)

# ECharts 默认主题色
_THEME = {
    "colors": [
        "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
        "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#48b4bb",
    ],
    "backgroundColor": "#ffffff",
    "textColor": "#333333",
}


class EChartsBuilder:
    """
    ECharts 构建器

    将 ChartConfig + 查询数据 → ECharts 完整 option

    用法:
        builder = EChartsBuilder()
        option = builder.build(config, result)      # 返回 ECharts option dict
        html = builder.build_html(config, result)    # 返回可直接嵌入的 HTML
    """

    def build(self, config: ChartConfig, result: QueryResult) -> Dict[str, Any]:
        """
        构建 ECharts option

        Args:
            config: 图表配置（来自 ChartRecommender）
            result: 查询结果数据

        Returns:
            ECharts option dict
        """
        if not result.success or not result.data:
            return self._empty_chart("无数据")

        df = pd.DataFrame(result.data, columns=result.columns)

        builders = {
            ChartType.LINE: self._build_line,
            ChartType.BAR: self._build_bar,
            ChartType.PIE: self._build_pie,
            ChartType.SCATTER: self._build_scatter,
            ChartType.HEATMAP: self._build_heatmap,
        }

        builder = builders.get(config.chart_type, self._build_bar)
        option = builder(config, df)
        option["title"] = {"text": config.title or "", "left": "center", "textStyle": {"fontSize": 16}}
        option["color"] = _THEME["colors"]
        option["toolbox"] = {
            "feature": {
                "saveAsImage": {"title": "保存为图片"},
                "dataView": {"title": "数据视图", "readOnly": True},
            }
        }

        return option

    def build_html(
        self,
        config: ChartConfig,
        result: QueryResult,
        height: int = 400,
        width: str = "100%",
    ) -> str:
        """
        构建可直接嵌入 Streamlit 的 HTML + ECharts 代码

        Returns:
            HTML 字符串（含 <script>）
        """
        import json
        option = self.build(config, result)
        option_json = json.dumps(option, ensure_ascii=False)
        chart_id = f"chart_{id(config)}_{hash(option_json) % 100000}"
        return f"""
        <div id="{chart_id}" style="width:{width};height:{height}px;"></div>
        <script>
            (function() {{
                var chart = echarts.init(document.getElementById('{chart_id}'));
                chart.setOption({option_json});
                window.addEventListener('resize', function(){{ chart.resize(); }});
            }})();
        </script>
        """

    # ---- 各类图表构建 ----

    def _build_line(self, cfg: ChartConfig, df: pd.DataFrame) -> dict:
        x_col = cfg.x_axis or df.columns[0]
        y_col = cfg.y_axis or self._pick_numeric(df, skip=[x_col])
        x_data = df[x_col].astype(str).tolist() if x_col in df.columns else list(range(len(df)))

        series = []
        num_cols = self._all_numeric(df, skip=[x_col])

        if num_cols:
            for col in num_cols:
                vals = df[col].fillna(0).tolist()
                series.append({
                    "name": str(col),
                    "type": "line",
                    "data": vals,
                    "smooth": True,
                    "symbolSize": 6,
                })
        elif y_col in df.columns:
            vals = df[y_col].fillna(0).tolist()
            series.append({
                "name": str(y_col),
                "type": "line",
                "data": vals,
                "smooth": True,
                "symbolSize": 6,
            })

        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0, "data": [s["name"] for s in series]} if len(series) > 1 else {},
            "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": self._auto_rotate(x_data)}},
            "yAxis": {"type": "value"},
            "series": series,
        }

    def _build_bar(self, cfg: ChartConfig, df: pd.DataFrame) -> dict:
        x_col = cfg.x_axis or df.columns[0]
        y_col = cfg.y_axis or self._pick_numeric(df, skip=[x_col])

        # Y轴倒序让最高值在顶部
        df_sorted = df.sort_values(y_col, ascending=True) if y_col in df.columns else df
        x_data = df_sorted[x_col].astype(str).tolist() if x_col in df_sorted.columns else []

        vals = df_sorted[y_col].fillna(0).tolist() if y_col in df_sorted.columns else []

        # 自动判断条形图方向：分类多时用横向
        use_horizontal = len(x_data) > 10

        option = {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": "3%", "right": "8%", "bottom": "3%", "containLabel": True},
        }

        if use_horizontal:
            option["yAxis"] = {"type": "category", "data": x_data, "inverse": True}
            option["xAxis"] = {"type": "value"}
            option["series"] = [{
                "name": str(y_col),
                "type": "bar",
                "data": vals,
                "itemStyle": {"borderRadius": [0, 4, 4, 0]},
                "label": {"show": True, "position": "right"},
            }]
        else:
            option["xAxis"] = {
                "type": "category",
                "data": x_data,
                "axisLabel": {"rotate": self._auto_rotate(x_data)},
            }
            option["yAxis"] = {"type": "value"}
            option["series"] = [{
                "name": str(y_col),
                "type": "bar",
                "data": vals,
                "itemStyle": {"borderRadius": [4, 4, 0, 0]},
                "barMaxWidth": 40,
            }]

        return option

    def _build_pie(self, cfg: ChartConfig, df: pd.DataFrame) -> dict:
        x_col = cfg.x_axis or df.columns[0]
        y_col = cfg.y_axis or self._pick_numeric(df, skip=[x_col])

        data = []
        if x_col in df.columns and y_col in df.columns:
            for _, row in df.iterrows():
                data.append({
                    "name": str(row[x_col]),
                    "value": float(row[y_col]) if pd.notna(row[y_col]) else 0,
                })

        return {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"bottom": 0, "type": "scroll"},
            "series": [{
                "type": "pie",
                "radius": ["45%", "72%"],
                "center": ["50%", "45%"],
                "avoidLabelOverlap": True,
                "itemStyle": {
                    "borderRadius": 4,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "label": {"show": True, "formatter": "{b}\n{d}%"},
                "emphasis": {
                    "label": {"show": True, "fontSize": 18, "fontWeight": "bold"},
                    "scaleSize": 10,
                },
                "data": data,
            }],
        }

    def _build_scatter(self, cfg: ChartConfig, df: pd.DataFrame) -> dict:
        x_col = cfg.x_axis or self._pick_numeric(df, index=0)
        y_col = cfg.y_axis or self._pick_numeric(df, index=1)

        data = []
        if x_col in df.columns and y_col in df.columns:
            data = [
                [float(df.iloc[i][x_col]), float(df.iloc[i][y_col])]
                for i in range(len(df))
                if pd.notna(df.iloc[i][x_col]) and pd.notna(df.iloc[i][y_col])
            ]

        return {
            "tooltip": {"trigger": "item", "formatter": f"{x_col}: {{c[0]}}<br/>{y_col}: {{c[1]}}"},
            "xAxis": {"type": "value", "name": str(x_col), "splitLine": {"show": False}},
            "yAxis": {"type": "value", "name": str(y_col), "splitLine": {"show": False}},
            "series": [{
                "type": "scatter",
                "data": data,
                "symbolSize": 10,
                "emphasis": {"scaleSize": 15},
            }],
        }

    def _build_heatmap(self, cfg: ChartConfig, df: pd.DataFrame) -> dict:
        """将矩阵数据构建为热力图"""
        num_cols = self._all_numeric(df)
        if len(num_cols) >= 3:
            # 假定: 前两列为行列坐标，第三列为值
            data = df.values.tolist()
            x_vals = sorted(df[num_cols[0]].unique().tolist())
            y_vals = sorted(df[num_cols[1]].unique().tolist())

            heat_data = [
                [x_vals.index(row[0]), y_vals.index(row[1]), float(row[2])]
                for row in data
                if pd.notna(row[2])
            ]

            return {
                "tooltip": {"position": "top"},
                "xAxis": {"type": "category", "data": [str(v) for v in x_vals], "splitArea": {"show": True}},
                "yAxis": {"type": "category", "data": [str(v) for v in y_vals], "splitArea": {"show": True}},
                "visualMap": {"min": min(h[2] for h in heat_data), "max": max(h[2] for h in heat_data),
                              "calculable": True, "orient": "horizontal", "left": "center", "bottom": 0},
                "series": [{"type": "heatmap", "data": heat_data, "label": {"show": True}}],
            }

        # 回退到柱状图
        return self._build_bar(cfg, df)

    # ---- 工具方法 ----

    @staticmethod
    def _pick_numeric(df: pd.DataFrame, skip: List[str] = None, index: int = 0) -> str:
        """选择数值列"""
        skip = skip or []
        nums = [c for c in df.select_dtypes(include=['number']).columns if c not in skip]
        if len(nums) > index:
            return nums[index]
        return df.columns[index] if len(df.columns) > index else ""

    @staticmethod
    def _all_numeric(df: pd.DataFrame, skip: List[str] = None) -> List[str]:
        """所有数值列"""
        skip = skip or []
        return [c for c in df.select_dtypes(include=['number']).columns if c not in skip]

    @staticmethod
    def _auto_rotate(labels: List[str]) -> int:
        """自动判断标签旋转角度"""
        if not labels:
            return 0
        max_len = max(len(str(l)) for l in labels)
        if max_len > 8:
            return 45
        if max_len > 4:
            return 20
        return 0

    @staticmethod
    def _empty_chart(msg: str = "暂无数据") -> dict:
        return {
            "title": {"text": msg, "left": "center", "top": "center", "textStyle": {"color": "#999", "fontSize": 18}},
            "xAxis": {"show": False},
            "yAxis": {"show": False},
            "series": [],
        }
