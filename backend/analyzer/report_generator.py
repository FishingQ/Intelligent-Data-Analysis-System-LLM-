"""
分析报告生成器 (M4-5 / Phase 3)
聚合统计分析 + 可视化 + LLM 语义理解，生成结构化的 AnalysisReport。

流程:
1. StatsCalculator → 描述性统计
2. ChartRecommender → 自动图表
3. LLM → 摘要 + 洞察 + 建议
4. 组装 MetricCard 指标卡片
"""
import logging
from typing import List, Optional

from backend.shared.schemas import (
    QueryResult, TableSchema, GeneratedQuery,
    AnalysisReport, MetricCard, ChartConfig,
)
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)


_REPORT_SYSTEM_PROMPT = """你是一个资深数据分析师。请根据以下查询结果和统计数据，生成一份专业的分析报告。

要求:
1. **summary**: 用2-3句话概述分析结果（100字内）
2. **insights**: 列出3-5条关键洞察，基于实际数据
3. **recommendations**: 给出2-4条可操作的决策建议

请严格按以下JSON格式输出，不要包含markdown标记:
{
  "title": "报告标题",
  "summary": "分析概述...",
  "insights": ["洞察1", "洞察2", "洞察3"],
  "recommendations": ["建议1", "建议2", "建议3"]
}"""


class ReportGenerator:
    """
    分析报告生成器

    用法:
        gen = ReportGenerator(llm_client)
        report = gen.generate(question, query_result, schemas)
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client

    def generate(
        self,
        question: str,
        query: GeneratedQuery,
        result: QueryResult,
        schemas: List[TableSchema],
    ) -> AnalysisReport:
        """
        生成分析报告

        Args:
            question: 用户原始问题
            query: 生成的查询对象
            result: 查询执行结果
            schemas: 数据源表结构列表

        Returns:
            AnalysisReport
        """
        # Step 1: 统计分析
        stats = self._get_stats(result)

        # Step 2: 构建图表
        charts = self._get_charts(result)

        # Step 3: 构建指标卡片
        key_metrics = self._build_metric_cards(result, stats)

        # Step 4: LLM 生成文本内容
        title = f"数据洞察报告"
        summary = ""
        insights = []
        recommendations = []

        if self._llm:
            try:
                llm_output = self._generate_with_llm(question, result, stats)
                title = llm_output.get("title", title)
                summary = llm_output.get("summary", "")
                insights = llm_output.get("insights", [])
                recommendations = llm_output.get("recommendations", [])
            except Exception as e:
                logger.warning(f"LLM报告生成失败: {e}")

        # 无 LLM 回退：基于统计生成
        if not summary:
            summary = self._build_summary_fallback(result, stats)
        if not insights:
            insights = self._build_insights_fallback(result, stats)

        return AnalysisReport(
            title=title,
            summary=summary,
            key_metrics=key_metrics,
            charts=charts,
            insights=insights,
            recommendations=recommendations,
        )

    # ---- 内部方法 ----

    @staticmethod
    def _get_stats(result: QueryResult) -> dict:
        """获取描述性统计"""
        try:
            from backend.analyzer.stats_calculator import StatsCalculator
            calc = StatsCalculator()
            return calc.describe(result)
        except Exception as e:
            logger.warning(f"统计计算失败: {e}")
            return {}

    @staticmethod
    def _get_charts(result: QueryResult) -> List[ChartConfig]:
        """获取推荐图表"""
        charts = []
        try:
            from backend.visualizer.chart_recommender import ChartRecommender
            from backend.visualizer.echarts_builder import EChartsBuilder

            recommender = ChartRecommender()
            builder = EChartsBuilder()

            config = recommender.recommend(result)
            if config:
                option = builder.build(config, result)
                config.echarts_option = option
                charts.append(config)
        except Exception as e:
            logger.warning(f"图表生成失败: {e}")

        return charts

    @staticmethod
    def _build_metric_cards(result: QueryResult, stats: dict) -> List[MetricCard]:
        """从统计结果中提取指标卡片"""
        cards = []

        # 行数卡片
        cards.append(MetricCard(
            label="数据总量",
            value=f"{result.row_count:,} 条",
            trend="flat",
        ))

        # 数值列统计卡片
        num_stats = stats.get("numeric_columns", {})
        for col_name, col_stats in list(num_stats.items())[:4]:
            mean_val = col_stats.get("mean", 0)
            max_val = col_stats.get("max", 0)
            cards.append(MetricCard(
                label=f"{col_name}（均值）",
                value=f"{mean_val:,.2f}",
                change="",
                trend="flat",
            ))
            if max_val != mean_val:
                cards.append(MetricCard(
                    label=f"{col_name}（最大值）",
                    value=f"{max_val:,.2f}",
                    change="",
                    trend="up",
                ))

        return cards[:6]

    def _generate_with_llm(
        self, question: str, result: QueryResult, stats: dict
    ) -> dict:
        """使用 LLM 生成报告文本"""
        import json

        # 准备数据预览
        preview = self._format_data_preview(result)
        stats_text = self._format_stats_text(stats)

        user_prompt = f"""用户问题: {question}

数据概览:
- 总行数: {result.row_count}
- 字段: {', '.join(result.columns)}

统计数据:
{stats_text}

数据预览（前5行）:
{preview}

请生成分析报告JSON。"""

        raw = self._llm.chat_with_system(_REPORT_SYSTEM_PROMPT, user_prompt)
        return self._parse_json(raw)

    @staticmethod
    def _format_data_preview(result: QueryResult, max_rows: int = 5) -> str:
        """格式化数据预览"""
        if not result.data:
            return "（无数据）"
        lines = []
        header = " | ".join(result.columns)
        lines.append(header)
        lines.append("-" * len(header))
        for row in result.data[:max_rows]:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)

    @staticmethod
    def _format_stats_text(stats: dict) -> str:
        """格式化统计文本"""
        lines = []
        num_stats = stats.get("numeric_columns", {})
        for col, s in num_stats.items():
            lines.append(
                f"  {col}: 均值={s.get('mean', '?')}, "
                f"中位数={s.get('median', '?')}, "
                f"标准差={s.get('std', '?')}, "
                f"最小值={s.get('min', '?')}, "
                f"最大值={s.get('max', '?')}"
            )

        cat_stats = stats.get("categorical_columns", {})
        for col, s in cat_stats.items():
            lines.append(
                f"  {col}: {s.get('unique', '?')}个分类, "
                f"最多的是'{s.get('top', '?')}'({s.get('top_freq', '?')}条)"
            )

        return "\n".join(lines) if lines else "（暂无详细统计）"

    def _build_summary_fallback(self, result: QueryResult, stats: dict) -> str:
        """无 LLM 时的摘要回退"""
        from backend.analyzer.stats_calculator import StatsCalculator
        calc = StatsCalculator()
        return calc.summary(result)

    @staticmethod
    def _build_insights_fallback(result: QueryResult, stats: dict) -> List[str]:
        """无 LLM 时的洞察回退"""
        insights = []
        num_stats = stats.get("numeric_columns", {})

        for col, s in num_stats.items():
            insights.append(
                f"📊 {col} 的平均值为 {s.get('mean', '?')}, "
                f"中位数为 {s.get('median', '?')}, "
                f"数据范围从 {s.get('min', '?')} 到 {s.get('max', '?')}"
            )

        cat_stats = stats.get("categorical_columns", {})
        for col, s in cat_stats.items():
            insights.append(
                f"📋 {col} 共有 {s.get('unique', '?')} 个不同类别, "
                f"其中 '{s.get('top', '?')}' 出现最多（{s.get('top_freq', '?')}次）"
            )

        return insights[:5] if insights else ["数据已成功查询，请查看详细结果。"]

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """稳健的 JSON 解析"""
        import json
        import re

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning(f"报告JSON解析失败: {raw[:200]}...")
        return {}
