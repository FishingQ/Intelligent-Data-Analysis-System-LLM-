"""
对话核心接口 POST /api/chat
一条请求贯穿所有模块的完整链路
"""
import logging
import traceback
from typing import List, Dict, Optional

from fastapi import APIRouter, HTTPException

from backend.shared.schemas import (
    ChatRequest, ChatResponse, QueryResult, QueryIntent,
    TableSchema, DataSourceConfig, DataSourceType,
    ChartConfig, ChartType,
)
from backend.llm.client import LLMClient
from backend.nlp.intent_classifier import IntentClassifier
from backend.nlp.schema_mapper import SchemaMapper
from backend.nlp.nl2sql_generator import NL2SQLGenerator
from backend.nlp.sql_validator import SQLValidator
from backend.nlp.result_explainer import ResultExplainer
from backend.data_sources.factory import DataSourceFactory
from backend.data_sources.base import BaseAdapter
from backend.analyzer.query_executor import QueryExecutor
# Phase 3 modules
from backend.analyzer.anomaly_detector import AnomalyDetector
from backend.analyzer.forecaster import Forecaster
from backend.analyzer.report_generator import ReportGenerator
from backend.nlp.conversation_manager import get_conversation_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# 服务实例 (生产环境用依赖注入)
# ============================================================

# 这些实例在 register_router() 时由 main.py 注入
_llm_client: Optional[LLMClient] = None
_intent_classifier: Optional[IntentClassifier] = None
_schema_mapper: Optional[SchemaMapper] = None
_nl2sql_generator: Optional[NL2SQLGenerator] = None
_sql_validator: Optional[SQLValidator] = None
_result_explainer: Optional[ResultExplainer] = None
_query_executor: Optional[QueryExecutor] = None

# 数据源配置注册表 (source_id → DataSourceConfig)
_source_registry: Dict[str, DataSourceConfig] = {}


def init_chat_services(llm_client: LLMClient):
    """初始化所有 chat 模块依赖的服务"""
    global _llm_client, _intent_classifier, _schema_mapper
    global _nl2sql_generator, _sql_validator, _result_explainer, _query_executor

    _llm_client = llm_client
    _intent_classifier = IntentClassifier(llm_client)
    _schema_mapper = SchemaMapper()
    _nl2sql_generator = NL2SQLGenerator(llm_client)
    _sql_validator = SQLValidator(max_sql_length=10000)
    _result_explainer = ResultExplainer(llm_client)
    _query_executor = QueryExecutor()

    logger.info("✅ Chat服务模块初始化完成")


def register_datasource(config: DataSourceConfig):
    """注册数据源配置"""
    _source_registry[config.source_id] = config
    logger.info(f"注册数据源: {config.display_name} (id={config.source_id})")


def get_active_sources(source_ids: List[str]) -> List[DataSourceConfig]:
    """获取活跃数据源配置列表"""
    if not source_ids:
        return list(_source_registry.values())
    return [
        cfg for sid, cfg in _source_registry.items()
        if sid in source_ids
    ]


# ============================================================
# 路由
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    核心对话接口 —— 一条请求贯穿所有模块

    完整链路:
    1. 获取数据源 Schema        (模块3)
    2. 意图分类                 (模块2-1)
    3. Schema格式化             (模块2-2)
    4. NL2SQL生成               (模块2-3)
    5. SQL校验                  (模块2-4)
    6. 执行查询                 (模块4-1)
    7. 结果解释                 (模块2-5)
    8. 图表推荐 + 组装响应      (模块5)
    """
    try:
        # Step 0: 参数校验
        if not request.question.strip():
            return ChatResponse(
                conversation_id=request.conversation_id,
                answer_text="请输入您的问题。",
                error="问题为空",
            )

        # Step 1: 获取数据源和 Schema
        source_configs = get_active_sources(request.data_source_ids)
        if not source_configs:
            return ChatResponse(
                conversation_id=request.conversation_id,
                answer_text=(
                    "当前没有连接任何数据源。\n\n"
                    "请先在侧边栏上传数据文件（Excel / CSV / SQLite），"
                    "或配置数据库连接。"
                ),
                error="无可用数据源",
            )

        all_schemas = DataSourceFactory.get_active_schemas_from_configs(source_configs)
        if not all_schemas:
            return ChatResponse(
                conversation_id=request.conversation_id,
                answer_text="数据源连接成功但未能读取表结构，请检查数据文件是否完整。",
                error="Schema读取失败",
            )

        logger.info(
            f"对话请求 | conv={request.conversation_id[:8]} | "
            f"问题=\"{request.question[:80]}...\" | "
            f"数据源={[s.display_name for s in source_configs]}"
        )

        # Step 2: 意图分类
        intent = _intent_classifier.classify(request.question)

        # Step 3-4: NL2SQL 生成
        # 先获取表关联提示
        join_hints_raw = _schema_mapper.suggest_joins(all_schemas)
        join_hints = "\n".join(
            f"- {j['left']}.{j['on']} ↔ {j['right']}.{j['on']} "
            f"(置信度:{j['confidence']:.0%})"
            for j in join_hints_raw[:5]
        ) if join_hints_raw else ""

        query = _nl2sql_generator.generate(
            question=request.question,
            schemas=all_schemas,
            intent=intent,
            history=None,
            join_hints=join_hints,
        )

        # 如果 LLM 需要澄清问题
        if query.clarifications and not query.code:
            return ChatResponse(
                conversation_id=request.conversation_id,
                answer_text=(
                    f"我需要确认以下几点：\n\n"
                    + "\n".join(f"• {c}" for c in query.clarifications)
                    + "\n\n请补充信息后重新提问。"
                ),
                generated_sql="",
            )

        # Step 5: SQL 校验
        is_valid = True
        errors = []
        allowed_tables = [t.table_name for t in all_schemas]
        allowed_columns = [c.name for t in all_schemas for c in t.columns]

        if query.query_type == "sql" and query.code:
            is_valid, errors, query.code = _sql_validator.validate(
                query.code, allowed_tables, allowed_columns
            )

        if not is_valid:
            return ChatResponse(
                conversation_id=request.conversation_id,
                answer_text=(
                    f"生成的查询语句存在问题：\n\n"
                    + "\n".join(f"• {e}" for e in errors)
                    + "\n\n请换个方式提问或检查数据源。"
                ),
                generated_sql=query.code,
                error="SQL校验失败",
            )

        # Step 6: 执行查询 —— 多数据源支持
        # 遍历所有活跃数据源，找到包含目标表的适配器执行
        result = None
        last_error = None
        for src_cfg in source_configs:
            adapter = DataSourceFactory.get_or_create(src_cfg)
            if not adapter or not adapter.is_connected:
                continue
            result = _query_executor.execute(query, adapter)
            if result.success:
                break
            last_error = result.error_message

        if result is None:
            return ChatResponse(
                conversation_id=request.conversation_id,
                answer_text="数据源连接已断开，请重新连接。",
                error="数据源未连接",
            )

        if not result.success:
            return ChatResponse(
                conversation_id=request.conversation_id,
                answer_text=(
                    f"查询执行失败：{last_error}\n\n"
                    f"SQL:\n```sql\n{query.code}\n```"
                ),
                generated_sql=query.code,
                error=f"SQL执行失败: {last_error}",
            )

        # ---- Phase 3: 进阶分析 ----
        anomaly_result = None
        forecast_result = None
        report = None

        if result.success and result.row_count > 0:
            if intent == QueryIntent.ANOMALY_DETECT:
                detector = AnomalyDetector()
                anomaly_result = detector.detect(result)
                logger.info(f"异常检测完成: {anomaly_result.anomaly_count}个异常点")

            elif intent == QueryIntent.FORECAST:
                forecaster = Forecaster()
                forecast_result = forecaster.forecast(result)
                logger.info(
                    f"时序预测完成: {len(forecast_result.forecast)}期, "
                    f"趋势={forecast_result.trend_direction}"
                )

            elif intent == QueryIntent.REPORT:
                report_gen = ReportGenerator(_llm_client)
                report = report_gen.generate(
                    request.question, query, result, all_schemas
                )
                logger.info(f"报告生成完成: {report.title}")

        # Step 7: 结果解释
        explanation = _result_explainer.explain(
            request.question, query, result
        )

        # Phase 3 上下文增强解释
        if anomaly_result and anomaly_result.anomaly_count > 0:
            explanation += (
                f"\n\n🔍 **异常检测**（{anomaly_result.method}法）："
                f"共发现 {anomaly_result.anomaly_count} 个异常数据点"
                f"（异常率 {anomaly_result.anomaly_rate:.1%}）。"
            )
            high_severity = [a for a in anomaly_result.anomalies if a.severity == "high"]
            if high_severity:
                explanation += f"其中 {len(high_severity)} 个高度异常，需重点关注。"

        if forecast_result and forecast_result.forecast:
            direction_text = {"up": "📈 上升", "down": "📉 下降", "flat": "➡️ 平稳"}
            explanation += (
                f"\n\n🔮 **趋势预测**：未来趋势为"
                f"{direction_text.get(forecast_result.trend_direction, '平稳')}"
                f"（变化率 {forecast_result.trend_strength:+.2%}）。"
            )

        # Step 8: 图表推荐 + ECharts构建 (Phase 2 完整实现)
        chart = None
        if result.success and result.row_count > 0 and len(result.columns) >= 2:
            # Phase 3 图表优先
            if anomaly_result and anomaly_result.chart:
                chart = anomaly_result.chart
            elif forecast_result and forecast_result.chart:
                chart = forecast_result.chart
            elif report and report.charts:
                chart = report.charts[0]
            else:
                chart = _build_chart(result)

        # Step 9: 生成追问建议
        suggestions = _generate_suggestions(result, all_schemas)

        # 持久化对话历史
        conv_mgr = get_conversation_manager()
        conv_mgr.add_message(request.conversation_id, "user", request.question)
        conv_mgr.add_message(request.conversation_id, "assistant", explanation)

        # 组装响应
        return ChatResponse(
            conversation_id=request.conversation_id,
            answer_text=explanation,
            generated_sql=query.code if query.query_type == "sql" else "",
            query_result=result if result.success else None,
            chart=chart,
            report=report,
            anomaly=anomaly_result,
            forecast=forecast_result,
            suggested_questions=suggestions,
        )

    except Exception as e:
        logger.error(f"对话处理异常: {traceback.format_exc()}")
        return ChatResponse(
            conversation_id=request.conversation_id,
            answer_text=f"系统处理请求时出现错误：{e}\n\n请重试或检查后端日志。",
            error=str(e),
        )


# ============================================================
# 辅助函数
# ============================================================

def _build_chart(result: QueryResult) -> Optional[ChartConfig]:
    """
    Phase 2 完整图表构建: ChartRecommender → EChartsBuilder
    """
    try:
        from backend.visualizer.chart_recommender import ChartRecommender
        from backend.visualizer.echarts_builder import EChartsBuilder

        recommender = ChartRecommender()
        config = recommender.recommend(result)

        if config is None:
            return None

        builder = EChartsBuilder()
        option = builder.build(config, result)

        config.echarts_option = option
        return config

    except Exception as e:
        logger.warning(f"图表构建失败: {e}")
        return None


def _analyze_stats(result: QueryResult) -> Optional[dict]:
    """
    Phase 2 统计分析 (M4-2)
    """
    try:
        from backend.analyzer.stats_calculator import StatsCalculator
        calc = StatsCalculator()
        return calc.describe(result)
    except Exception as e:
        logger.warning(f"统计分析失败: {e}")
        return None


def _generate_suggestions(
    result: QueryResult,
    schemas: List[TableSchema],
) -> List[str]:
    """基于当前查询结果和数据源生成追问建议"""
    suggestions = []

    if not schemas:
        return suggestions

    table = schemas[0]
    num_cols = [c.name for c in table.columns if _is_numeric_col(c)]
    date_cols = [c.name for c in table.columns if _is_date_col(c)]

    if result.success and result.row_count > 0:
        suggestions.append("📊 用图表展示这些数据")
        suggestions.append("📋 导出查询结果为Excel")

    if num_cols:
        suggestions.append(f"📈 统计{num_cols[0]}的排名分布")

    if num_cols and len(num_cols) >= 2:
        suggestions.append(f"🔍 分析{num_cols[0]}和{num_cols[1]}的关系")

    if date_cols and num_cols:
        suggestions.append(f"📅 按时间趋势查看{num_cols[0]}")

    if len(table.columns) > 3:
        suggestions.append("📝 生成数据分析报告")

    return suggestions[:4]


def _is_numeric_col(col) -> bool:
    dt = col.data_type.upper()
    return any(t in dt for t in ["INT", "FLOAT", "DOUBLE", "REAL", "NUMERIC", "NUMBER"])


def _is_date_col(col) -> bool:
    dt = col.data_type.upper()
    return any(t in dt for t in ["DATE", "TIME", "DATETIME", "TIMESTAMP"])
