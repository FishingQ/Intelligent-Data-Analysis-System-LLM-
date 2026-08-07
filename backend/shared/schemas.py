"""
全系统统一数据结构定义
所有模块间通信必须使用此文件中定义的数据对象
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from enum import Enum
from datetime import datetime


# ============================================================
# 数据源相关
# ============================================================

class DataSourceType(str, Enum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    EXCEL = "excel"
    CSV = "csv"


class ColumnInfo(BaseModel):
    """字段元信息"""
    name: str
    data_type: str                         # INTEGER / TEXT / REAL / DATE / VARCHAR...
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    referenced_table: Optional[str] = None # 外键引用的表名
    sample_values: List[Any] = Field(default_factory=list)


class TableSchema(BaseModel):
    """表/Sheet 结构信息"""
    table_name: str
    columns: List[ColumnInfo] = Field(default_factory=list)
    row_count: int = 0
    description: str = ""                  # LLM生成的表中文描述


class DataSourceConfig(BaseModel):
    """数据源连接配置"""
    source_type: DataSourceType
    source_id: str                         # 唯一标识
    display_name: str                      # 界面显示名
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    # 示例: {"file_path": "data/xxx.sqlite"} 或 {"host":"localhost","port":3306,...}


# ============================================================
# 查询相关
# ============================================================

class QueryIntent(str, Enum):
    SIMPLE_QUERY = "simple_query"           # 简单查询/筛选
    AGGREGATION = "aggregation"             # 聚合统计
    MULTI_TABLE = "multi_table"             # 多表关联
    FORMULA_CALC = "formula_calc"           # 公式/比率计算
    ANOMALY_DETECT = "anomaly_detect"       # 异常检测
    FORECAST = "forecast"                   # 时序预测
    REPORT = "report"                       # 决策报告
    OTHER = "other"


class GeneratedQuery(BaseModel):
    """NL2SQL / NL2Python 生成的查询对象"""
    query_type: Literal["sql", "python", "hybrid"] = "sql"
    code: str                               # SQL语句 或 Python代码
    target_source_id: str = ""              # 目标数据源ID
    target_table: str = ""                  # 目标表名
    explanation: str = ""                   # AI对用户问题的转述
    clarifications: List[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """查询执行结果"""
    success: bool
    columns: List[str] = Field(default_factory=list)
    data: List[List[Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error_message: str = ""


# ============================================================
# 对话相关
# ============================================================

class Message(BaseModel):
    """单条对话消息"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)


class Conversation(BaseModel):
    """一次完整对话会话"""
    conversation_id: str
    title: str = "新对话"
    messages: List[Message] = Field(default_factory=list)
    active_data_source_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ============================================================
# 可视化相关
# ============================================================

class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"


class ChartConfig(BaseModel):
    """前端渲染图表所需的完整配置"""
    chart_type: ChartType = ChartType.BAR
    title: str = ""
    x_axis: str = ""                        # X轴字段名
    y_axis: str = ""                        # Y轴字段名
    echarts_option: Dict[str, Any] = Field(default_factory=dict)  # ECharts完整option
    series_data: Dict[str, List] = Field(default_factory=dict)


class MetricCard(BaseModel):
    """KPI指标卡片"""
    label: str                              # "总销售额"
    value: str                              # "¥1.2亿"
    change: str = ""                        # "+8%"
    trend: Literal["up", "down", "flat"] = "flat"


class AnalysisReport(BaseModel):
    """决策分析报告"""
    title: str = ""
    summary: str = ""                       # 分析概述
    key_metrics: List[MetricCard] = Field(default_factory=list)
    charts: List[ChartConfig] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ============================================================
# Phase 3: 异常检测 / 时序预测
# ============================================================

class AnomalyPoint(BaseModel):
    """单个异常数据点"""
    row_index: int                          # 数据行索引
    column: str                             # 异常所在列名
    value: float                            # 异常值
    z_score: float = 0.0                    # Z-Score (|z|越大越异常)
    severity: Literal["low", "medium", "high"] = "medium"
    reason: str = ""                        # 中文解释（如"超过上界2.3倍"）


class AnomalyResult(BaseModel):
    """异常检测完整结果"""
    anomalies: List[AnomalyPoint] = Field(default_factory=list)
    total_rows: int = 0
    anomaly_count: int = 0
    anomaly_rate: float = 0.0               # 异常比例
    method: str = ""                        # "iqr" | "zscore" | "isolation_forest"
    chart: Optional[ChartConfig] = None     # 异常标注散点图


class ForecastPoint(BaseModel):
    """单个预测数据点"""
    date: str                               # 日期 (YYYY-MM-DD)
    value: float                            # 预测值 / 历史值
    lower_bound: float = 0.0                # 置信区间下界
    upper_bound: float = 0.0                # 置信区间上界
    is_historical: bool = True              # True=历史数据, False=预测值


class ForecastResult(BaseModel):
    """时序预测完整结果"""
    historical: List[ForecastPoint] = Field(default_factory=list)
    forecast: List[ForecastPoint] = Field(default_factory=list)
    trend_direction: Literal["up", "down", "flat"] = "flat"
    trend_strength: float = 0.0             # 趋势变化率
    chart: Optional[ChartConfig] = None     # 预测折线图


# ============================================================
# API 请求 / 响应
# ============================================================

class ChatRequest(BaseModel):
    """POST /api/chat 请求体"""
    conversation_id: str
    question: str
    data_source_ids: List[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """POST /api/chat 响应体"""
    conversation_id: str
    answer_text: str                        # 自然语言回答
    generated_sql: str = ""                 # 生成的SQL(便于调试)
    query_result: Optional[QueryResult] = None
    chart: Optional[ChartConfig] = None
    report: Optional[AnalysisReport] = None
    anomaly: Optional[AnomalyResult] = None         # Phase 3: 异常检测结果
    forecast: Optional[ForecastResult] = None       # Phase 3: 时序预测结果
    suggested_questions: List[str] = Field(default_factory=list)
    error: str = ""


class DataSourceUploadResponse(BaseModel):
    """POST /api/datasources/upload 响应"""
    source_id: str
    display_name: str
    source_type: str
    filepath: str
    tables: List[TableSchema] = Field(default_factory=list)


class ConversationListItem(BaseModel):
    """对话列表项"""
    conversation_id: str
    title: str
    created_at: datetime
    message_count: int = 0
