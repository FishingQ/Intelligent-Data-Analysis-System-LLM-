"""
NL2SQL 生成器 —— 基于 LangChain
使用 ChatPromptTemplate + StrOutputParser + 手动 JSON 解析
(DeepSeek 对 JsonOutputParser 支持不稳定，回退到结构化 Prompt + 手动解析)
"""
import json
import re
import logging
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.shared.schemas import (
    GeneratedQuery, TableSchema, QueryIntent, Message,
)
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)


# ============================================================
# 系统提示词模板（含强制 JSON 格式指令）
# ============================================================

_JSON_FORMAT_INSTRUCTION = """
请严格按以下 JSON 格式输出，不要包含 markdown 代码块标记，只输出纯 JSON:
{{"code": "完整的SQL语句", "explanation": "对问题的理解转述", "clarifications": []}}"""

_SYSTEM_SIMPLE_QUERY = """你是一个数据库查询专家。根据表结构将用户问题转化为SQL查询。

{formatted_schema}
{history_context}

要求:
1. 只生成 SELECT 语句
2. 使用准确的字段名和表名
3. 查询可能返回大量数据时加 LIMIT 100
4. 字段别名使用中文
""" + _JSON_FORMAT_INSTRUCTION

_SYSTEM_AGGREGATION = """你是一个数据分析专家。根据表结构生成聚合统计SQL查询。

{formatted_schema}
{history_context}

要求:
1. 使用 GROUP BY 和聚合函数（SUM/AVG/COUNT/MAX/MIN）
2. ORDER BY 降序排列
3. 使用 ROUND() 保留合适小数位
4. 字段别名使用中文
""" + _JSON_FORMAT_INSTRUCTION

_SYSTEM_MULTI_TABLE = """你是一个数据库专家。根据表结构生成多表关联SQL查询。

{formatted_schema}

已知表关联关系:
{join_hints}
{history_context}

要求:
1. 使用 JOIN 关联表，明确 ON 条件
2. 字段名冲突时使用 表名.字段名
""" + _JSON_FORMAT_INSTRUCTION

_SYSTEM_FORMULA_CALC = """你是一个数据分析专家。根据表结构生成含公式计算的SQL查询。

{formatted_schema}
{history_context}

要求:
1. 在SQL中完成计算（同比/环比/占比等）
2. 使用窗口函数（LAG/LEAD）处理行间计算
3. 使用 CASE WHEN 处理条件计算
""" + _JSON_FORMAT_INSTRUCTION

# ============================================================
# Phase 3: 进阶分析意图 Prompt
# ============================================================

_SYSTEM_ANOMALY_DETECT = """你是一个数据异常检测专家。根据表结构生成SQL查询，获取用于异常检测的完整数据。

{formatted_schema}
{history_context}

要求:
1. 查询数值列的全量数据（不要聚合），供后续异常检测算法分析
2. 如果有日期列，按时间排序
3. 数据量不超过10000行时不需要 LIMIT
4. 确保包含所有数值字段用于多维异常分析
""" + _JSON_FORMAT_INSTRUCTION

_SYSTEM_FORECAST = """你是一个时序预测专家。根据表结构生成SQL查询，获取用于时序预测的历史数据。

{formatted_schema}
{history_context}

要求:
1. 查询日期列和数值列，按日期排序
2. 获取尽可能多的历史数据（不要聚合），供后续预测模型使用
3. 只取日期列和预测目标数值列，保持数据结构简洁
4. 确保包含日期列以便进行时序分析
""" + _JSON_FORMAT_INSTRUCTION

_SYSTEM_REPORT = """你是一个数据分析专家。根据表结构生成SQL查询，获取用于生成综合分析报告的全面数据。

{formatted_schema}
{history_context}

要求:
1. 查询核心数值列的汇总统计（SUM/AVG/COUNT）
2. 按主要维度分组，展示数据的多维度特征
3. 适中的聚合粒度（不要太细也不要太粗）
4. 查询结果适合用于生成图表和报告
""" + _JSON_FORMAT_INSTRUCTION

_FALLBACK_SYSTEM = """你是数据库查询专家。将用户问题转化为SQL。

{formatted_schema}
{history_context}
""" + _JSON_FORMAT_INSTRUCTION


_INTENT_SYSTEM_PROMPTS = {
    QueryIntent.SIMPLE_QUERY: _SYSTEM_SIMPLE_QUERY,
    QueryIntent.AGGREGATION: _SYSTEM_AGGREGATION,
    QueryIntent.MULTI_TABLE: _SYSTEM_MULTI_TABLE,
    QueryIntent.FORMULA_CALC: _SYSTEM_FORMULA_CALC,
    QueryIntent.ANOMALY_DETECT: _SYSTEM_ANOMALY_DETECT,
    QueryIntent.FORECAST: _SYSTEM_FORECAST,
    QueryIntent.REPORT: _SYSTEM_REPORT,
}


class NL2SQLGenerator:
    """
    NL2SQL 生成器 (LangChain 版)

    Chain: ChatPromptTemplate → ChatOpenAI → StrOutputParser → 手动JSON解析

    用法:
        gen = NL2SQLGenerator(llm_client)
        query = gen.generate("统计各部门销售额", schemas, intent)
    """

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        # 预编译各意图的 LangChain chain
        self._chains = {}
        for intent, system_tpl in _INTENT_SYSTEM_PROMPTS.items():
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_tpl),
                HumanMessagePromptTemplate.from_template("{question}"),
            ])
            self._chains[intent] = prompt | self._llm.chat_model | StrOutputParser()

        # 回退 chain
        fallback_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_FALLBACK_SYSTEM),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])
        self._fallback_chain = fallback_prompt | self._llm.chat_model | StrOutputParser()

    def generate(
        self,
        question: str,
        schemas: List[TableSchema],
        intent: QueryIntent,
        history: Optional[List[Message]] = None,
        join_hints: Optional[str] = None,
    ) -> GeneratedQuery:
        """
        使用 LangChain chain 生成 SQL 查询
        """
        from backend.nlp.schema_mapper import SchemaMapper
        mapper = SchemaMapper()
        formatted_schema = mapper.format_for_prompt(schemas)
        history_context = self._format_history(history)
        join_hints_str = join_hints or "（未自动发现关联，请根据字段名判断）"

        chain = self._chains.get(intent, self._fallback_chain)

        logger.info(f"NL2SQL生成 (LangChain) | 意图: {intent.value} | 问题: \"{question[:60]}...\"")
        try:
            raw = chain.invoke({
                "formatted_schema": formatted_schema,
                "history_context": history_context,
                "join_hints": join_hints_str,
                "question": question,
            })

            parsed = self._parse_json(raw)
            return GeneratedQuery(
                query_type=parsed.get("query_type", "sql"),
                code=parsed.get("code", ""),
                explanation=parsed.get("explanation", ""),
                clarifications=parsed.get("clarifications", []),
            )

        except Exception as e:
            logger.error(f"NL2SQL生成失败: {e}")
            return GeneratedQuery(
                query_type="sql", code="",
                explanation=f"查询生成失败: {e}",
                clarifications=[],
            )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        稳健的 JSON 解析 —— 支持 DeepSeek 各种响应格式

        策略:
        1. 直接 json.loads
        2. 从 markdown ```json ``` 块提取
        3. 从文本中提取第一个 {...}
        """
        # 1. 直接解析
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. markdown 代码块
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        # 3. 提取第一个完整 JSON 对象
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning(f"无法解析 LLM 返回为 JSON: {raw[:300]}...")
        return {}

    @staticmethod
    def _format_history(history: Optional[List[Message]]) -> str:
        if not history:
            return ""
        lines = ["\n对话历史 (供上下文参考):"]
        for msg in history[-6:]:
            role_label = "用户" if msg.role == "user" else "AI"
            lines.append(f"- {role_label}: {msg.content[:200]}")
        return "\n".join(lines)
