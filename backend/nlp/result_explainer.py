"""
结果解释器 —— 基于 LangChain
将查询结果转化为用户友好的自然语言解释
"""
import logging
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from backend.shared.schemas import QueryResult, GeneratedQuery
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)


_EXPLAIN_SYSTEM = """你是一个数据分析助手。请用自然语言向用户解释以下查询结果。

要求:
1. 用口语化的语言总结关键发现（控制在150字以内）
2. 如果有明显的排名、占比、趋势，请主动指出
3. 如果有异常或值得注意的数据点，请标注出来
4. 不要编造数据中没有的信息"""


class ResultExplainer:
    """
    结果解释器 (LangChain 版)

    用法:
        explainer = ResultExplainer(llm_client)
        text = explainer.explain(question, query, result)
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client
        if self._llm:
            self._prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(_EXPLAIN_SYSTEM),
                HumanMessagePromptTemplate.from_template("""用户问题: {question}
AI的理解: {explanation}
查询耗时: {execution_time_ms:.0f}毫秒

查询结果:
- 列: {columns}
- 行数: {row_count}
- 数据(前{preview_rows}行):
{data_preview}

请解释这个结果。"""),
            ])
            self._chain = self._prompt | self._llm.chat_model
        else:
            self._chain = None

    def explain(
        self,
        question: str,
        query: GeneratedQuery,
        result: QueryResult,
    ) -> str:
        """
        生成查询结果的自然语言解释
        """
        # 非成功 —— 固定话术
        if not result.success:
            return (
                f"抱歉，查询执行失败了。\n\n"
                f"错误信息: {result.error_message}\n\n"
                f"请检查数据是否完整，或换个方式提问。"
            )

        if result.row_count == 0:
            return (
                f"查询已执行完成，但没有找到匹配的数据。\n\n"
                f"可能的原因:\n"
                f"1. 筛选条件过严，没有符合条件的数据\n"
                f"2. 数据中确实不包含相关信息\n\n"
                f"建议尝试放宽条件或调整查询范围。"
            )

        # 小结果集——直接列表展示
        if result.row_count <= 5:
            return self._explain_small_result(question, query, result)

        # LLM 解释
        if self._chain:
            try:
                return self._explain_with_llm(question, query, result)
            except Exception as e:
                logger.warning(f"LLM结果解释失败，使用统计概要: {e}")

        return self._explain_with_stats(question, query, result)

    def _explain_small_result(self, question, query, result) -> str:
        lines = [f"查询完成，返回 {result.row_count} 条结果。\n"]
        if query.explanation:
            lines.append(f"（{query.explanation}）\n")
        for i, row in enumerate(result.data, 1):
            row_items = [f"  {col}: {val}" for col, val in zip(result.columns, row)]
            lines.append(f"{i}. " + " | ".join(row_items))
        return "\n".join(lines)

    def _explain_with_llm(self, question, query, result) -> str:
        preview_rows = min(10, result.row_count)
        data_preview = "\n".join(str(row) for row in result.data[:preview_rows])

        response = self._chain.invoke({
            "question": question,
            "explanation": query.explanation or "（无）",
            "columns": ", ".join(result.columns),
            "row_count": result.row_count,
            "preview_rows": preview_rows,
            "data_preview": data_preview or "（空）",
            "execution_time_ms": result.execution_time_ms,
        })
        return response.content

    def _explain_with_stats(self, question, query, result) -> str:
        lines = [
            f"查询完成！共找到 {result.row_count} 条结果，"
            f"包含 {len(result.columns)} 个字段。\n"
        ]
        if query.explanation:
            lines.append(f"查询逻辑: {query.explanation}\n")
        lines.append(f"返回字段: {', '.join(result.columns)}")
        lines.append(f"前3条数据预览:")
        for row in result.data[:3]:
            lines.append(f"  {row}")
        if result.row_count > 3:
            lines.append(f"  ... (还有 {result.row_count - 3} 条)")
        lines.append(f"\n查询耗时: {result.execution_time_ms:.0f}毫秒")
        return "\n".join(lines)
