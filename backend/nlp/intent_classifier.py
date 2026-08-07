"""
意图分类器 —— 基于 LangChain
判断用户自然语言问题的分析意图，决定后续处理路径
"""
import logging
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.shared.schemas import QueryIntent, Message
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)


_INTENT_CLASSIFY_SYSTEM = """你是一个数据分析意图识别助手。分析用户的问题，判断属于以下哪种意图类型:

- simple_query: 简单数据查询、筛选、查看明细
- aggregation: 聚合统计（求和、平均、计数、排名、分组汇总）
- multi_table: 需要关联多张表的查询
- formula_calc: 需要计算比率、同比、环比、占比
- anomaly_detect: 检测异常数据、离群值
- forecast: 预测未来趋势
- report: 要求生成分析报告
- other: 其他类型

请只输出意图类型（英文），不要输出其他任何内容。"""


class IntentClassifier:
    """
    意图分类器 (LangChain 版)

    两阶段分类:
    1. 关键词快速匹配 (零延迟)
    2. LangChain chain (关键词匹配失败时)

    Chain: PromptTemplate → ChatOpenAI → StrOutputParser

    用法:
        classifier = IntentClassifier(llm_client)
        intent = classifier.classify("统计各部门销售额")
    """

    KEYWORD_RULES = [
        (QueryIntent.REPORT,        ["报告", "总结", "综合分析", "出个报告"]),
        (QueryIntent.FORECAST,      ["预测", "趋势", "走势", "未来", "预估", "推算"]),
        (QueryIntent.ANOMALY_DETECT, ["异常", "离群", "突出", "不正常", "奇怪", "检测"]),
        (QueryIntent.FORMULA_CALC,  ["同比", "环比", "占比", "比率", "计算", "公式"]),
        (QueryIntent.MULTI_TABLE,   ["关联", "联合", "对应", "跨表", "连接", "结合"]),
        (QueryIntent.AGGREGATION,   ["统计", "汇总", "求和", "平均", "最多", "最少", "排名",
                                      "各", "每个", "按", "分组", "多少", "数量", "总"]),
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client
        # 构建 LangChain chain (仅 LLM 阶段使用)
        if self._llm:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(_INTENT_CLASSIFY_SYSTEM),
                HumanMessagePromptTemplate.from_template("{question}"),
            ])
            self._chain = prompt | self._llm.chat_model | StrOutputParser()
        else:
            self._chain = None

    def classify(
        self,
        question: str,
        history: Optional[List[Message]] = None,
    ) -> QueryIntent:
        """
        分类用户问题意图

        Args:
            question: 用户自然语言问题
            history: 对话历史（可选）

        Returns:
            QueryIntent 枚举值
        """
        # Stage 1: 关键词匹配
        for intent, keywords in self.KEYWORD_RULES:
            if any(kw in question for kw in keywords):
                logger.info(f"意图分类(关键词): {intent.value} ← \"{question[:50]}...\"")
                return intent

        # Stage 2: LangChain + LLM
        if self._chain:
            logger.info(f"意图分类(LLM/LangChain): \"{question[:50]}...\"")
            return self._llm_classify(question)

        # 无 LLM 时默认简单查询
        logger.info(f"意图分类(默认): simple_query")
        return QueryIntent.SIMPLE_QUERY

    def _llm_classify(self, question: str) -> QueryIntent:
        """使用 LangChain chain 分类意图"""
        try:
            raw = self._chain.invoke({"question": question})
            raw = raw.strip().lower()
            for intent in QueryIntent:
                if intent.value in raw:
                    return intent
        except Exception as e:
            logger.warning(f"LLM意图分类失败: {e}")

        return QueryIntent.SIMPLE_QUERY
