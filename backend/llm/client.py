"""
统一LLM调用客户端 —— 基于 LangChain
支持: DeepSeek / 九天大模型 / OpenAI / Qwen (均为 OpenAI 兼容 API)
"""
import os
import logging
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from backend.llm.jiutian_adapter import JiutianChatModel

logger = logging.getLogger(__name__)


# ============================================================
# Provider 默认配置 (base_url 映射)
# ============================================================

PROVIDER_DEFAULTS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "jiutian": {
        "base_url": "http://127.0.0.1:8090/generate_stream",
        "model": "",   # 九天单模型部署可不填模型名
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


# ============================================================
# LangChain LLM 客户端
# ============================================================

class LLMClient:
    """
    基于 LangChain ChatOpenAI 的统一 LLM 客户端

    使用方式:
        client = LLMClient(provider="deepseek", api_key="sk-xxx")
        answer = client.chat("你好")
        answer = client.chat_with_system("你是助手", "用户问题")

    LangChain 组件:
        ChatOpenAI  → 模型调用 (OpenAI 兼容 API)
        ChatPromptTemplate → 模板化 Prompt (见 nl2sql_generator)
        JsonOutputParser  → 结构化 JSON 输出解析
    """

    def __init__(
        self,
        provider: str = "deepseek",
        api_key: str = "",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 60,
        max_retries: int = 2,
    ):
        defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["deepseek"])

        self.provider = provider
        self.model_name = model or defaults["model"]
        self.base_url = base_url or defaults["base_url"]
        _env_key = "JIUTIAN_APP_CODE" if provider == "jiutian" else f"{provider.upper()}_API_KEY"
        self.api_key = api_key or os.getenv(_env_key, "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # ---- LLM 后端实例 ----
        if provider == "jiutian":
            # 九天 generate_stream 推理服务 (AppCode Bearer 鉴权)
            self._chat = JiutianChatModel(
                base_url=self.base_url,
                app_code=self.api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=timeout,
            )
        else:
            self._chat: ChatOpenAI = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )

        logger.info(
            f"LangChain LLM 已初始化: provider={provider}, "
            f"model={self.model_name}, base_url={self.base_url}"
        )

    # ---- 简单对话 ----

    def chat(self, user_message: str, temperature: Optional[float] = None) -> str:
        """简单对话——发一条消息，返回文本"""
        return self._invoke([HumanMessage(content=user_message)], temperature)

    def chat_with_system(self, system_prompt: str, user_message: str,
                         temperature: Optional[float] = None) -> str:
        """带系统提示词的对话"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        return self._invoke(messages, temperature)

    # ---- 结构化 JSON 输出 ----

    def chat_json(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        对话并返回 JSON 对象

        使用手动 JSON 解析（比 JsonOutputParser 更兼容 DeepSeek）
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_message))

        raw = self._invoke(messages, temperature)
        return self._parse_json_fallback(raw)

    # ---- 便捷方法 ----

    def invoke(self, prompt: str, temperature: Optional[float] = None) -> str:
        """LangChain 风格: invoke(prompt) → str"""
        return self._invoke([HumanMessage(content=prompt)], temperature)

    def get_chat_model(self, temperature: Optional[float] = None):
        """获取 LLM 实例（用于 LangChain Chain 组合）"""
        if temperature is not None and temperature != self.temperature:
            if self.provider == "jiutian":
                return JiutianChatModel(
                    base_url=self.base_url,
                    app_code=self.api_key,
                    model_name=self.model_name,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
        return self._chat

    @property
    def chat_model(self) -> ChatOpenAI:
        """ChatOpenAI 实例（只读）"""
        return self._chat

    # ---- 内部方法 ----

    def _invoke(self, messages: List, temperature: Optional[float] = None) -> str:
        """调用 LangChain ChatOpenAI.invoke()"""
        try:
            if temperature is not None:
                chat = self.get_chat_model(temperature)
                response = chat.invoke(messages)
            else:
                response = self._chat.invoke(messages)

            content = response.content

            # 记录 Token 使用
            usage = getattr(response, 'response_metadata', {}).get('token_usage', {})
            if usage:
                logger.info(
                    f"LLM调用完成 | model={self.model_name} | "
                    f"input={usage.get('prompt_tokens')} | "
                    f"output={usage.get('completion_tokens')}"
                )

            return content

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

    @staticmethod
    def _parse_json_fallback(raw: str) -> dict:
        """JSON 解析回退方案"""
        import json
        import re

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"无法解析 LLM 返回为 JSON: {raw[:200]}...")
        return {}


# ============================================================
# 工厂函数
# ============================================================

def create_llm_client(
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    工厂函数: 创建 LLM 客户端

    使用:
        client = create_llm_client("deepseek", api_key="sk-xxx")
        client = create_llm_client("openai", api_key="sk-xxx", temperature=0.2)
    """
    return LLMClient(
        provider=provider,
        api_key=api_key or "",
        model=kwargs.get("model"),
        base_url=kwargs.get("base_url"),
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 4096),
        timeout=kwargs.get("timeout", 60),
        max_retries=kwargs.get("max_retries", 2),
    )
