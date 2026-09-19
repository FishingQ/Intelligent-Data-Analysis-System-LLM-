"""
焕新社区 (aihuanxin.cn) LLM 客户端

平台推理服务为自定义接口（非 OpenAI 兼容），调用方式：
  - 外部 ingress: POST {ingress_url}（Authorization: Bearer ${AppCode}）
  - 内部/容器:    POST {host}:8090/generate（无需鉴权）

鉴权：Authorization: Bearer ${AppCode}

请求体格式（generate 与 generate_stream 相同）:
  {"inputs": "Human:\\n...\\nAssistant:\\n", "parameters": {"do_sample": false, "max_new_tokens": 2048}}

本模块实现：
  - JiutianChatModel : LangChain BaseChatModel 适配，让现有 `prompt | chat_model | StrOutputParser()` 链无需改动
  - JiutianClient    : 与 LLMClient 同接口（chat / chat_with_system / chat_json / invoke / chat_model）
"""
import json
import re
import logging
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


def extract_generated_text(data: Any) -> str:
    """从九天接口响应中提取生成文本（容错多种返回结构）"""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        if not data:
            return ""
        return extract_generated_text(data[0])
    if isinstance(data, dict):
        for key in ("generated_text", "outputs", "output", "response", "answer", "text"):
            if isinstance(data.get(key), str):
                return data[key]
        for sub in ("data", "result", "results", "detail"):
            if isinstance(data.get(sub), (dict, list)):
                txt = extract_generated_text(data[sub])
                if txt:
                    return txt
    return ""


class JiutianChatModel(BaseChatModel):
    """九天 generate 接口的 LangChain ChatModel 适配"""

    app_code: str = ""
    generate_url: str = ""
    max_new_tokens: int = 2048
    temperature: float = 0.1
    timeout: int = 120

    @property
    def _llm_type(self) -> str:
        return "jiutian-generate"

    @staticmethod
    def _format_inputs(messages: List[BaseMessage]) -> str:
        parts: List[str] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                parts.append(str(m.content))
            elif isinstance(m, HumanMessage):
                parts.append(f"Human:\n{m.content}\n")
            elif isinstance(m, AIMessage):
                parts.append(f"Assistant:\n{m.content}\n")
            else:
                parts.append(f"{m.content}\n")
        return "\n".join(parts) + "\nAssistant:\n"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.app_code:
            raise ValueError(
                "焕新社区 AppCode 未配置（Authorization 为空），无法调用模型。"
                "请在平台「模型推理 → 应用接入」获取 AppCode 后设置环境变量：\n"
                "  Windows:  setx JIUTIAN_APP_CODE \"你的AppCode\"（设置后重启终端与服务）"
            )
        inputs_text = self._format_inputs(messages)
        payload = json.dumps({
            "inputs": inputs_text,
            "parameters": {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
            },
        })
        try:
            resp = requests.post(
                self.generate_url,
                data=payload,
                headers={
                    "content-type": "application/json",
                    "Authorization": f"Bearer {self.app_code}",
                },
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise RuntimeError(
                f"请求焕新社区接口失败（请检查 base_url 与网络/代理，本地调用需 unset http_proxy https_proxy）: {e}"
            ) from e
        if resp.status_code != 200:
            raise RuntimeError(
                f"焕新社区接口返回 HTTP {resp.status_code}（AppCode 或 URL 可能不正确）: {resp.text[:300]}"
            )
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"焕新社区接口返回非 JSON 响应: {resp.text[:300]}") from None
        text = extract_generated_text(data).strip()
        if not text:
            raise RuntimeError(f"焕新社区接口返回空文本，原始响应: {resp.text[:300]}")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class JiutianClient:
    """与 LLMClient 同接口的九天平台客户端"""

    def __init__(
        self,
        app_code: str = "",
        base_url: str = "",
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 2,
    ):
        self.provider = "jiutian"
        self.app_code = app_code
        self.base_url = (base_url or "").rstrip("/")
        self.generate_url = f"{self.base_url}/generate"  # 外部 ingress 非流式端点：{service-id}/generate
        self.model_name = model or "jiutian-large"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._chat = JiutianChatModel(
            app_code=app_code,
            generate_url=self.generate_url,
            max_new_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        logger.info(
            f"九天 LLM 已初始化: model={self.model_name}, url={self.generate_url}, "
            f"app_code={'***' if app_code else '(未配置)'}"
        )

    @property
    def chat_model(self) -> BaseChatModel:
        return self._chat

    def _invoke(self, messages: List[BaseMessage], temperature: Optional[float] = None) -> str:
        model = self._chat
        if temperature is not None and temperature != self.temperature:
            model = JiutianChatModel(
                app_code=self.app_code,
                generate_url=self.generate_url,
                max_new_tokens=self.max_tokens,
                temperature=temperature,
                timeout=self.timeout,
            )
        return model.invoke(messages).content

    def chat(self, user_message: str, temperature: Optional[float] = None) -> str:
        return self._invoke([HumanMessage(content=user_message)], temperature)

    def chat_with_system(self, system_prompt: str, user_message: str,
                         temperature: Optional[float] = None) -> str:
        return self._invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)],
            temperature,
        )

    def invoke(self, prompt: str, temperature: Optional[float] = None) -> str:
        return self._invoke([HumanMessage(content=prompt)], temperature)

    def chat_json(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_message))
        raw = self._invoke(messages, temperature)
        return self._parse_json_fallback(raw)

    @staticmethod
    def _parse_json_fallback(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"无法解析 LLM 返回为 JSON: {raw[:200]}...")
        return {}
