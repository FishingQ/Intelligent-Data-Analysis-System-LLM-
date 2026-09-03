"""
九天平台大模型适配器 —— 实现 LangChain BaseChatModel 接口

九天推理服务 (应用接入 + AppCode 鉴权):
    POST http://127.0.0.1:8090/generate_stream
    Header: Authorization: Bearer <AppCode>
    Body:   {"model": "...", "messages": [{"role": "user", "content": "..."}], ...}

    响应为流式 (SSE) 或普通 JSON，两种都兼容解析。

实现 BaseChatModel 后，项目里现有的 LangChain chain
(prompt | chat_model | StrOutputParser) 无需任何改动即可切换后端。
"""
import json
import logging
from typing import Any, List, Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

logger = logging.getLogger(__name__)


class JiutianChatModel(BaseChatModel):
    """九天 generate_stream 推理服务的 LangChain 适配器"""

    base_url: str = "http://127.0.0.1:8090/generate_stream"
    app_code: str = ""          # 九天平台应用 AppCode (Bearer 鉴权)
    model_name: str = ""        # 单模型部署可留空
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60

    @property
    def _llm_type(self) -> str:
        return "jiutian"

    # ---- 核心: LangChain 调用入口 ----

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict = {
            "messages": [
                {"role": self._role_name(m), "content": self._content(m)}
                for m in messages
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model_name:
            payload["model"] = self.model_name

        headers = {
            "Authorization": f"Bearer {self.app_code}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"九天接口鉴权/请求失败: {e.response.status_code} {e.response.text[:200]}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"九天接口连接失败 (服务未启动?): {e}")
            raise

        text = self._extract_text(resp)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    # ---- 响应解析: 兼容流式 SSE 与普通 JSON ----

    @staticmethod
    def _extract_text(resp: httpx.Response) -> str:
        ctype = resp.headers.get("content-type", "")
        if "event-stream" in ctype or "stream" in ctype:
            return JiutianChatModel._parse_sse(resp.text)
        try:
            return JiutianChatModel._extract_content(resp.json())
        except Exception:
            return resp.text

    @staticmethod
    def _parse_sse(text: str) -> str:
        """解析 SSE 流: 每行 data: {...}，OpenAI chunk 格式逐段拼接"""
        parts: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = JiutianChatModel._extract_content(obj, delta=True)
            if piece:
                parts.append(piece)
        return "".join(parts)

    @staticmethod
    def _extract_content(obj: Any, delta: bool = False) -> str:
        """从 OpenAI 风格响应中提取文本，兼容 chunk(delta)/完整(message) 及兜底字段"""
        if isinstance(obj, str):
            return obj
        if not isinstance(obj, dict):
            return ""
        choices = obj.get("choices") or []
        if choices:
            first = choices[0] or {}
            if delta:
                d = first.get("delta") or {}
                if d.get("content"):
                    return d["content"]
            msg = first.get("message") or {}
            if msg.get("content"):
                return msg["content"]
            if first.get("text"):
                return first["text"]
        for key in ("content", "generated_text", "text"):
            if obj.get(key):
                return obj[key]
        return ""

    # ---- 消息转换 ----

    @staticmethod
    def _role_name(m: BaseMessage) -> str:
        t = getattr(m, "type", "human")
        if t == "human":
            return "user"
        if t == "ai":
            return "assistant"
        return t  # system / assistant / ...

    @staticmethod
    def _content(m: BaseMessage) -> str:
        c = m.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return str(c)


# ---- 自检: 解析逻辑离线验证 (不依赖真实服务) ----

def _self_check() -> None:
    import httpx as _httpx

    # 非流式 JSON
    r1 = _httpx.Response(200, json={"choices": [{"message": {"content": "你好"}}]})
    assert JiutianChatModel._extract_text(r1) == "你好", "非流式解析失败"

    # 流式 SSE (OpenAI chunk)
    sse = 'data: {"choices":[{"delta":{"content":"你"}}]}\n\ndata: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]\n'
    r2 = _httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse)
    assert JiutianChatModel._extract_text(r2) == "你好", "SSE 解析失败"

    # 兜底字段
    r3 = _httpx.Response(200, json={"generated_text": "ok"})
    assert JiutianChatModel._extract_text(r3) == "ok", "兜底字段解析失败"

    print("JiutianChatModel 解析自检通过")


if __name__ == "__main__":
    _self_check()
