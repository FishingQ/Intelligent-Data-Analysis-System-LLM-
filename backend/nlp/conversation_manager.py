"""
对话管理器 (M2-6 / Phase 3)
多轮对话上下文管理，支持自动压缩和持久化。

用法:
    mgr = ConversationManager(max_turns=20)
    mgr.add_message("session_1", "user", "查询销售额")
    mgr.add_message("session_1", "assistant", "SELECT SUM(sales)...")
    context = mgr.get_context("session_1")  # 最近N轮对话
"""
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """单条消息"""
    role: str          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Conversation:
    """单个会话"""
    session_id: str
    title: str = "新对话"
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ConversationManager:
    """
    对话管理器

    特性:
    - 内存存储，多会话隔离
    - 每会话最多保留 max_turns 轮（1轮 = user + assistant）
    - 超长自动压缩：保留最早5轮 + 最近15轮，中间生成摘要
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._store: Dict[str, Conversation] = {}

    # ---- 会话管理 ----

    def create_session(self, session_id: str, title: str = "新对话") -> Conversation:
        """创建新会话"""
        conv = Conversation(
            session_id=session_id,
            title=title,
        )
        self._store[session_id] = conv
        logger.info(f"创建会话: {session_id}")
        return conv

    def get_session(self, session_id: str) -> Optional[Conversation]:
        """获取会话，不存在则自动创建"""
        if session_id not in self._store:
            self.create_session(session_id)
        return self._store[session_id]

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False

    def list_sessions(self) -> List[dict]:
        """列出所有会话（摘要信息）"""
        sessions = []
        for conv in self._store.values():
            sessions.append({
                "session_id": conv.session_id,
                "title": conv.title,
                "message_count": len(conv.messages),
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
            })
        # 按更新时间倒序
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    # ---- 消息管理 ----

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """添加一条消息到会话"""
        conv = self.get_session(session_id)
        msg = Message(role=role, content=content)
        conv.messages.append(msg)
        conv.updated_at = time.time()

        # 自动设置对话标题（取第一条用户消息前30字）
        if conv.title == "新对话" and role == "user":
            conv.title = content[:30] + ("..." if len(content) > 30 else "")

        # 超长压缩
        max_messages = self.max_turns * 2  # 每轮2条
        if len(conv.messages) > max_messages:
            self._compress(conv)

    def get_context(
        self,
        session_id: str,
        last_n: Optional[int] = None,
    ) -> List[dict]:
        """
        获取对话上下文（最近N条消息）

        Args:
            session_id: 会话ID
            last_n: 返回最近N条消息，默认返回全部（已压缩后）

        Returns:
            [{"role": "user", "content": "..."}, ...]
        """
        conv = self.get_session(session_id)
        messages = conv.messages

        if last_n and last_n < len(messages):
            messages = messages[-last_n:]

        return [{"role": m.role, "content": m.content} for m in messages]

    def get_context_text(
        self,
        session_id: str,
        last_n: int = 8,
    ) -> str:
        """
        获取格式化的对话上下文字符串（供 LLM prompt 使用）

        Returns:
            "用户: ...\n助手: ...\n用户: ..."
        """
        context = self.get_context(session_id, last_n=last_n)
        if not context:
            return ""

        lines = ["--- 对话历史 ---"]
        for m in context:
            role_cn = "用户" if m["role"] == "user" else "助手"
            lines.append(f"{role_cn}: {m['content']}")
        lines.append("--- 当前问题 ---")

        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """清空会话消息（保留会话本身）"""
        conv = self.get_session(session_id)
        conv.messages.clear()
        conv.title = "新对话"
        conv.updated_at = time.time()
        logger.info(f"清空会话: {session_id}")

    # ---- 内部方法 ----

    def _compress(self, conv: Conversation) -> None:
        """
        压缩超长对话：
        保留最早5条 + 最近(max_turns*2-5)条，中间生成1条摘要
        """
        keep_head = 5
        keep_tail = self.max_turns * 2 - keep_head - 1  # 减1给摘要

        if len(conv.messages) <= keep_head + keep_tail:
            return

        head = conv.messages[:keep_head]
        tail = conv.messages[-keep_tail:]

        # 生成简单摘要
        middle_texts = [
            m.content[:50] for m in conv.messages[keep_head:-keep_tail]
            if m.role == "user"
        ]
        summary = "（此前讨论: " + "；".join(middle_texts[:5]) + "）"

        conv.messages = head + [
            Message(role="system", content=summary)
        ] + tail

        logger.info(f"压缩会话 {conv.session_id}: "
                     f"{len(conv.messages)}条 → {keep_head + 1 + keep_tail}条")


# 全局单例
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """获取全局单例"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager(max_turns=20)
    return _conversation_manager
