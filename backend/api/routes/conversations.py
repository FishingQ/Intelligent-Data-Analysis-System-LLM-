"""
对话历史管理接口
GET /api/conversations — 列出所有会话
GET /api/conversations/{id} — 获取单个会话详情
DELETE /api/conversations/{id} — 删除会话
"""
import logging
from fastapi import APIRouter, HTTPException

from backend.nlp.conversation_manager import get_conversation_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/conversations")
async def list_conversations():
    """列出所有会话（摘要信息）"""
    mgr = get_conversation_manager()
    sessions = mgr.list_sessions()
    return {
        "count": len(sessions),
        "sessions": sessions,
    }


@router.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    """获取单个会话的完整消息历史"""
    mgr = get_conversation_manager()
    conv = mgr.get_session(session_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")

    return {
        "session_id": conv.session_id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
            }
            for m in conv.messages
        ],
    }


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """删除指定会话"""
    mgr = get_conversation_manager()
    ok = mgr.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或已删除")

    logger.info(f"已删除会话: {session_id}")
    return {"status": "deleted", "session_id": session_id}


@router.delete("/conversations")
async def clear_all_conversations():
    """清空所有会话"""
    mgr = get_conversation_manager()
    for sid in list(mgr._store.keys()):
        mgr.delete_session(sid)
    return {"status": "cleared"}
