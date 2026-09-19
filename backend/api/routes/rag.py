"""
知识库 RAG 接口
POST /api/rag/upload    —— 上传文档(PDF/Word/图片/文本)并建索引
POST /api/rag/answer    —— 基于已上传文档回答
GET  /api/rag/documents —— 列出已上传文档
"""
import os
import uuid
import shutil
import logging
from typing import List, Tuple, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.rag.document_parser import (
    ingest_files, TEXT_EXTS, PDF_EXTS, DOCX_EXTS, IMAGE_EXTS,
)
from backend.rag.retriever import HybridRetriever
from backend.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    "data", "uploads", "rag",
)

SUPPORTED_EXTS = TEXT_EXTS | PDF_EXTS | DOCX_EXTS | IMAGE_EXTS

_llm_client = None
_rag_retriever: Optional[HybridRetriever] = None
_documents: List[Tuple[str, str]] = []   # 已索引的 (id, text) 块
_uploaded_files: List[str] = []


def init_rag_services(llm_client):
    """由 main._init_services 注入共享 LLM 客户端"""
    global _llm_client
    _llm_client = llm_client


def _rebuild_retriever():
    global _rag_retriever
    if not _documents:
        _rag_retriever = None
        return
    _rag_retriever = HybridRetriever(
        [d[0] for d in _documents], [d[1] for d in _documents], use_semantic=True
    )


class RagAnswerRequest(BaseModel):
    question: str


@router.post("/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档 → 解析 → 分块 → 加入知识库索引"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    ext = os.path.splitext(file.filename)[1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {sorted(SUPPORTED_EXTS)}",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())[:12]
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file.filename)
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        docs = ingest_files([save_path])
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"文档解析失败: {e}")

    if not docs:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="未能从文件中提取到文本")

    _documents.extend(docs)
    _uploaded_files.append(file.filename)
    _rebuild_retriever()

    logger.info(f"知识库文档入库: {file.filename} → {len(docs)} 块 (累计 {len(_documents)})")
    return {
        "status": "ok",
        "filename": file.filename,
        "chunks": len(docs),
        "total_chunks": len(_documents),
    }


@router.post("/rag/answer")
async def rag_answer(req: RagAnswerRequest):
    """基于知识库检索并作答"""
    if _rag_retriever is None:
        raise HTTPException(status_code=400, detail="尚未上传文档，请先上传 PDF / Word / 图片 / 文本")
    if _llm_client is None:
        raise HTTPException(status_code=500, detail="LLM 未初始化")

    rag = RAGPipeline(store=_rag_retriever, llm=_llm_client, top_k=3)
    try:
        answer = rag.answer(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"作答失败: {e}")

    hits = _rag_retriever.search(req.question, top_k=3)
    return {
        "answer": answer or "未找到相关内容，请换一种问法。",
        "sources": [h[0] for h in hits],
    }


@router.get("/rag/documents")
async def list_documents():
    """列出已上传的知识库文档"""
    return {"documents": _uploaded_files, "chunk_count": len(_documents)}
