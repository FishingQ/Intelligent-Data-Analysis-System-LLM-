"""
RAG 问答流水线：向量召回 top-k 段落 → LLM 生成答案
"""
import os
import logging
from typing import List, Tuple, Optional

from backend.rag.vector_store import TFIDFVectorStore
from backend.rag.retriever import HybridRetriever
from backend.training.dataset_loader import DatasetLoader
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "rag_index")


def parse_markdown_chunks(filepath: str) -> List[Tuple[str, str]]:
    """解析 === id === 分隔的 markdown 段落，返回 [(id, text), ...]"""
    chunks = DatasetLoader._parse_markdown_contexts(filepath)
    return list(chunks.items())


def load_or_build_store(
    md_paths: List[str],
    index_dir: str = DEFAULT_INDEX_DIR,
    force: bool = False,
) -> TFIDFVectorStore:
    """加载缓存向量索引；不存在则从 md 文件构建并缓存"""
    index_path = os.path.join(index_dir, "tfidf_index.joblib")
    if not force and os.path.exists(index_path):
        logger.info(f"加载缓存向量索引: {index_path}")
        return TFIDFVectorStore.load(index_path)

    documents: List[Tuple[str, str]] = []
    for path in md_paths:
        if not os.path.exists(path):
            logger.warning(f"数据文件不存在，跳过: {path}")
            continue
        documents.extend(parse_markdown_chunks(path))

    if not documents:
        raise FileNotFoundError(f"未找到任何数据段落: {md_paths}")

    store = TFIDFVectorStore().index(documents)
    store.save(index_dir)
    return store


def load_or_build_retriever(
    md_paths: List[str],
    index_dir: str = DEFAULT_INDEX_DIR,
    force: bool = False,
) -> HybridRetriever:
    """加载缓存混合检索索引（BM25+语义）；不存在则从 md 文件构建并缓存"""
    index_path = os.path.join(index_dir, "hybrid_index.joblib")
    if not force and os.path.exists(index_path):
        logger.info(f"加载缓存混合检索索引: {index_path}")
        return HybridRetriever.load(index_path)

    documents: List[Tuple[str, str]] = []
    for path in md_paths:
        if not os.path.exists(path):
            logger.warning(f"数据文件不存在，跳过: {path}")
            continue
        documents.extend(parse_markdown_chunks(path))

    if not documents:
        raise FileNotFoundError(f"未找到任何数据段落: {md_paths}")

    retriever = HybridRetriever(
        [d[0] for d in documents], [d[1] for d in documents], use_semantic=True
    )
    retriever.save(index_dir)
    return retriever


class RAGPipeline:
    """非结构化数据 RAG：检索 + 作答"""

    def __init__(
        self,
        store: TFIDFVectorStore,
        llm: Optional[LLMClient] = None,
        top_k: int = 3,
        max_chunk_chars: int = 8000,
    ):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        # ponytail: 整块截断到 max_chunk_chars，超长表格会丢尾部；
        # 若答案依赖尾部，改成分段/表级检索。
        self.max_chunk_chars = max_chunk_chars

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Tuple[str, str, float]]:
        return self.store.search(question, top_k or self.top_k)

    def answer(self, question: str, top_k: Optional[int] = None) -> str:
        hits = self.retrieve(question, top_k)
        if not hits:
            return ""

        context = "\n\n".join(
            f"[数据{i + 1}]\n{text[: self.max_chunk_chars]}"
            for i, (_, text, _) in enumerate(hits)
        )
        prompt = (
            "你是一名数据分析助手，请根据下面给出的数据回答用户问题。\n"
            "数据是 Markdown 表格。请仔细阅读并计算，直接给出最终答案，"
            "不要复述数据、不要解释过程。\n\n"
            f"{context}\n\n"
            f"用户问题：{question}\n\n答案："
        )
        if self.llm is None:
            raise RuntimeError("RAGPipeline 需要 LLMClient 才能生成答案")
        return self.llm.chat(prompt)
