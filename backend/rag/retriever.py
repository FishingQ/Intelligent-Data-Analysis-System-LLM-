"""
混合检索器：BM25（关键词）+ 本地语义向量（fastembed）

- BM25 手动实现，零依赖，负责精确关键词/实体匹配。
- 语义向量用 fastembed（ONNX，无 torch）+ 多语言模型，负责中文↔英文跨语言匹配。
- 语义向量不可用（未安装/下载失败）时自动回退纯 BM25。
"""
import os
import re
import logging
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 多语言 embedding 模型（中英混合，体积小）
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# 国内镜像（huggingface.co 不可达时走这里）
HF_ENDPOINT = "https://hf-mirror.com"


# ============================================================
# BM25 关键词检索
# ============================================================

class BM25:
    """BM25 排名（手动实现，零依赖）"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """英文/数字按词切分，中文按单字切分"""
        toks = []
        for seg in re.split(r"[^A-Za-z0-9一-鿿]+", text.lower()):
            if not seg:
                continue
            if re.search(r"[A-Za-z0-9]", seg):
                toks.append(seg)
            else:
                toks.extend(seg)  # 纯中文 → 单字
        return toks

    def fit(self, corpus: List[str]) -> "BM25":
        self.corpus = corpus
        self.n = len(corpus)
        self.tfs = [Counter(self._tokenize(d)) for d in corpus]
        self.doc_len = np.array([sum(c.values()) for c in self.tfs])
        self.avgdl = self.doc_len.mean() if self.n else 0.0
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.df = df
        self.idf = {
            t: np.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }
        self._fitted = True
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self._fitted or self.n == 0:
            return []
        scores = np.zeros(self.n)
        for t in self._tokenize(query):
            if t not in self.df:
                continue
            idf = self.idf[t]
            for i in range(self.n):
                tf = self.tfs[i].get(t, 0)
                if tf == 0:
                    continue
                dl = self.doc_len[i]
                scores[i] += (
                    idf
                    * (tf * (self.k1 + 1))
                    / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                )
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


# ============================================================
# 本地语义向量
# ============================================================

class SemanticEmbedder:
    """fastembed 本地语义向量（ONNX，无 torch）"""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
            # 关闭 Xet 存储后端：hf-mirror 不支持其 CAS 重建，强制走 LFS 直连下载
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
            return True
        except Exception as e:
            logger.warning(f"语义向量不可用，回退 BM25: {e}")
            return False

    def encode(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self._ensure_model():
            return None
        try:
            return np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        except Exception as e:
            logger.warning(f"语义向量编码失败: {e}")
            return None


# ============================================================
# 混合检索器
# ============================================================

class HybridRetriever:
    """BM25 + 语义向量 融合检索，接口与 TFIDFVectorStore.search 一致"""

    def __init__(self, ids: List[str], texts: List[str], use_semantic: bool = True):
        self.ids = list(ids)
        self.texts = list(texts)
        self.bm25 = BM25().fit(self.texts)
        self.embedder = SemanticEmbedder() if use_semantic else None
        self._matrix: Optional[np.ndarray] = None

    def _build_semantic(self) -> None:
        if self.embedder is None or self._matrix is not None:
            return
        vecs = self.embedder.encode(self.texts)
        if vecs is None:
            return
        self._matrix = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        logger.info(f"语义向量构建完成: {self._matrix.shape}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        bm_hits = self.bm25.search(query, top_k=top_k)

        if self._matrix is None and self.embedder is not None:
            self._build_semantic()

        if self._matrix is None:
            return [(self.ids[i], self.texts[i], s) for i, s in bm_hits]

        qv = self.embedder.encode([query])
        if qv is None:
            return [(self.ids[i], self.texts[i], s) for i, s in bm_hits]

        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        sem = self._matrix @ qv[0]

        bm = np.zeros(len(self.ids))
        for i, s in bm_hits:
            bm[i] = s
        bm = bm / (bm.max() + 1e-9)
        sem = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
        combined = 0.4 * bm + 0.6 * sem

        idx = np.argsort(combined)[::-1][:top_k]
        return [(self.ids[i], self.texts[i], float(combined[i])) for i in idx]

    def save(self, dirpath: str) -> str:
        os.makedirs(dirpath, exist_ok=True)
        self._build_semantic()  # 确保矩阵已构建再持久化（内部有 embedder/已建 双重守卫）
        import joblib
        path = os.path.join(dirpath, "hybrid_index.joblib")
        joblib.dump(
            {"ids": self.ids, "texts": self.texts, "matrix": self._matrix}, path
        )
        return path

    @classmethod
    def load(cls, path: str) -> "HybridRetriever":
        import joblib
        d = joblib.load(path)
        r = cls(d["ids"], d["texts"], use_semantic=d.get("matrix") is not None)
        r._matrix = d.get("matrix")
        return r


if __name__ == "__main__":
    # 自检：BM25 关键词召回正确（不加载模型，快速）
    logging.basicConfig(level=logging.INFO)
    docs = [
        ("a", "2024年5月 咖啡品类 Latte 现金支付 销量 38.7"),
        ("b", "CCUS 项目 storage hub 钢铁行业 捕集能力 2 Mt"),
        ("c", "电动汽车 BEV 保有量 2015-2020 年均增长率"),
    ]
    bm = BM25().fit([d[1] for d in docs])
    hits = bm.search("Latte 现金支付 销量", top_k=1)
    assert hits and hits[0][0] == 0, hits
    print("[OK] BM25 自检通过:", docs[hits[0][0]][1][:24], round(hits[0][1], 4))
