"""
TF-IDF 向量存储与相似度检索

将非结构化 Markdown 段落转为 TF-IDF 稀疏向量，用余弦相似度召回。
基于 scikit-learn，零新增依赖，离线可用。持久化用 joblib（sklearn 自带）。
"""
import os
import logging
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TFIDFVectorStore:
    """轻量 TF-IDF 向量库：索引 → 持久化 → 余弦检索"""

    def __init__(self, max_features: int = 200_000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None          # 稀疏 TF-IDF 矩阵 (n_docs, n_features)
        self.ids: List[str] = []
        self.texts: List[str] = []

    def index(self, documents: List[Tuple[str, str]]) -> "TFIDFVectorStore":
        """documents: [(id, text), ...]"""
        self.ids = [d[0] for d in documents]
        self.texts = [d[1] for d in documents]
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)
        logger.info(
            f"TF-IDF 索引完成: {len(self.ids)} 段落, 词表 {len(self.vectorizer.vocabulary_)}"
        )
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """返回 [(id, text, score), ...]，按相似度降序"""
        if self.matrix is None or self.vectorizer is None or self.matrix.shape[0] == 0:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        top_k = min(top_k, len(self.ids))
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.ids[i], self.texts[i], float(sims[i])) for i in top_idx]

    def save(self, dirpath: str) -> str:
        """持久化到目录，返回索引文件路径"""
        os.makedirs(dirpath, exist_ok=True)
        import joblib
        path = os.path.join(dirpath, "tfidf_index.joblib")
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "ids": self.ids,
                "texts": self.texts,
            },
            path,
        )
        logger.info(f"向量索引已保存: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "TFIDFVectorStore":
        import joblib
        data = joblib.load(path)
        store = cls()
        store.vectorizer = data["vectorizer"]
        store.matrix = data["matrix"]
        store.ids = data["ids"]
        store.texts = data["texts"]
        return store


if __name__ == "__main__":
    # 自检：构建小索引，验证关键词召回正确
    logging.basicConfig(level=logging.INFO)
    docs = [
        ("a", "2024年5月 咖啡品类 Latte 现金支付 销量 38.7"),
        ("b", "CCUS 项目 storage hub 钢铁行业 捕集能力 2 Mt"),
        ("c", "电动汽车 BEV 保有量 2015-2020 年均增长率"),
    ]
    store = TFIDFVectorStore().index(docs)
    hits = store.search("Latte 现金支付 销量", top_k=1)
    assert hits and hits[0][0] == "a", hits
    print("[OK] 自检通过:", hits[0][0], hits[0][2])
