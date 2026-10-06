"""
真实稠密向量检索轨 (Vector Retrieval Engine)
接入 AstrBot 上下文中已配置的 Embedding Provider，
计算语义向量并执行余弦相似度检索，彻底终结伪双轨自导自演。
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from astrbot.api import logger

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

class VectorRetrievalEngine:
    def __init__(self, context: Any = None, preferred_provider_id: Optional[str] = None):
        self.context = context
        self.preferred_provider_id = preferred_provider_id
        self.doc_ids: List[int] = []
        self.vectors: List[List[float]] = []
        self._provider = None
        self._dim: int = 0

    def set_context(self, context: Any, preferred_provider_id: Optional[str] = None):
        self.context = context
        if preferred_provider_id:
            self.preferred_provider_id = preferred_provider_id
        self._provider = None

    def get_embedding_provider(self):
        """从 AstrBot Context 解析激活的 Embedding Provider"""
        if self._provider is not None:
            return self._provider

        if not self.context:
            return None

        providers = []
        try:
            if hasattr(self.context, "get_all_embedding_providers"):
                providers = self.context.get_all_embedding_providers()
            elif hasattr(self.context, "provider_manager") and hasattr(self.context.provider_manager, "embedding_provider_insts"):
                providers = self.context.provider_manager.embedding_provider_insts
        except Exception as e:
            logger.warning(f"[rag_pro.vector] 获取 embedding provider 失败: {e}")

        if not providers:
            return None

        # 1. 优先使用用户配置的 preferred_provider_id
        if self.preferred_provider_id:
            for p in providers:
                p_id = getattr(p, "id", None) or getattr(getattr(p, "config", None), "id", "")
                if p_id == self.preferred_provider_id:
                    self._provider = p
                    return p

        # 2. 默认选取第一个可用的 embedding provider
        self._provider = providers[0]
        return self._provider

    def is_available(self) -> bool:
        """检查是否有可用的 Embedding Provider"""
        return self.get_embedding_provider() is not None

    def clear(self):
        self.doc_ids = []
        self.vectors = []
        self._dim = 0

    def add_vectors(self, doc_ids: List[int], vectors: List[List[float]]):
        """批量注册文档 ID 与对应向量"""
        for doc_id, vec in zip(doc_ids, vectors):
            if vec and any(vec):
                self.doc_ids.append(doc_id)
                self.vectors.append(vec)
                if not self._dim:
                    self._dim = len(vec)

    async def compute_embeddings(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """调用 Provider 计算文本嵌入向量"""
        provider = self.get_embedding_provider()
        if not provider:
            raise RuntimeError("没有找到可用的 Embedding Provider，请在 AstrBot 服务提供商中配置 Embedding 模型")

        if not texts:
            return []

        # 优先调用批量接口
        try:
            if hasattr(provider, "get_embeddings_batch"):
                return await provider.get_embeddings_batch(texts, batch_size=batch_size)
            elif hasattr(provider, "get_embeddings"):
                return await provider.get_embeddings(texts)
            elif hasattr(provider, "get_embedding"):
                res = []
                for t in texts:
                    res.append(await provider.get_embedding(t))
                return res
            else:
                raise AttributeError("Provider 不支持 get_embedding/get_embeddings 方法")
        except Exception as e:
            logger.error(f"[rag_pro.vector] 计算 embedding 异常: {e}")
            raise

    async def compute_query_embedding(self, query: str) -> List[float]:
        """计算单条查询 Query 的 Embedding"""
        provider = self.get_embedding_provider()
        if not provider:
            return []
        try:
            return await provider.get_embedding(query)
        except Exception as e:
            logger.error(f"[rag_pro.vector] 计算 query embedding 失败: {e}")
            return []

    def search_by_vector(self, query_vector: List[float], top_k: int = 20) -> List[Tuple[int, float]]:
        """
        基于余弦相似度检索相关切片。
        返回 [(doc_id, cosine_similarity), ...]
        """
        if not query_vector or not self.vectors:
            return []

        if HAVE_NUMPY:
            try:
                q = np.array(query_vector, dtype=np.float32)
                q_norm = np.linalg.norm(q)
                if q_norm == 0:
                    return []
                q = q / q_norm

                matrix = np.array(self.vectors, dtype=np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12
                normed_matrix = matrix / norms

                sims = np.dot(normed_matrix, q)
                # 获取排序索引
                indices = np.argsort(-sims)[:top_k]
                results = []
                for idx in indices:
                    results.append((self.doc_ids[idx], float(sims[idx])))
                return results
            except Exception as e:
                logger.debug(f"[rag_pro.vector] numpy 矩阵加速失败，回退纯 Python: {e}")

        # 纯 Python 降级余弦相似度计算
        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0:
            return []

        scores = []
        for doc_id, vec in zip(self.doc_ids, self.vectors):
            dot = sum(a * b for a, b in zip(query_vector, vec))
            v_norm = math.sqrt(sum(x * x for x in vec))
            if v_norm == 0:
                sim = 0.0
            else:
                sim = dot / (q_norm * v_norm)
            scores.append((doc_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """全流程稠密向量检索：文本 -> Embedding -> 余弦相似度召回"""
        q_vec = await self.compute_query_embedding(query)
        if not q_vec:
            return []
        return self.search_by_vector(q_vec, top_k=top_k)
