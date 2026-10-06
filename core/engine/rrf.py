"""
双轨倒数排名融合算法 (Reciprocal Rank Fusion, RRF)
消除 BM25 绝对得分与向量余弦相似度之间的量纲差异，
支持任意单轨退化、双轨加权融合及 Top-K 综合重排。
"""

from typing import List, Dict, Any, Tuple

class HybridRRF:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        bm25_ranked_indices: List[int],
        vector_ranked_indices: List[int],
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        融合两个独立检索轨的排名列表。
        当某一轨为空时（例如未配置 Embedding Provider），自动以另一轨为主并安全退化。
        """
        # 若两轨都为空，直接返回空
        if not bm25_ranked_indices and not vector_ranked_indices:
            return []

        # 若单轨为空，动态调增另一轨权重
        if not vector_ranked_indices:
            vector_weight = 0.0
            bm25_weight = 1.0
        elif not bm25_ranked_indices:
            bm25_weight = 0.0
            vector_weight = 1.0

        rrf_scores: Dict[int, float] = {}
        rank_records: Dict[int, Dict[str, Any]] = {}

        if bm25_weight > 0:
            for rank, doc_id in enumerate(bm25_ranked_indices):
                score = bm25_weight / (self.k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
                rank_records.setdefault(doc_id, {})["bm25_rank"] = rank + 1

        if vector_weight > 0:
            for rank, doc_id in enumerate(vector_ranked_indices):
                score = vector_weight / (self.k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
                rank_records.setdefault(doc_id, {})["vector_rank"] = rank + 1

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        max_possible = (bm25_weight + vector_weight) / (self.k + 1)

        final_output = []
        for doc_id, raw_score in sorted_results:
            normalized = min(1.0, raw_score / max_possible if max_possible > 0 else 0.0)
            final_output.append({
                "doc_id": doc_id,
                "score": round(normalized, 4),
                "raw_rrf": raw_score,
                "ranks": rank_records.get(doc_id, {})
            })

        return final_output
