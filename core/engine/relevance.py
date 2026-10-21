"""
查询相关性判定

RRF 只融合排名，丢弃了「这个查询到底像不像」的绝对信息。
后果是：任何查询只要命中一个常见词，候选切片就会进入融合，
并被归一化成看起来很高的分数，最终把无关内容注入提示词。

本模块提供一个与 RRF 互补的判定：按 IDF 加权的词项覆盖率。
关键点是**语料里不存在的词按不覆盖且高权重处理**——查询里出现
一个知识库根本不认识的词，正是离题最强的信号。
"""

from typing import Dict, Iterable, Set

from .bm25 import BM25Okapi


def query_coverage(query: str, corpus_tokens: Iterable[Set[str]], idf: Dict[str, float]) -> float:
    """查询词项被语料覆盖的 IDF 加权比例（0~1）。

    query          : 用户提问
    corpus_tokens  : 每个候选切片已分好的词项集合
    idf            : BM25 的 IDF 表；不在表里的词视为未登录词

    返回 0 表示完全对不上，1 表示每个词都在语料中出现。
    """
    q_tokens = set(BM25Okapi.tokenize(query))
    if not q_tokens:
        return 0.0

    covered: Set[str] = set()
    for toks in corpus_tokens:
        covered.update(q_tokens & toks)

    # 未登录词给一个高于所有已知词的权重，确保它主导分母
    max_known_idf = max(idf.values()) if idf else 1.0
    unknown_weight = max_known_idf * 1.5

    total = 0.0
    hit = 0.0
    for tok in q_tokens:
        if tok in idf:
            w = max(idf[tok], 1e-6)
        else:
            w = unknown_weight
        total += w
        if tok in covered:
            hit += w

    return hit / total if total > 0 else 0.0
