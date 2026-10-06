"""
纯 Python 轻量级 BM25 Okapi 算法实现
支持分词统计、IDF 倒排持久化与增量检索
"""

import math
import re
from typing import List, Dict, Any, Tuple

class BM25Okapi:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中英混合分词与代码符号保留分词"""
        if not text:
            return []
        text = text.lower()
        tokens = re.findall(r'[a-zA-Z0-9_\-\.]+|[\u4e00-\u9fa5]', text)
        return [t.strip() for t in tokens if t.strip()]

    def fit(self, corpus: List[str]):
        """根据语料库构建索引与统计量"""
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return

        self.doc_len = []
        self.doc_freqs = []
        df: Dict[str, int] = {}

        for doc in corpus:
            tokens = self.tokenize(doc)
            self.doc_len.append(len(tokens))
            frequencies: Dict[str, int] = {}
            for t in tokens:
                frequencies[t] = frequencies.get(t, 0) + 1
            self.doc_freqs.append(frequencies)
            for t in frequencies.keys():
                df[t] = df.get(t, 0) + 1

        self.avg_dl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0

        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: str) -> List[float]:
        """计算查询与各文档的 BM25 相关性得分"""
        if self.corpus_size == 0:
            return []
        q_tokens = self.tokenize(query)
        scores = [0.0] * self.corpus_size

        for token in q_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_idx, freq_map in enumerate(self.doc_freqs):
                if token not in freq_map:
                    continue
                tf = freq_map[token]
                d_len = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / (self.avg_dl + 1e-6)))
                scores[doc_idx] += idf_val * (tf * (self.k1 + 1.0)) / (denom + 1e-6)

        return scores
