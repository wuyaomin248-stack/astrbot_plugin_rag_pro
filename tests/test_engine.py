"""
单元测试：BM25 分词、RRF 融合、相关性门槛

不依赖 AstrBot 宿主，直接 python -m unittest 运行：
    python -m unittest discover -s tests -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from core.engine.bm25 import BM25Okapi
from core.engine.relevance import query_coverage
from core.engine.rrf import HybridRRF


class TestTokenizer(unittest.TestCase):
    def test_english_identifier_is_one_token(self):
        """精确标识符不能被切碎，这是稀疏轨存在的意义。"""
        toks = BM25Okapi.tokenize("get_embeddings_batch")
        self.assertEqual(toks, ["get_embeddings_batch"])

    def test_error_code_is_one_token(self):
        self.assertEqual(BM25Okapi.tokenize("error_code_401"), ["error_code_401"])

    def test_chinese_uses_bigrams(self):
        """中文必须切二元组；逐字切分会丢失词内搭配信息。"""
        toks = BM25Okapi.tokenize("安装插件")
        self.assertEqual(toks, ["安装", "装插", "插件"])

    def test_single_chinese_char(self):
        self.assertEqual(BM25Okapi.tokenize("好"), ["好"])

    def test_mixed(self):
        toks = BM25Okapi.tokenize("配置 embedding_provider_id 参数")
        self.assertIn("embedding_provider_id", toks)
        self.assertIn("配置", toks)

    def test_empty(self):
        self.assertEqual(BM25Okapi.tokenize(""), [])


class TestBM25(unittest.TestCase):
    def test_exact_identifier_ranks_first(self):
        corpus = [
            "how to configure the embedding provider",
            "call get_embeddings_batch to embed documents",
            "unrelated text about bread and ovens",
        ]
        bm = BM25Okapi()
        bm.fit(corpus)
        scores = bm.get_scores("get_embeddings_batch")
        self.assertEqual(max(range(len(scores)), key=lambda i: scores[i]), 1)

    def test_chinese_query_hits_chinese_doc(self):
        corpus = ["安装插件的方法", "配置向量模型", "无关内容"]
        bm = BM25Okapi()
        bm.fit(corpus)
        scores = bm.get_scores("如何安装插件")
        self.assertEqual(max(range(len(scores)), key=lambda i: scores[i]), 0)

    def test_empty_corpus(self):
        bm = BM25Okapi()
        bm.fit([])
        self.assertEqual(bm.get_scores("anything"), [])


class TestRRF(unittest.TestCase):
    def test_both_tracks_agree(self):
        out = HybridRRF().fuse([1, 2, 3], [1, 2], 0.5, 0.5)
        self.assertEqual(out[0]["doc_id"], 1)
        self.assertAlmostEqual(out[0]["score"], 1.0, places=3)

    def test_single_track_top1_is_half(self):
        """两轨都存在但只有一轨命中时，该条归一化分是 0.5。

        这是 score_threshold 默认 0.35 的依据：阈值低于 0.5 意味着
        "单轨命中即注入"，高于 0.5 则要求两轨共识。
        """
        out = HybridRRF().fuse([7, 8], [9, 10], 0.5, 0.5)
        by_id = {item["doc_id"]: item["score"] for item in out}
        self.assertAlmostEqual(by_id[7], 0.5, places=3)

    def test_absent_track_gives_full_weight(self):
        """一轨为空时权重全给存活轨，Top-1 归一化为 1.0（单轨退化模式）。"""
        out = HybridRRF().fuse([7, 8], [], 0.5, 0.5)
        self.assertEqual(out[0]["doc_id"], 7)
        self.assertAlmostEqual(out[0]["score"], 1.0, places=3)

    def test_degrades_to_single_track(self):
        out = HybridRRF().fuse([], [5, 6], 0.5, 0.5)
        self.assertEqual(out[0]["doc_id"], 5)
        self.assertAlmostEqual(out[0]["score"], 1.0, places=3)

    def test_both_empty(self):
        self.assertEqual(HybridRRF().fuse([], [], 0.5, 0.5), [])


class TestRelevance(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "how to configure the embedding provider",
            "BM25 weight versus vector weight",
        ]
        bm = BM25Okapi()
        bm.fit(self.corpus)
        self.idf = bm.idf
        self.vocab = set()
        for doc in self.corpus:
            self.vocab.update(BM25Okapi.tokenize(doc))

    def test_fully_covered_query(self):
        cov = query_coverage("embedding provider", [self.vocab], self.idf)
        self.assertGreater(cov, 0.9)

    def test_unknown_words_lower_coverage(self):
        """语料里不存在的词是离题信号，应显著拉低覆盖率。"""
        good = query_coverage("embedding provider", [self.vocab], self.idf)
        bad = query_coverage("sourdough bread bake", [self.vocab], self.idf)
        self.assertLess(bad, good)
        self.assertLess(bad, 0.1)

    def test_empty_query(self):
        self.assertEqual(query_coverage("", [self.vocab], self.idf), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
