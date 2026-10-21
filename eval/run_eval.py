"""
Offline retrieval evaluation for astrbot_plugin_rag_pro.

Runs the real indexing and retrieval code paths (BM25Okapi, HybridRRF,
MarkdownSplitter, IndexDatabase) against the documents in eval/docs and the
queries in eval/benchmark.json. No AstrBot host and no embedding provider
are required: the dense track is stubbed with a deterministic bag-of-words
embedding so the fusion behaviour is exercised without network calls.

Reported metrics, computed over the queries that have expected chunks:
  Hit@1    fraction of queries whose top-1 chunk is an expected chunk
  Hit@5    fraction of queries with at least one expected chunk in top-5
  MRR      mean reciprocal rank of the first expected chunk
  NDCG@5   normalized discounted cumulative gain at 5
Also reported:
  False-positive rate on the off-topic queries (expected == [])

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --config bm25_only     # single-track baseline
  python eval/run_eval.py --json                 # machine-readable output
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

# Allow running from the repository root or from inside eval/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.engine.bm25 import BM25Okapi          # noqa: E402
from core.engine.relevance import query_coverage as _cov  # noqa: E402
from core.engine.rrf import HybridRRF           # noqa: E402
from core.storage.index_db import IndexDatabase  # noqa: E402
from core.storage.splitter import MarkdownSplitter  # noqa: E402

EVAL_DIR = os.path.join(ROOT, "eval")
DOCS_DIR = os.path.join(EVAL_DIR, "docs")
BENCH_FILE = os.path.join(EVAL_DIR, "benchmark.json")


# ── 确定性的伪 embedding ────────────────────────────────────────
# 用词袋哈希生成固定维度的稠密向量，模拟"语义相近则向量相近"的行为。
# 目的是让双轨融合链路可被离线测试，而不是替代真实 embedding 模型。
DIM = 256


def fake_embed(text: str) -> List[float]:
    vec = [0.0] * DIM
    for tok in BM25Okapi.tokenize(text):
        # 稳定哈希：不用内置 hash()，它按进程随机化
        h = 0
        for ch in tok:
            h = (h * 31 + ord(ch)) % 1000003
        vec[h % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── 索引构建 ────────────────────────────────────────────────────
def build_index(chunk_size: int = 400, chunk_overlap: int = 60):
    """用真实的 IndexDatabase + MarkdownSplitter 建索引，返回内存中的结构。"""
    tmpdir = tempfile.mkdtemp(prefix="ragpro-eval-")
    db_path = os.path.join(tmpdir, "index.db")
    db = IndexDatabase(db_path)

    chunks = []
    next_id = 0
    for name in sorted(os.listdir(DOCS_DIR)):
        if not name.endswith(".md"):
            continue
        path = os.path.join(DOCS_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        new_chunks = MarkdownSplitter.split_text(
            text,
            source_file=name,
            start_id=next_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        next_id += len(new_chunks)

        # 走真实持久化路径，顺带验证 SQLite 层
        vec_map = {c.doc_id: fake_embed(f"{c.header}\n{c.content}") for c in new_chunks}
        db.save_file_chunks(path, "hash-" + name, os.path.getmtime(path), new_chunks, vectors=vec_map)
        chunks.extend(new_chunks)

    # 从库里读回来，确保评测用的是持久化后的数据
    loaded, vectors = db.load_all_chunks()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return loaded, vectors


def label(chunk) -> str:
    """把切片映射成 benchmark 里的 expected 标识：file.md::header

    source_file 在库中保存的是磁盘绝对路径，比对前只取文件名，
    并把切片产生的 "(第N部分)" 后缀去掉，让同一章节的所有子片共享标签。
    """
    header = chunk.header.split(" (第")[0].strip()
    name = os.path.basename(chunk.source_file)
    return f"{name}::{header}"


# ── 检索 ────────────────────────────────────────────────────────
def build_vocab(chunks) -> set:
    """全库词表，对应 main.py 里的 self._vocab。"""
    v = set()
    for c in chunks:
        v.update(BM25Okapi.tokenize(f"{c.header}\n{c.content}"))
    return v


def query_coverage(query: str, vocab: set, bm25: BM25Okapi) -> float:
    """直接复用插件里的实现，避免评测与线上行为漂移。"""
    return _cov(query, [vocab], bm25.idf)


def retrieve(
    query: str,
    chunks,
    vectors: Dict[int, List[float]],
    bm25: BM25Okapi,
    vocab: set,
    mode: str = "hybrid",
    bm25_w: float = 0.5,
    vec_w: float = 0.5,
    threshold: float = 0.0,
    top_k: int = 5,
    min_coverage: float = 0.0,
) -> List[int]:
    """复刻 main.py 的融合流程，返回排好序的 doc_id 列表。"""
    # 稀疏轨
    scores = bm25.get_scores(query)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    picked = [i for i in order if scores[i] > 0.05][:30]
    bm25_ids = [chunks[i].doc_id for i in picked]

    # 稠密轨
    vec_ids: List[int] = []
    if mode == "hybrid" and vectors:
        qv = fake_embed(query)
        sims = [(did, cosine(qv, vec)) for did, vec in vectors.items()]
        sims.sort(key=lambda x: x[1], reverse=True)
        vec_ids = [did for did, s in sims if s > 0.15][:30]

    # 绝对相关性门槛：先把完全离题的提问挡掉
    if min_coverage > 0 and query_coverage(query, vocab, bm25) < min_coverage:
        return []

    if mode == "bm25_only":
        return bm25_ids[:top_k]
    if mode == "vector_only":
        return vec_ids[:top_k]

    fused = HybridRRF().fuse(bm25_ids, vec_ids, bm25_weight=bm25_w, vector_weight=vec_w)
    out = [item["doc_id"] for item in fused if item["score"] >= threshold]
    return out[:top_k]


# ── 指标 ────────────────────────────────────────────────────────
def ndcg_at_k(ranked_labels: List[str], expected: List[str], k: int = 5) -> float:
    """NDCG@k，按"章节"计分：同一章节的多个子片只算一次收益。

    切分会让一个章节产生多个 chunk，若按 chunk 逐条计收益，
    同一份知识被反复计入，NDCG 会超过 1。
    """
    seen = set()
    gains = []
    for lbl in ranked_labels[:k]:
        if lbl in expected and lbl not in seen:
            seen.add(lbl)
            gains.append(1.0)
        else:
            gains.append(0.0)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(set(expected)), k)))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def reciprocal_rank(ranked_labels: List[str], expected: List[str]) -> float:
    for i, lbl in enumerate(ranked_labels, start=1):
        if lbl in expected:
            return 1.0 / i
    return 0.0


def evaluate(config: str, bm25_w: float, vec_w: float, threshold: float, min_cov: float = 0.0) -> Dict:
    chunks, vectors = build_index()
    bm25 = BM25Okapi()
    bm25.fit([f"{c.header}\n{c.content}" for c in chunks])
    vocab = build_vocab(chunks)

    with open(BENCH_FILE, "r", encoding="utf-8") as f:
        bench = json.load(f)

    hit1 = hit5 = 0
    rr_sum = ndcg_sum = 0.0
    scored = 0
    fp_total = fp_hit = 0
    per_query = []

    for q in bench["queries"]:
        expected = q.get("expected") or []
        ids = retrieve(
            q["query"], chunks, vectors, bm25, vocab,
            mode=config, bm25_w=bm25_w, vec_w=vec_w,
            threshold=threshold, top_k=5, min_coverage=min_cov,
        )
        ranked_labels = [label(chunks[i]) for i in ids if i < len(chunks)]

        if not expected:
            # 离题查询：任何注入都算误报
            fp_total += 1
            if ids:
                fp_hit += 1
            per_query.append({
                "id": q["id"], "query": q["query"], "expected": [],
                "retrieved": ranked_labels, "false_positive": bool(ids),
            })
            continue

        scored += 1
        if ranked_labels and ranked_labels[0] in expected:
            hit1 += 1
        if any(lbl in expected for lbl in ranked_labels[:5]):
            hit5 += 1
        rr_sum += reciprocal_rank(ranked_labels, expected)
        ndcg_sum += ndcg_at_k(ranked_labels, expected, 5)

        per_query.append({
            "id": q["id"], "query": q["query"], "expected": expected,
            "retrieved": ranked_labels,
            "hit1": bool(ranked_labels and ranked_labels[0] in expected),
            "hit5": any(lbl in expected for lbl in ranked_labels[:5]),
        })

    n = max(scored, 1)
    return {
        "config": config,
        "num_chunks": len(chunks),
        "scored_queries": scored,
        "hit@1": round(hit1 / n, 4),
        "hit@5": round(hit5 / n, 4),
        "mrr": round(rr_sum / n, 4),
        "ndcg@5": round(ndcg_sum / n, 4),
        "false_positive_rate": round(fp_hit / fp_total, 4) if fp_total else 0.0,
        "off_topic_queries": fp_total,
        "per_query": per_query,
    }


def main():
    ap = argparse.ArgumentParser(description="Offline retrieval evaluation")
    ap.add_argument("--config", default="hybrid",
                    choices=["hybrid", "bm25_only", "vector_only"],
                    help="retrieval configuration to evaluate")
    ap.add_argument("--bm25-weight", type=float, default=0.5)
    ap.add_argument("--vector-weight", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="RRF score threshold (plugin default is 0.35)")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="minimum query-term coverage before injecting (0 disables)")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    if not os.path.isdir(DOCS_DIR):
        print(f"missing corpus directory: {DOCS_DIR}", file=sys.stderr)
        return 1

    result = evaluate(args.config, args.bm25_weight, args.vector_weight,
                      args.threshold, args.min_coverage)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"config            : {result['config']}")
    print(f"chunks indexed    : {result['num_chunks']}")
    print(f"scored queries    : {result['scored_queries']}")
    print(f"Hit@1             : {result['hit@1']:.4f}")
    print(f"Hit@5             : {result['hit@5']:.4f}")
    print(f"MRR               : {result['mrr']:.4f}")
    print(f"NDCG@5            : {result['ndcg@5']:.4f}")
    print(f"min coverage      : {args.min_coverage}")
    print(f"false-positive    : {result['false_positive_rate']:.4f} "
          f"({result['off_topic_queries']} off-topic queries)")

    misses = [q for q in result["per_query"] if q.get("expected") and not q.get("hit5")]
    if misses:
        print("\nmisses:")
        for q in misses:
            print(f"  {q['id']}  {q['query']}")
            print(f"      want: {q['expected']}")
            print(f"      got : {q['retrieved'][:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
