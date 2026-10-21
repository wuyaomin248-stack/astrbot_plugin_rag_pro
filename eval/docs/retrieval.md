# Retrieval

## Hybrid weights

Two retrieval tracks run against the same query. The sparse track is BM25
over an in-memory inverted index. The dense track is cosine similarity over
provider-computed embeddings.

Their scores are not comparable: BM25 is unbounded, cosine is bounded to
[-1, 1]. Reciprocal Rank Fusion resolves this by discarding the raw scores
and combining the rank positions instead, with the constant `k` set to 60.

`bm25_weight` and `vector_weight` default to 0.5 each. If one track returns
nothing — for example when no embedding provider is configured — the RRF
layer sets that weight to 0 and assigns the full weight to the surviving
track, so the plugin keeps working in single-track mode.

## Threshold

`score_threshold` filters the fused results. The default is 0.35. Fused
scores are normalized against the theoretical maximum `(bm25_weight +
vector_weight) / (k + 1)`, so the threshold is comparable across different
weight configurations. Lowering it injects more context at the cost of
precision; raising it above 0.6 in practice returns nothing for most queries.

## Vector search

Vector search uses numpy when it is importable, computing the whole cosine
matrix in one shot. If numpy is absent the engine falls back to a pure
Python loop computing the dot product and norm per document. The fallback is
slower but produces identical rankings, so the plugin never hard-depends on
numpy.

## Tokenization

The BM25 tokenizer is deliberately permissive for mixed Chinese and English.
English identifiers are matched as whole runs of `[a-zA-Z0-9_\-\.]`, and CJK
text is split into overlapping bigrams.

Keeping underscores, hyphens and dots inside tokens is what allows
`get_embeddings_batch` and `error_code_401` to be matched as whole identifiers
instead of being shredded into generic words.

Chinese uses bigrams rather than single characters. Splitting 「安装插件」
per character yields 安 / 装 / 插 / 件, which match almost any Chinese
document and therefore carry no signal. Bigrams keep the adjacency
information: 「安装」「装插」「插件」. 中文分词用二元组，这样不需要引入
外部词典也能让中文查询正常命中。

