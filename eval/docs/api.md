# API

## Embedding methods

The vector engine negotiates with the host's embedding provider and supports
three call shapes, tried in this order:

1. `get_embeddings_batch(texts, batch_size=16)` — preferred, batch call
2. `get_embeddings(texts)` — batch without explicit size
3. `get_embedding(text)` — single item, called in a loop

If none of these exist on the provider, the engine raises `AttributeError`
and the plugin degrades to the BM25-only track rather than failing the
whole request.

## Splitter methods

`MarkdownSplitter.split_text(text, source_file, start_id, chunk_size, chunk_overlap)`
is the single entry point for indexing. It first splits on Markdown headings
levels 1 through 4, then splits each section further if it exceeds
`chunk_size`. Sections that get split are renamed with a `(第N部分)` suffix so
the origin of each chunk stays traceable.
