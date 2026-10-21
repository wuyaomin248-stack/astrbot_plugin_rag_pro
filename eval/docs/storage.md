# Storage

## Incremental

`IndexDatabase` persists everything to a single SQLite file. Two tables:
`doc_meta` holds one row per source file with its `file_hash`, `mtime` and
chunk count; `chunks` holds one row per chunk including the optional
`vector_json` blob.

On startup the plugin compares every discovered file against `doc_meta`
using a two-step check: first `mtime`, then a SHA-256 `file_hash` computed
only when the mtime changed. Files whose hash is unchanged are skipped
entirely, so a restart with no edits does zero embedding work. Files that
disappeared from disk are deleted from both tables in one pass.

This matters because recomputing embeddings is the dominant cost of
indexing a large knowledge base.

## Code fence safety

Splitting respects fenced code blocks. The splitter first separates a
section into text blocks and code blocks, then treats each code block as
atomic — it is never cut at an arbitrary character offset.

When a single code block is longer than twice `chunk_size`, it is split by
lines instead, and each resulting piece is re-wrapped with opening and
closing fences so that no chunk contains an unbalanced code fence. A code
block that gets cut in half is the most common cause of malformed context
being injected into the prompt.
