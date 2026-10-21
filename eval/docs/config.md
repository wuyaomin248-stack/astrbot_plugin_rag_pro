# Configuration

## Chunking

`chunk_size` controls the target character count of a single indexed chunk.
The default is 500. `chunk_overlap` sets how many characters of trailing
context are carried into the next chunk when a section has to be split; the
default is 60.

Raising `chunk_size` reduces the number of chunks and therefore the number
of entries injected per request, but each entry becomes less focused.
`top_k` caps how many chunks are injected at all; the default is 3.

Practical guidance:

- Short, factual knowledge bases (error codes, config reference): keep
  `chunk_size` around 300 and `top_k` at 3.
- Long prose documentation: raise `chunk_size` to 800 and keep `top_k` low.

## Budget

`max_context_chars` is the total character budget for injected knowledge,
default 2500. When the next chunk would exceed the budget, assembly stops
instead of truncating mid-chunk. The only exception is the first-ranked
chunk, which is truncated but with its code fences rebalanced so the block
stays syntactically closed.
