# Integration

## Native RAG

AstrBot has its own knowledge-base retrieval. When both it and this plugin
fire, the model receives two knowledge blocks for the same query, which is
the double injection problem.

When `suppress_native_rag` is enabled and this plugin has a qualifying
result, the bridge does two things before the request continues:

1. Sets the session's `kb_config.kb_ids` to an empty list so the native
   retrieval returns nothing for this turn.
2. Removes any `[Related Knowledge Base Results]` part that the native path
   already appended to `req.extra_user_content_parts`.

Suppression only happens when this plugin actually has something better to
inject. If retrieval returns nothing, the native path is left untouched.

## Injection

The plugin registers an `on_llm_request` filter. It reads the user's latest
message from the request, retrieves chunks, and prepends them to
`req.system_prompt` under a `[RAG Pro 知识增强上下文]` header. Nothing is
written back to the conversation history, so the injected context does not
accumulate across turns.
