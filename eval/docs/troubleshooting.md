# Troubleshooting

## No retrieval

If questions that should hit the knowledge base return nothing:

- Check that documents exist. `/rag status` prints the loaded chunk count.
  Zero means discovery found no files.
- Check `extra_kb_dirs`. The default discovery paths assume the container
  layout `/AstrBot/data/...`; on a different install the paths must be
  overridden.
- Check the score threshold. With only one of the two tracks active, fused
  scores are lower than with both, and a threshold of 0.5 or higher can
  filter out everything. Try 0.35.
- Files smaller than 50 bytes are skipped by discovery, as are directories
  named `__pycache__`, `node_modules`, `.git` and a few others.

## Startup

If the plugin does not appear or fails to load:

- The host must be AstrBot with the `astrbot.api` package importable. A
  bare Python interpreter cannot import `main.py`.
- Index errors during `initialize()` are logged but do not abort loading;
  the plugin stays registered in a not-ready state. Look for
  `[rag_pro] 初始化索引失败` in the log for the underlying cause.
- A corrupted index can be rebuilt with `/rag rebuild`, which clears both
  tables and re-indexes from disk.
