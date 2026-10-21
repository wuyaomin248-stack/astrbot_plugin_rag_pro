import os
import time
import json
from typing import List, Dict, Any, Optional, Set

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import register, Star
from astrbot.core.provider.entities import ProviderRequest

from .core.engine.bm25 import BM25Okapi
from .core.engine.rrf import HybridRRF
from .core.engine.relevance import query_coverage
from .core.engine.vector import VectorRetrievalEngine
from .core.engine.kb_bridge import KnowledgeSourceBridge
from .core.storage.splitter import MarkdownSplitter, DocumentChunk
from .core.storage.index_db import IndexDatabase

@register(
    "astrbot_plugin_rag_pro",
    "日海 & 橙子汐",
    "基于 BM25 + 向量真双轨 RRF 混合召回的精准 RAG 插件（增量缓存与代码块保护）",
    "0.2.0",
    "https://github.com/wuyaomin248-stack/astrbot_plugin_rag_pro"
)
class RagProPlugin(Star):
    def __init__(self, context: Any, config: Dict[str, Any] = None):
        super().__init__(context)
        self.config = config or {}
        
        # 核心引擎实例化
        self.bm25_engine = BM25Okapi()
        self.rrf_engine = HybridRRF()
        self.vector_engine = VectorRetrievalEngine(
            context=self.context,
            preferred_provider_id=self.config.get("embedding_provider_id")
        )
        self.bridge = KnowledgeSourceBridge()
        
        # 缓存持久化数据库
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../plugin_data/astrbot_plugin_rag_pro/rag_cache.db"
        )
        self.index_db = IndexDatabase(db_path)
        
        self.chunks: List[DocumentChunk] = []
        self._doc_id_to_chunk: Dict[int, DocumentChunk] = {}
        self._vocab: Set[str] = set()
        self._is_ready = False

    async def initialize(self):
        try:
            # 动态同步 context 至向量检索引擎
            self.vector_engine.set_context(
                self.context,
                preferred_provider_id=self.config.get("embedding_provider_id")
            )
            await self._sync_incremental_index()
            self._is_ready = True
            logger.info(f"[rag_pro] 知识库就绪，共索引 {len(self.chunks)} 个文档切片 (真双轨就绪)")
        except Exception as e:
            logger.error(f"[rag_pro] 初始化索引失败: {e}", exc_info=True)

    async def _sync_incremental_index(self):
        """
        基于文件 SHA-256 和 mtime 的高效增量索引同步
        """
        extra_dirs = self.config.get("extra_kb_dirs", [])
        discovered_files = self.bridge.discover_document_files(extra_dirs=extra_dirs)
        cached_meta = self.index_db.get_all_doc_meta()
        
        active_files = set(discovered_files)
        # 1. 清理已在磁盘上被删除的文件
        deleted_files = [f for f in cached_meta.keys() if f not in active_files]
        if deleted_files:
            logger.info(f"[rag_pro] 清理已删除文件切片缓存: {len(deleted_files)} 个")
            self.index_db.delete_files(deleted_files)

        chunk_size = int(self.config.get("chunk_size", 500))
        chunk_overlap = int(self.config.get("chunk_overlap", 60))
        enable_vector = bool(self.config.get("enable_vector", True))
        vector_available = self.vector_engine.is_available() if enable_vector else False

        # 2. 识别需要新增或发生修改的文件
        modified_or_new = []
        for fpath in discovered_files:
            try:
                mtime = os.path.getmtime(fpath)
                old = cached_meta.get(fpath)
                if not old or old["mtime"] != mtime:
                    f_hash = self.index_db.calculate_file_hash(fpath)
                    if not old or old["file_hash"] != f_hash:
                        modified_or_new.append((fpath, f_hash, mtime))
            except Exception as e:
                logger.debug(f"[rag_pro] 校验文件异常 {fpath}: {e}")

        # 3. 仅对增量/变更文件重新切片与计算向量
        if modified_or_new:
            logger.info(f"[rag_pro] 检测到 {len(modified_or_new)} 个文件需增量切分建索")
            # 获取当前最大 doc_id
            existing_chunks, _ = self.index_db.load_all_chunks()
            next_id = max([c.doc_id for c in existing_chunks], default=-1) + 1

            for fpath, f_hash, mtime in modified_or_new:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    fname = os.path.basename(fpath)
                    new_chunks = MarkdownSplitter.split_text(
                        text,
                        source_file=fname,
                        start_id=next_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    next_id += len(new_chunks)

                    # 若向量引擎就绪，批量计算新切片向量
                    vec_map: Dict[int, List[float]] = {}
                    if vector_available and new_chunks:
                        try:
                            contents = [f"{c.header}\n{c.content}" for c in new_chunks]
                            vecs = await self.vector_engine.compute_embeddings(contents)
                            for c, v in zip(new_chunks, vecs):
                                vec_map[c.doc_id] = v
                        except Exception as ve:
                            logger.warning(f"[rag_pro] 批量计算向量失败 (将回退至 BM25): {ve}")

                    self.index_db.save_file_chunks(fpath, f_hash, mtime, new_chunks, vectors=vec_map)
                except Exception as err:
                    logger.error(f"[rag_pro] 处理增量文件失败 {fpath}: {err}")

        # 4. 从 SQLite 极速载入全量 chunks 并训练 BM25 / 装配向量池
        all_chunks, vector_map = self.index_db.load_all_chunks()
        self.chunks = all_chunks
        self._doc_id_to_chunk = {c.doc_id: c for c in all_chunks}

        # 训练 BM25 倒排索引
        corpus = [f"{c.header}\n{c.content}" for c in self.chunks]
        self.bm25_engine.fit(corpus)

        # 全库词表，供相关性门槛使用
        self._vocab = set()
        for text in corpus:
            self._vocab.update(BM25Okapi.tokenize(text))

        # 填充向量引擎
        self.vector_engine.clear()
        if vector_map:
            doc_ids = list(vector_map.keys())
            vectors = list(vector_map.values())
            self.vector_engine.add_vectors(doc_ids, vectors)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.config.get("enable_rag", True) or not self._is_ready or not self.chunks:
            return

        prompt_text = req.prompt or ""
        if not prompt_text and hasattr(req, "contexts") and req.contexts:
            for m in reversed(req.contexts):
                if isinstance(m, dict) and m.get("role") == "user":
                    prompt_text = m.get("content", "")
                    break

        query = str(prompt_text).strip()
        if not query or len(query) < 2:
            return

        # 1. 稀疏检索轨 (BM25)
        bm25_scores = self.bm25_engine.get_scores(query)
        ranked_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        # 过滤低分噪音，保留候选 ID
        ranked_bm25_doc_ids = [self.chunks[i].doc_id for i in ranked_bm25_idx if bm25_scores[i] > 0.05][:30]

        # 2. 稠密向量检索轨 (Vector Cosine)
        ranked_vector_doc_ids = []
        enable_vector = bool(self.config.get("enable_vector", True))
        if enable_vector and self.vector_engine.is_available() and self.vector_engine.vectors:
            try:
                vec_results = await self.vector_engine.retrieve(query, top_k=30)
                ranked_vector_doc_ids = [doc_id for doc_id, sim in vec_results if sim > 0.15]
            except Exception as e:
                logger.warning(f"[rag_pro] 向量检索异常: {e}")

        # 若双轨均无召回，安全退出
        if not ranked_bm25_doc_ids and not ranked_vector_doc_ids:
            return

        # 2.5 绝对相关性门槛：RRF 只按排名融合，丢掉了"到底像不像"的信息。
        # 若不加这一层，像 "how do I bake sourdough bread" 这种完全离题的提问，
        # 只要命中一个常见字就会进候选并被归一化成高分，最终污染提示词。
        # 判定基于全库词表，而不是候选集——候选集本身可能已经漏召。
        min_coverage = float(self.config.get("min_query_coverage", 0.0))
        if min_coverage > 0 and self._query_coverage(query) < min_coverage:
            logger.debug(
                f"[rag_pro] 查询与知识库相关性不足 (coverage < {min_coverage})，跳过注入"
            )
            return

        # 3. 真实 RRF 多路加权融合
        bm25_w = float(self.config.get("bm25_weight", 0.5))
        vec_w = float(self.config.get("vector_weight", 0.5))
        fused = self.rrf_engine.fuse(
            ranked_bm25_doc_ids,
            ranked_vector_doc_ids,
            bm25_weight=bm25_w,
            vector_weight=vec_w
        )

        threshold = float(self.config.get("score_threshold", 0.35))
        top_k = int(self.config.get("top_k", 3))
        qualified = [item for item in fused if item["score"] >= threshold][:top_k]
        if not qualified:
            return

        # 4. 冲突抑制：压制 AstrBot 原生知识库干扰与双重注入
        if self.config.get("suppress_native_rag", True):
            await KnowledgeSourceBridge.suppress_native_rag(event, req)

        # 5. 上下文预算装配（替代脆弱的 [:800] 物理截断）
        max_context_chars = int(self.config.get("max_context_chars", 2500))
        rag_blocks = ["[RAG Pro 知识增强上下文]"]
        total_chars = 0

        for q in qualified:
            doc_id = q["doc_id"]
            c = self._doc_id_to_chunk.get(doc_id)
            if not c:
                continue

            chunk_header = f"--- 来源: {c.source_file} | 章节: {c.header} (综合置信度: {q['score']}) ---"
            chunk_content = c.content

            # 如果加入当前块会超出总预算，且已有块，则停止拼接；若为首个块，做安全截断
            if total_chars + len(chunk_content) > max_context_chars:
                if total_chars > 0:
                    break
                else:
                    # 保障第一名至少有部分展示，但检查代码块闭合
                    truncated = chunk_content[:max_context_chars]
                    if truncated.count("```") % 2 != 0:
                        truncated += "\n```"
                    chunk_content = truncated

            rag_blocks.append(chunk_header)
            rag_blocks.append(chunk_content)
            total_chars += len(chunk_header) + len(chunk_content)

        rag_inject_str = "\n".join(rag_blocks) + "\n\n"

        if hasattr(req, "system_prompt") and req.system_prompt:
            req.system_prompt = rag_inject_str + req.system_prompt
        else:
            req.system_prompt = rag_inject_str

        logger.info(f"[rag_pro] 成功为用户提问隐式注入 {len(qualified)} 条高质知识 (双轨融合，总长度 {total_chars} 字符)")

    def _query_coverage(self, query: str) -> float:
        """查询词项被**整个知识库**覆盖的 IDF 加权比例（0~1）。

        用于把"完全离题"的提问挡在注入之前。语料里不存在的词按不覆盖
        且高权重计入，因此 "how do I bake sourdough bread" 会被
        sourdough/bread 拉低。

        统计范围是全库而不是候选集：候选集本身可能已经漏召，用它算
        覆盖率会形成「召回差 → 覆盖率低 → 被否决」的循环依赖。
        """
        return query_coverage(query, [self._vocab], self.bm25_engine.idf)

    @filter.command("rag")
    async def rag_command(self, event: AstrMessageEvent, action: str = "status"):
        if action == "rebuild":
            await event.send(event.plain_result("⏳ RAG Pro 正在增量同步与重构索引..."))
            self.index_db.clear_all()
            await self._sync_incremental_index()
            await event.send(event.plain_result(f"✅ RAG Pro 索引全量重构完毕，共索引 {len(self.chunks)} 个切片。"))
        elif action == "sync":
            await self._sync_incremental_index()
            await event.send(event.plain_result(f"✅ RAG Pro 增量同步完成，当前共索引 {len(self.chunks)} 个切片。"))
        else:
            vec_status = "正常" if self.vector_engine.is_available() else "未配置/不可用 (退化单轨)"
            await event.send(event.plain_result(
                f"📊 RAG Pro 引擎状态:\n"
                f"- 已加载切片数: {len(self.chunks)}\n"
                f"- 向量检索轨: {vec_status}\n"
                f"- BM25 权重 / 向量权重: {self.config.get('bm25_weight', 0.5)} / {self.config.get('vector_weight', 0.5)}\n"
                f"- 截断阈值: {self.config.get('score_threshold', 0.35)}\n"
                f"- 状态: {'就绪' if self._is_ready else '未初始化'}"
            ))
