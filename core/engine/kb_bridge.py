"""
多源知识发现与原生 AstrBot 知识库桥接器 (KB Bridge)
1. 聚合发现：支持扫描工作区 (workspaces)、插件文档、知识库目录中的 Markdown/TXT 文档；
2. 原生对接：可直接读取 AstrBot 原生知识库 doc.db 中的真实切片；
3. 原生冲突抑制：当 RAG Pro 命中高质知识时，通过 AstrBot Session 配置或清理标记防止原生低质检索重复注入。
"""

import os
import glob
import sqlite3
import json
from typing import List, Dict, Any, Optional, Set
from astrbot.api import logger, sp

class KnowledgeSourceBridge:
    def __init__(self, base_dirs: Optional[List[str]] = None):
        self.base_dirs = base_dirs or [
            "/AstrBot/data/workspaces",
            "/AstrBot/data/plugins",
            "/AstrBot/data/knowledge_base",
            "/opt/astrbot/data/workspaces",
            "/opt/astrbot/data/plugins",
            "/opt/astrbot/data/knowledge_base",
        ]

    def discover_document_files(self, extra_dirs: Optional[List[str]] = None) -> List[str]:
        """
        全量检索磁盘上所有可用的 Markdown 与文本文件
        自动跳过临时文件、缓存、备份及无意义依赖目录
        """
        all_dirs = list(self.base_dirs)
        if extra_dirs:
            all_dirs.extend(extra_dirs)

        discovered: Set[str] = set()
        exclude_patterns = [
            "/__pycache__/", "/node_modules/", "/.git/", "/.pytest_cache/",
            "/plugin_backups/", "/temp/", "/vendor/"
        ]

        for d in all_dirs:
            if not os.path.exists(d):
                continue
            for ext in ("*.md", "*.txt"):
                pattern = os.path.join(d, "**", ext)
                for fpath in glob.glob(pattern, recursive=True):
                    normalized = os.path.abspath(fpath)
                    # 排除内部无关依赖和临时文件
                    if any(ex in normalized for ex in exclude_patterns):
                        continue
                    if os.path.isfile(normalized) and os.path.getsize(normalized) > 50:
                        discovered.add(normalized)

        return sorted(list(discovered))

    def load_native_astrbot_documents(self, kb_base_dir: str = "/AstrBot/data/knowledge_base") -> List[Dict[str, Any]]:
        """
        直接读取 AstrBot 原生已建立的知识库（如 doc.db），
        无需用户手工导出或重复复制，直接共享知识库资产。
        """
        if not os.path.exists(kb_base_dir):
            return []

        native_docs: List[Dict[str, Any]] = []
        # 查找所有 doc.db
        for root, dirs, files in os.walk(kb_base_dir):
            if "doc.db" in files:
                db_path = os.path.join(root, "doc.db")
                kb_id = os.path.basename(root)
                try:
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    c = conn.cursor()
                    # 检查是否有 documents 表
                    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
                    if not c.fetchone():
                        conn.close()
                        continue
                    
                    c.execute("SELECT id, title, content, metadata FROM documents")
                    rows = c.fetchall()
                    for r in rows:
                        native_docs.append({
                            "source_file": f"AstrBot-KB:{kb_id}:{r['title'] or 'doc'}",
                            "header": r["title"] or "原生知识库",
                            "content": r["content"],
                            "doc_id": r["id"],
                        })
                    conn.close()
                    logger.info(f"[rag_pro.bridge] 成功从原生知识库 {kb_id} 导入 {len(rows)} 条文档切片")
                except Exception as e:
                    logger.warning(f"[rag_pro.bridge] 读取原生 doc.db ({db_path}) 失败: {e}")

        return native_docs

    @staticmethod
    async def suppress_native_rag(event: Any, req: Any):
        """
        抑制 AstrBot 原生知识库检索，避免双重注入与 0.12/0.08 低分污染。
        1. 临时设置 session 的 kb_config 为空列表；
        2. 清理 req.extra_user_content_parts 中可能存在的原生知识注入标记。
        """
        try:
            umo = getattr(event, "unified_msg_origin", None)
            if umo:
                # 设置会话级别 kb_ids 为空，使原生 retrieve_knowledge_base 直接返回 None
                await sp.session_set(umo, "kb_config", {"kb_ids": []})
        except Exception as e:
            logger.debug(f"[rag_pro.bridge] 设置 session kb_config 抑制失败: {e}")

        try:
            # 清理 extra_user_content_parts 中已被原生注入的块
            if hasattr(req, "extra_user_content_parts") and req.extra_user_content_parts:
                filtered_parts = []
                for p in req.extra_user_content_parts:
                    p_text = getattr(p, "text", "")
                    if "[Related Knowledge Base Results]" not in p_text:
                        filtered_parts.append(p)
                req.extra_user_content_parts = filtered_parts
        except Exception as e:
            logger.debug(f"[rag_pro.bridge] 清理 extra_user_content_parts 失败: {e}")
