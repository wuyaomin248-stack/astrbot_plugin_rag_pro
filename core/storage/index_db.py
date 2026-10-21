"""
RAG 索引持久化与增量更新管理器
采用 SQLite 存储文档元数据、切片、向量与 BM25 词频统计，
利用 SHA-256 和 mtime 进行文件级增量感知，实现秒级热启动。
"""

import os
import json
import sqlite3
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple

from .splitter import DocumentChunk

class IndexDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            c = conn.cursor()
            # 1. 文档元数据表：记录文件路径、哈希、修改时间等
            c.execute("""
                CREATE TABLE IF NOT EXISTS doc_meta (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            # 2. 切片表：记录切片内容及可选的向量 BLOB
            c.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    doc_id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    header TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    vector_json TEXT,
                    FOREIGN KEY(file_path) REFERENCES doc_meta(file_path) ON DELETE CASCADE
                )
            """)
            # 3. 辅助索引加速
            c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_hash ON chunks(chunk_hash)")
            conn.commit()

    @staticmethod
    def calculate_file_hash(filepath: str) -> str:
        """计算文件的 SHA-256 哈希"""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def get_all_doc_meta(self) -> Dict[str, Dict[str, Any]]:
        """获取所有已索引文件的元数据字典 {file_path: {file_hash, mtime, chunk_count}}"""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT file_path, file_hash, mtime, chunk_count FROM doc_meta")
            rows = c.fetchall()
            return {
                r["file_path"]: {
                    "file_hash": r["file_hash"],
                    "mtime": r["mtime"],
                    "chunk_count": r["chunk_count"],
                }
                for r in rows
            }

    def load_all_chunks(self) -> Tuple[List[DocumentChunk], Dict[int, List[float]]]:
        """
        从缓存加载所有切片及预计算的向量。
        返回 (chunks, {doc_id: vector})
        """
        chunks: List[DocumentChunk] = []
        vectors: Dict[int, List[float]] = {}

        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT doc_id, file_path, header, content, chunk_hash, vector_json FROM chunks ORDER BY doc_id ASC")
            rows = c.fetchall()
            for r in rows:
                chk = DocumentChunk(
                    doc_id=r["doc_id"],
                    source_file=r["file_path"],
                    header=r["header"],
                    content=r["content"],
                    chunk_hash=r["chunk_hash"],
                )
                chunks.append(chk)
                if r["vector_json"]:
                    try:
                        vec = json.loads(r["vector_json"])
                        if isinstance(vec, list):
                            vectors[r["doc_id"]] = vec
                    except Exception:
                        pass

        return chunks, vectors

    def delete_files(self, file_paths: List[str]):
        """从数据库中删除指定已废弃的文件及其关联的切片"""
        if not file_paths:
            return
        with self._get_conn() as conn:
            c = conn.cursor()
            c.executemany("DELETE FROM chunks WHERE file_path = ?", [(p,) for p in file_paths])
            c.executemany("DELETE FROM doc_meta WHERE file_path = ?", [(p,) for p in file_paths])
            conn.commit()

    def save_file_chunks(
        self,
        file_path: str,
        file_hash: str,
        mtime: float,
        chunks: List[DocumentChunk],
        vectors: Optional[Dict[int, List[float]]] = None,
    ):
        """
        保存单个文件的切片及元数据（原子事务）。
        """
        vectors = vectors or {}
        now = time.time()
        with self._get_conn() as conn:
            c = conn.cursor()
            # 先清空该文件已有的 chunks
            c.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            
            # 插入新 chunks
            chunk_data = []
            for chk in chunks:
                vec_json = json.dumps(vectors.get(chk.doc_id)) if chk.doc_id in vectors else None
                chunk_data.append((
                    chk.doc_id,
                    file_path,
                    chk.header,
                    chk.content,
                    chk.chunk_hash,
                    vec_json
                ))
            c.executemany(
                "INSERT OR REPLACE INTO chunks (doc_id, file_path, header, content, chunk_hash, vector_json) VALUES (?, ?, ?, ?, ?, ?)",
                chunk_data
            )

            # 更新文件元数据
            c.execute(
                "INSERT OR REPLACE INTO doc_meta (file_path, file_hash, mtime, chunk_count, updated_at) VALUES (?, ?, ?, ?, ?)",
                (file_path, file_hash, mtime, len(chunks), now)
            )
            conn.commit()

    def update_chunk_vectors(self, vector_map: Dict[int, List[float]]):
        """批量更新切片的向量数据"""
        if not vector_map:
            return
        with self._get_conn() as conn:
            c = conn.cursor()
            data = [(json.dumps(vec), doc_id) for doc_id, vec in vector_map.items()]
            c.executemany("UPDATE chunks SET vector_json = ? WHERE doc_id = ?", data)
            conn.commit()

    def clear_all(self):
        """清空全量索引缓存"""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM chunks")
            c.execute("DELETE FROM doc_meta")
            conn.commit()
