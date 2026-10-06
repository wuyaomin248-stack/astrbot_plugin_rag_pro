"""
结构化切片与文档持久化存储
支持基于 Markdown 标题层级、代码块保护与滑动窗口的语义切片
"""

import hashlib
import re
from typing import List, Dict, Any, Optional

class DocumentChunk:
    def __init__(
        self,
        doc_id: int,
        source_file: str,
        header: str,
        content: str,
        chunk_hash: Optional[str] = None,
        has_code_block: bool = False,
    ):
        self.doc_id = doc_id
        self.source_file = source_file
        self.header = header
        self.content = content.strip()
        self.char_count = len(self.content)
        self.has_code_block = has_code_block or ("```" in self.content)
        if chunk_hash:
            self.chunk_hash = chunk_hash
        else:
            h = hashlib.sha256()
            h.update(f"{source_file}:{header}:{self.content}".encode("utf-8"))
            self.chunk_hash = h.hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "header": self.header,
            "content": self.content,
            "chunk_hash": self.chunk_hash,
            "char_count": self.char_count,
            "has_code_block": self.has_code_block,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            doc_id=data["doc_id"],
            source_file=data.get("source_file", "unknown"),
            header=data.get("header", "正文"),
            content=data.get("content", ""),
            chunk_hash=data.get("chunk_hash"),
            has_code_block=data.get("has_code_block", False),
        )


class MarkdownSplitter:
    @staticmethod
    def _extract_blocks(text: str) -> List[Dict[str, Any]]:
        """
        将章节文本分割为自然段落和受保护的代码块。
        确保 ``` 代码块保持原子性，不被段落拆分打断。
        """
        lines = text.splitlines(keepends=True)
        blocks: List[Dict[str, Any]] = []
        in_code = False
        code_lines: List[str] = []
        normal_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    code_lines.append(line)
                    code_text = "".join(code_lines).strip()
                    if code_text:
                        blocks.append({"type": "code", "text": code_text})
                    code_lines = []
                    in_code = False
                else:
                    if normal_lines:
                        normal_text = "".join(normal_lines).strip()
                        if normal_text:
                            blocks.append({"type": "text", "text": normal_text})
                        normal_lines = []
                    in_code = True
                    code_lines.append(line)
            elif in_code:
                code_lines.append(line)
            else:
                normal_lines.append(line)

        if in_code and code_lines:
            code_text = "".join(code_lines).strip()
            if code_text:
                blocks.append({"type": "code", "text": code_text})

        if normal_lines:
            normal_text = "".join(normal_lines).strip()
            if normal_text:
                blocks.append({"type": "text", "text": normal_text})

        return blocks

    @classmethod
    def _split_section_content(
        cls,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 60,
    ) -> List[str]:
        """
        二级切分：在代码块保护前提下，将超长章节递归/滑动切分为子片段。
        """
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        blocks = cls._extract_blocks(text)
        sub_chunks: List[str] = []
        current_acc: List[str] = []
        current_acc_len = 0

        for block in blocks:
            b_text = block["text"]
            b_len = len(b_text)

            # 代码块原子保护
            if block["type"] == "code":
                if current_acc:
                    sub_chunks.append("\n\n".join(current_acc).strip())
                    current_acc = []
                    current_acc_len = 0
                
                # 如果单个代码块过长，按行保护切分，而不是字符级断裂
                if b_len > chunk_size * 2:
                    code_lines = b_text.splitlines()
                    tmp_code = []
                    tmp_len = 0
                    for cline in code_lines:
                        tmp_code.append(cline)
                        tmp_len += len(cline) + 1
                        if tmp_len >= chunk_size:
                            chunk_c = "\n".join(tmp_code).strip()
                            if not chunk_c.endswith("```"):
                                chunk_c += "\n```"
                            if not chunk_c.startswith("```"):
                                chunk_c = "```\n" + chunk_c
                            sub_chunks.append(chunk_c)
                            tmp_code = []
                            tmp_len = 0
                    if tmp_code:
                        tail_c = "\n".join(tmp_code).strip()
                        if not tail_c.endswith("```"):
                            tail_c += "\n```"
                        if not tail_c.startswith("```"):
                            tail_c = "```\n" + tail_c
                        sub_chunks.append(tail_c)
                else:
                    sub_chunks.append(b_text)
                continue

            # 普通文本块：按段落与标点细切
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', b_text) if p.strip()]
            for para in paragraphs:
                if current_acc_len + len(para) <= chunk_size:
                    current_acc.append(para)
                    current_acc_len += len(para) + 2
                else:
                    if current_acc:
                        chunk_str = "\n\n".join(current_acc).strip()
                        sub_chunks.append(chunk_str)
                        # 保留重叠上下文
                        overlap_tail = chunk_str[-chunk_overlap:] if len(chunk_str) > chunk_overlap else ""
                        current_acc = [overlap_tail] if overlap_tail else []
                        current_acc_len = len(overlap_tail)

                    # 如果单个段落仍大于 chunk_size，按标点切
                    if len(para) > chunk_size:
                        sentences = re.split(r'([。！？.!?\n])', para)
                        sent_acc = []
                        sent_len = 0
                        for i in range(0, len(sentences), 2):
                            s = sentences[i]
                            sep = sentences[i+1] if i+1 < len(sentences) else ""
                            full_s = s + sep
                            if sent_len + len(full_s) <= chunk_size:
                                sent_acc.append(full_s)
                                sent_len += len(full_s)
                            else:
                                if sent_acc:
                                    sub_chunks.append("".join(sent_acc).strip())
                                sent_acc = [full_s]
                                sent_len = len(full_s)
                        if sent_acc:
                            sub_chunks.append("".join(sent_acc).strip())
                    else:
                        current_acc.append(para)
                        current_acc_len += len(para) + 2

        if current_acc:
            chunk_str = "\n\n".join(current_acc).strip()
            if len(chunk_str) > 20:
                sub_chunks.append(chunk_str)

        # 过滤过短空块
        return [c for c in sub_chunks if len(c.strip()) > 20]

    @classmethod
    def split_text(
        cls,
        text: str,
        source_file: str = "unknown",
        start_id: int = 0,
        chunk_size: int = 500,
        chunk_overlap: int = 60,
    ) -> List[DocumentChunk]:
        """
        一级标题切分 + 二级长度及代码块感知递归切分
        """
        lines = text.splitlines()
        sections: List[Dict[str, Any]] = []
        current_header = "正文"
        current_lines: List[str] = []

        for line in lines:
            header_match = re.match(r'^(#{1,4})\s+(.+)$', line)
            if header_match:
                if current_lines:
                    sections.append({
                        "header": current_header,
                        "content": "\n".join(current_lines).strip()
                    })
                    current_lines = []
                current_header = header_match.group(2).strip()
            current_lines.append(line)

        if current_lines:
            sections.append({
                "header": current_header,
                "content": "\n".join(current_lines).strip()
            })

        chunks: List[DocumentChunk] = []
        curr_id = start_id

        for sec in sections:
            header = sec["header"]
            content = sec["content"]
            if not content or len(content.strip()) < 20:
                continue

            sub_texts = cls._split_section_content(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for idx, st in enumerate(sub_texts):
                # 若一个章节被切分为多个切片，标题后添加分段标识
                sub_header = f"{header} (第{idx+1}部分)" if len(sub_texts) > 1 else header
                chunk = DocumentChunk(
                    doc_id=curr_id,
                    source_file=source_file,
                    header=sub_header,
                    content=st,
                )
                chunks.append(chunk)
                curr_id += 1

        return chunks
