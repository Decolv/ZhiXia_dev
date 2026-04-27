"""文档加载与切分器

支持 Markdown 文件按标题和段落智能切分，生成适合向量检索的 chunk。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """切分后的文档片段。"""

    id: str
    text: str
    metadata: Dict[str, str]


class MarkdownSplitter:
    """Markdown 文档智能切分器。

    切分策略：
    1. 按标题（# ## ###）切分，保留标题上下文
    2. 大段落按句子/语义边界进一步切分
    3. 每个 chunk 保留来源文件名和标题层级信息

    Args:
        chunk_size: 目标 chunk 字符数（默认 400）
        chunk_overlap: 相邻 chunk 的重叠字符数（默认 50）
        max_chunk_size: 单个 chunk 的最大字符数（默认 800）
    """

    # Markdown 标题正则
    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    # 段落分隔（空行）
    _PARA_SPLIT_RE = re.compile(r"\n\s*\n")

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        max_chunk_size: int = 800,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size

    def split_file(self, file_path: Path, source_name: Optional[str] = None) -> List[DocumentChunk]:
        """切分单个 Markdown 文件。

        Returns:
            DocumentChunk 列表，按文档顺序排列。
        """
        if not file_path.exists():
            logger.warning("文件不存在: %s", file_path)
            return []

        source = source_name or file_path.stem
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("读取文件失败 %s: %s", file_path, exc)
            return []

        return self.split_text(text, source=source)

    def split_text(self, text: str, source: str = "unknown") -> List[DocumentChunk]:
        """切分文本内容。

        策略：
        1. 先按标题切分为"章节"
        2. 每个章节如果过大，再按段落/句子切分
        3. 确保每个 chunk 保留章节标题上下文
        """
        text = text.strip()
        if not text:
            return []

        # 提取标题位置
        headings = list(self._HEADING_RE.finditer(text))

        # 如果没有标题，整体作为一节处理
        if not headings:
            sections = [("", text)]
        else:
            sections = []
            last_end = 0
            for i, m in enumerate(headings):
                level = len(m.group(1))
                title = m.group(2).strip()
                start = m.start()
                end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
                section_text = text[start:end].strip()
                if section_text:
                    sections.append((title, section_text))
                last_end = end

        # 对每个章节进一步切分
        chunks: List[DocumentChunk] = []
        chunk_idx = 0
        for title, section_text in sections:
            section_chunks = self._split_section(section_text, title, source)
            for ch in section_chunks:
                ch.id = f"{source}_{chunk_idx:04d}"
                chunk_idx += 1
            chunks.extend(section_chunks)

        logger.info("文档 '%s' 切分为 %d 个 chunk", source, len(chunks))
        return chunks

    def _split_section(
        self, text: str, heading: str, source: str
    ) -> List[DocumentChunk]:
        """将单个章节切分为适当大小的 chunk。

        如果章节文本较短（< chunk_size），直接作为一个 chunk。
        如果较长，按段落切分，并尽量保持语义完整。
        超长段落会在段落内部按字符边界进一步切分。
        """
        # 移除标题行本身（如果还在）
        text = self._HEADING_RE.sub("", text).strip()
        if not text:
            if heading:
                # 只有标题没有内容，也保留标题作为索引
                return [
                    DocumentChunk(
                        id="",
                        text=heading,
                        metadata={"source": source, "heading": heading, "level": "0"},
                    )
                ]
            return []

        # 构建上下文前缀（标题）
        context_prefix = f"{heading}\n" if heading else ""

        # 如果整体不大，直接作为一个 chunk
        total_len = len(context_prefix) + len(text)
        if total_len <= self.chunk_size:
            return [
                DocumentChunk(
                    id="",
                    text=f"{context_prefix}{text}",
                    metadata={"source": source, "heading": heading, "level": "1"},
                )
            ]

        # 按段落切分
        paragraphs = [p.strip() for p in self._PARA_SPLIT_RE.split(text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        # 滑动窗口合并段落为 chunk
        chunks: List[DocumentChunk] = []
        current_text = ""
        for para in paragraphs:
            # 如果单个段落过长，先内部切分
            if len(para) > self.chunk_size:
                # 先 flush 当前积累的文本
                if current_text:
                    chunks.append(
                        DocumentChunk(
                            id="",
                            text=f"{context_prefix}{current_text}".strip(),
                            metadata={"source": source, "heading": heading, "level": "2"},
                        )
                    )
                    current_text = ""
                # 将长段落按字符边界切分
                sub_chunks = self._split_long_text(para, self.chunk_size, self.chunk_overlap)
                for sub in sub_chunks:
                    chunks.append(
                        DocumentChunk(
                            id="",
                            text=f"{context_prefix}{sub}".strip() if context_prefix else sub,
                            metadata={"source": source, "heading": heading, "level": "3"},
                        )
                    )
                continue

            # 如果加入当前段落会超出 chunk_size，先结束当前 chunk
            projected = f"{context_prefix}{current_text}\n\n{para}".strip()
            if current_text and len(projected) > self.chunk_size:
                chunks.append(
                    DocumentChunk(
                        id="",
                        text=f"{context_prefix}{current_text}".strip(),
                        metadata={"source": source, "heading": heading, "level": "2"},
                    )
                )
                # 保留重叠部分（最后一句或最后一段）
                overlap = self._extract_overlap(current_text)
                current_text = f"{overlap}\n\n{para}".strip() if overlap else para
            else:
                if current_text:
                    current_text += f"\n\n{para}"
                else:
                    current_text = para

        # 最后一个 chunk
        if current_text:
            chunks.append(
                DocumentChunk(
                    id="",
                    text=f"{context_prefix}{current_text}".strip(),
                    metadata={"source": source, "heading": heading, "level": "2"},
                )
            )

        return chunks

    def _split_long_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """按字符边界切分长文本（句子优先）。"""
        result: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # 尽量在句子边界截断
            if end < len(text):
                for sep in ["\n\n", "。", "；", ";", "\n"]:
                    pos = text.rfind(sep, start, end)
                    if pos > start + chunk_size // 2:
                        end = pos + len(sep)
                        break
            chunk_text = text[start:end].strip()
            if chunk_text:
                result.append(chunk_text)
            start = end - chunk_overlap if end < len(text) else end
        return result

    def _extract_overlap(self, text: str) -> str:
        """从文本末尾提取重叠部分（用于滑动窗口）。"""
        if len(text) <= self.chunk_overlap:
            return text
        # 尝试找到句子边界
        sentences = re.split(r'([。！？!?;；\.])', text)
        overlap = ""
        for part in reversed(sentences):
            if len(overlap) + len(part) > self.chunk_overlap:
                break
            overlap = part + overlap
        return overlap.strip()


class SimpleTextLoader:
    """简单文本文件加载器（非 Markdown）。"""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: Path, source_name: Optional[str] = None) -> List[DocumentChunk]:
        if not file_path.exists():
            return []
        source = source_name or file_path.stem
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            return []
        return self._split_text(text, source)

    def _split_text(self, text: str, source: str) -> List[DocumentChunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # 尽量在句子边界截断
            if end < len(text):
                for sep in ["\n\n", "。", "；", ";", "\n"]:
                    pos = text.rfind(sep, start, end)
                    if pos > start + self.chunk_size // 2:
                        end = pos + len(sep)
                        break
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        id=f"{source}_{idx:04d}",
                        text=chunk_text,
                        metadata={"source": source},
                    )
                )
                idx += 1
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks
