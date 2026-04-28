"""RAG 检索器抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class RAGContext:
    chunks: List[str] = field(default_factory=list)
    source_description: str = ""
    sources: List[str] = field(default_factory=list)  # 每个 chunk 对应的来源名称


class RAGRetriever(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> RAGContext:
        """根据用户查询检索相关上下文。无结果时返回空 chunks。"""
