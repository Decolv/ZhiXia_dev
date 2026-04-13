"""空 RAG 检索器（默认，不执行任何检索）"""

import logging

from zhixia.llm.rag.base import RAGContext, RAGRetriever

logger = logging.getLogger(__name__)


class NullRAGRetriever(RAGRetriever):

    @property
    def name(self) -> str:
        return "null"

    def retrieve(self, query: str, top_k: int = 3) -> RAGContext:
        return RAGContext(chunks=[], source_description="none")
