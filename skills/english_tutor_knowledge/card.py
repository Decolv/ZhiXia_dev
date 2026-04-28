"""英语考试知识卡

提供英语考试相关的知识检索功能，包括：
- 听力技巧与真题解析
- 写作模板与范文
- 词汇表与记忆方法
- 长难句分析与理解
"""

from pathlib import Path
from typing import Dict, List, Optional

from zhixia.core.card_base import CardManifest, HostContext, KnowledgeCard
from zhixia.llm.rag.base import RAGContext, RAGRetriever


class SimpleKeywordRetriever(RAGRetriever):
    """简单关键词检索器（示例实现）。

    生产环境建议替换为 ChromaStore 等向量检索器。
    """

    def __init__(self, documents: Dict[str, str]) -> None:
        self.documents = documents

    @property
    def name(self) -> str:
        return "simple_keyword"

    def retrieve(self, query: str, top_k: int = 3) -> RAGContext:
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) >= 2]
        if not keywords:
            keywords = list(query_lower)

        scored = []
        for title, content in self.documents.items():
            score = sum(1 for k in keywords if k in content.lower())
            if score > 0:
                scored.append((score, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [content for _, content in scored[:top_k]]
        return RAGContext(chunks=chunks)


class EnglishTutorKnowledge(KnowledgeCard):
    """英语考试知识卡。

    插卡时自动注册：
    - 知识文档：docs/ 目录下的所有 Markdown 文件
    - 检索器：关键词检索（可升级为向量检索）
    """

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        super().__init__(manifest, card_root)
        self._retriever: Optional[SimpleKeywordRetriever] = None

    def on_mount(self, host: HostContext) -> None:
        """插卡时：加载文档 -> 构建检索器 -> 注册到主机。"""
        docs = self._load_documents()
        self._retriever = SimpleKeywordRetriever(docs)
        host.knowledge_hub.register_retriever(self.name, self._retriever)

        print(f"[MOUNT] 英语考试知识卡已插入: {self.display_name}")
        print(f"   文档: {list(docs.keys())}")
        print(f"   检索: 关键词检索 ({len(docs)} 篇文档)")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销知识检索器。"""
        host.knowledge_hub.unregister_retriever(self.name)
        self._retriever = None
        print(f"[UNMOUNT] 英语考试知识卡已拔出: {self.display_name}")

    def get_retriever(self) -> RAGRetriever:
        if self._retriever is None:
            docs = self._load_documents()
            return SimpleKeywordRetriever(docs)
        return self._retriever

    def get_assets(self) -> Dict[str, Path]:
        """返回资源映射。"""
        return {}

    def _load_documents(self) -> Dict[str, str]:
        """加载 docs/ 目录及其子目录下的所有 Markdown 文档。"""
        docs_dir = self.card_root / "docs"
        documents = {}
        if not docs_dir.exists() or not docs_dir.is_dir():
            return documents

        for doc_path in sorted(docs_dir.rglob("*.md")):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    relative_path = doc_path.relative_to(docs_dir)
                    documents[str(relative_path.with_suffix(""))] = f.read()
            except Exception as exc:
                print(f"   警告: 无法读取文档 {doc_path}: {exc}")

        return documents
