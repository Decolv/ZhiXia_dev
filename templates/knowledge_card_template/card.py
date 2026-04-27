"""Knowledge 卡模板 —— 最小可运行示例

展示知识检索规范：
1. 加载 docs/ 目录下的 Markdown 文档
2. 构建简单的关键词检索器
3. 注册到主机的 KnowledgeHub
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


class KnowledgeTemplate(KnowledgeCard):
    """知识卡模板。

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

        print(f"[MOUNT] Knowledge 卡已插入: {self.display_name}")
        print(f"   文档: {list(docs.keys())}")
        print(f"   检索: 关键词检索 ({len(docs)} 篇文档)")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销知识检索器。"""
        host.knowledge_hub.unregister_retriever(self.name)
        self._retriever = None
        print(f"[UNMOUNT] Knowledge 卡已拔出: {self.display_name}")

    def get_retriever(self) -> RAGRetriever:
        if self._retriever is None:
            docs = self._load_documents()
            return SimpleKeywordRetriever(docs)
        return self._retriever

    def get_assets(self) -> Dict[str, Path]:
        """返回资源映射。可扩展为扫描 maps/、images/ 等目录。"""
        return {}

    def _load_documents(self) -> Dict[str, str]:
        """加载 docs/ 目录下的所有 Markdown 文档。"""
        docs_dir = self.card_root / "docs"
        documents = {}
        if not docs_dir.exists() or not docs_dir.is_dir():
            return documents

        for doc_path in sorted(docs_dir.glob("*.md")):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    documents[doc_path.stem] = f.read()
            except Exception as exc:
                print(f"   警告: 无法读取文档 {doc_path}: {exc}")

        return documents
