"""湖南大学校园知识库 — Knowledge 卡入口

插卡时自动注册：
- 知识文档：校史、教学楼、生活指南
- 资源文件：校园地图等
"""

from pathlib import Path
from typing import Dict

from zhixia.core.card_base import CardManifest, HostContext, KnowledgeCard
from zhixia.llm.rag.base import RAGContext, RAGRetriever


class SimpleKeywordRetriever(RAGRetriever):
    """基于关键词匹配的简单检索器（MVP 版本）。

    未来可替换为向量检索（FAISS/Chroma）。
    """

    def __init__(self, documents: Dict[str, str]) -> None:
        self.documents = documents

    @property
    def name(self) -> str:
        return "simple_keyword"

    def retrieve(self, query: str, top_k: int = 3) -> RAGContext:
        query_lower = query.lower()
        scores = []
        for title, content in self.documents.items():
            score = self._score(query_lower, content.lower())
            scores.append((score, title, content))

        scores.sort(key=lambda x: x[0], reverse=True)
        chunks = [content for score, _, content in scores[:top_k] if score > 0]
        return RAGContext(chunks=chunks)

    def _score(self, query: str, content: str) -> float:
        """简单评分：匹配关键词数量。"""
        keywords = [w for w in query if len(w) >= 2]
        if not keywords:
            keywords = list(query)
        matches = sum(1 for k in keywords if k in content)
        return matches


class HNUCampusKnowledge(KnowledgeCard):
    """湖南大学校园知识卡。"""

    def on_mount(self, host: HostContext) -> None:
        """插卡时：加载知识文档 + 注册资源。"""
        # 加载文档
        docs = self._load_documents()
        retriever = SimpleKeywordRetriever(docs)
        host.knowledge_hub.register_retriever(self.name, retriever)

        # 注册资源
        assets = self._scan_assets()
        host.knowledge_hub.register_assets(self.name, assets)

        print(f"[MOUNT] Knowledge 卡已插入: {self.display_name}")
        print(f"   文档: {list(docs.keys())}")
        print(f"   资源: {list(assets.keys())}")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销知识 + 资源。"""
        host.knowledge_hub.unregister_retriever(self.name)
        host.knowledge_hub.unregister_assets(self.name)
        print(f"[UNMOUNT] Knowledge 卡已拔出: {self.display_name}")

    def get_retriever(self) -> RAGRetriever:
        docs = self._load_documents()
        return SimpleKeywordRetriever(docs)

    def get_assets(self) -> Dict[str, Path]:
        return self._scan_assets()

    def _load_documents(self) -> Dict[str, str]:
        """加载 docs/ 目录下的所有 markdown 文档。"""
        docs_dir = self.card_root / "docs"
        documents = {}
        if not docs_dir.exists():
            return documents

        for doc_path in sorted(docs_dir.glob("*.md")):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    documents[doc_path.stem] = f.read()
            except Exception as exc:
                print(f"   警告: 无法读取文档 {doc_path}: {exc}")
        return documents

    def _scan_assets(self) -> Dict[str, Path]:
        """扫描 maps/ 和 schemes/ 目录下的资源文件。"""
        assets = {}

        maps_dir = self.card_root / "maps"
        if maps_dir.exists():
            for f in maps_dir.iterdir():
                if f.is_file():
                    assets[f"map_{f.stem}"] = f

        schemes_dir = self.card_root / "schemes"
        if schemes_dir.exists():
            for f in schemes_dir.iterdir():
                if f.is_file():
                    assets[f"scheme_{f.stem}"] = f

        return assets
