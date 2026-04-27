"""湖南大学校园知识库 — Knowledge 卡入口（Chroma 向量检索版）

插卡时自动注册：
- 知识文档：校史、教学楼、生活指南
- 资源文件：校园地图等
- 向量索引：使用 ChromaDB 构建语义检索

拔卡时自动清除：
- 注销知识检索器
- 注销资源
- 删除 Chroma 向量数据目录
"""

from pathlib import Path
from typing import Dict, List, Optional

from zhixia.core.card_base import CardManifest, HostContext, KnowledgeCard
from zhixia.llm.rag.base import RAGContext, RAGRetriever
from zhixia.rag.chroma_store import ChromaStore
from zhixia.rag.document_loader import MarkdownSplitter


class FallbackKeywordRetriever(RAGRetriever):
    """关键词回退检索器（Chroma 未安装时使用）。"""

    def __init__(self, documents: Dict[str, str]) -> None:
        self.documents = documents

    @property
    def name(self) -> str:
        return "fallback_keyword"

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
        keywords = [w for w in query if len(w) >= 2]
        if not keywords:
            keywords = list(query)
        matches = sum(1 for k in keywords if k in content)
        return matches


class HNUCampusKnowledge(KnowledgeCard):
    """湖南大学校园知识卡（Chroma 向量检索版）。"""

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        super().__init__(manifest, card_root)
        self._chroma_store: Optional[ChromaStore] = None
        self._fallback_retriever: Optional[FallbackKeywordRetriever] = None

    def on_mount(self, host: HostContext) -> None:
        """插卡时：加载文档 -> 切分 -> 构建 Chroma 索引 -> 注册资源。"""
        # 1. 加载文档
        docs = self._load_documents()

        # 2. 切分文档
        splitter = MarkdownSplitter(chunk_size=400, chunk_overlap=50)
        all_chunks = []
        for title, text in docs.items():
            chunks = splitter.split_text(text, source=title)
            all_chunks.extend(chunks)

        # 3. 尝试构建 Chroma 向量索引
        chroma = self._build_chroma_index(all_chunks)
        if chroma and chroma.is_available():
            self._chroma_store = chroma
            host.knowledge_hub.register_retriever(self.name, chroma)
            logger_msg = "Chroma 向量索引"
        else:
            # 回退到关键词检索
            self._fallback_retriever = FallbackKeywordRetriever(docs)
            host.knowledge_hub.register_retriever(self.name, self._fallback_retriever)
            logger_msg = "关键词回退检索"

        # 4. 注册资源
        assets = self._scan_assets()
        host.knowledge_hub.register_assets(self.name, assets)

        print(f"[MOUNT] Knowledge 卡已插入: {self.display_name}")
        print(f"   文档: {list(docs.keys())}")
        print(f"   检索: {logger_msg} ({len(all_chunks)} chunks)")
        print(f"   资源: {list(assets.keys())}")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销知识 + 资源 + 删除 Chroma 向量数据。"""
        host.knowledge_hub.unregister_retriever(self.name)
        host.knowledge_hub.unregister_assets(self.name)

        # 删除 Chroma 向量数据（痕迹清除）
        if self._chroma_store is not None:
            self._chroma_store.delete()
            self._chroma_store = None

        print(f"[UNMOUNT] Knowledge 卡已拔出: {self.display_name}")

    def get_retriever(self) -> RAGRetriever:
        if self._chroma_store is not None:
            return self._chroma_store
        docs = self._load_documents()
        return FallbackKeywordRetriever(docs)

    def get_assets(self) -> Dict[str, Path]:
        return self._scan_assets()

    def _build_chroma_index(self, chunks) -> Optional[ChromaStore]:
        """构建 Chroma 向量索引。"""
        # 向量数据存储在卡片目录内的 .vectors/ 子目录中
        # 拔卡时整个目录会被删除，自然清除痕迹
        persist_dir = self.card_root / ".vectors"
        store = ChromaStore(
            persist_dir=persist_dir,
            collection_name=self.name,
            embedding_model="all-MiniLM-L6-v2",  # 轻量级多语言模型
            device="cpu",
        )
        if not store.is_available():
            return None
        try:
            store.build_index(chunks)
            return store
        except Exception as exc:
            print(f"   警告: Chroma 索引构建失败 ({exc})，回退到关键词检索")
            return None

    def _load_documents(self) -> Dict[str, str]:
        """加载 docs/ 目录下的所有 markdown 文档。"""
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
