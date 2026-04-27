"""Chroma 向量存储封装

对 ChromaDB 的轻量级封装，支持：
- 本地持久化（SQLite 后端）
- 自定义嵌入模型（中文轻量级模型）
- 懒加载（首次查询时才加载模型）
- 优雅回退（未安装 chromadb 时降级到关键词检索）

依赖:
    pip install chromadb sentence-transformers

可选中文模型:
    - "shibing624/text2vec-base-chinese" (约100MB，推荐)
    - "BAAI/bge-small-zh" (约30MB，更轻量)
    - 默认 "all-MiniLM-L6-v2" (约20MB，多语言)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from zhixia.llm.rag.base import RAGContext, RAGRetriever
from zhixia.rag.document_loader import DocumentChunk

logger = logging.getLogger(__name__)

# 延迟导入 chromadb，未安装时不报错
_chromadb = None
_sentence_transformers = None


def _ensure_chroma():
    """确保 chromadb 可用，返回模块或 None。"""
    global _chromadb
    if _chromadb is not None:
        return _chromadb
    try:
        import chromadb

        _chromadb = chromadb
        return _chromadb
    except ImportError:
        logger.warning("chromadb 未安装，向量检索不可用。安装方式: pip install chromadb")
        return None


def _ensure_st():
    """确保 sentence-transformers 可用，返回模块或 None。"""
    global _sentence_transformers
    if _sentence_transformers is not None:
        return _sentence_transformers
    try:
        import sentence_transformers

        _sentence_transformers = sentence_transformers
        return _sentence_transformers
    except ImportError:
        logger.warning("sentence-transformers 未安装。安装方式: pip install sentence-transformers")
        return None


class _EmbeddingFunction:
    """Chroma 自定义嵌入函数包装器。

    兼容 Chroma 的 EmbeddingFunction 协议，内部使用 sentence-transformers。
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        st = _ensure_st()
        if st is None:
            raise RuntimeError("sentence-transformers 未安装，无法加载嵌入模型")
        logger.info("加载嵌入模型: %s (device=%s)", self.model_name, self.device)
        self._model = st.SentenceTransformer(self.model_name, device=self.device)
        logger.info("嵌入模型加载完成")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()


class ChromaStore(RAGRetriever):
    """Chroma 向量存储。

    Args:
        persist_dir: Chroma 持久化目录（建议放在卡片目录内）
        collection_name: collection 名称
        embedding_model: 嵌入模型名称（默认 all-MiniLM-L6-v2）
        device: 运行设备（cpu / cuda）
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.device = device

        self._client = None
        self._collection = None
        self._embed_fn = None
        self._available = _ensure_chroma() is not None

    @property
    def name(self) -> str:
        return f"chroma:{self.collection_name}"

    def is_available(self) -> bool:
        return self._available

    def _ensure_client(self):
        """延迟初始化 Chroma 客户端。"""
        if self._client is not None:
            return
        chroma = _ensure_chroma()
        if chroma is None:
            raise RuntimeError("chromadb 未安装")

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chroma.PersistentClient(path=str(self.persist_dir))
        self._embed_fn = _EmbeddingFunction(self.embedding_model, self.device)

        # 获取或创建 collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embed_fn,
        )
        logger.info(
            "Chroma collection 就绪: %s (persist=%s, model=%s)",
            self.collection_name,
            self.persist_dir,
            self.embedding_model,
        )

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """构建向量索引。

        如果 collection 中已有数据，先清空再重建。
        """
        if not chunks:
            logger.warning("无文档可索引")
            return

        self._ensure_client()
        assert self._collection is not None

        # 清空旧数据
        existing = self._collection.count()
        if existing > 0:
            self._collection.delete(where={"source": {"$ne": ""}})
            logger.info("清空旧索引: %d 条", existing)

        # 分批添加（避免单次过大）
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._collection.add(
                ids=[ch.id for ch in batch],
                documents=[ch.text for ch in batch],
                metadatas=[ch.metadata for ch in batch],
            )
            logger.debug("索引批次 %d-%d", i, i + len(batch))

        logger.info(
            "索引构建完成: %d chunks -> %s",
            len(chunks),
            self.collection_name,
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """增量添加文档（不清空旧数据）。"""
        if not chunks:
            return
        self._ensure_client()
        assert self._collection is not None
        self._collection.add(
            ids=[ch.id for ch in chunks],
            documents=[ch.text for ch in chunks],
            metadatas=[ch.metadata for ch in chunks],
        )

    def retrieve(self, query: str, top_k: int = 3) -> RAGContext:
        """向量检索。

        Returns:
            RAGContext，包含最相关的文本片段。
        """
        if not self._available:
            logger.warning("Chroma 不可用，返回空结果")
            return RAGContext(chunks=[])

        try:
            self._ensure_client()
        except Exception as exc:
            logger.error("Chroma 初始化失败: %s", exc)
            return RAGContext(chunks=[])

        assert self._collection is not None
        count = self._collection.count()
        if count == 0:
            logger.debug("Collection 为空，无结果")
            return RAGContext(chunks=[])

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
        except Exception as exc:
            logger.error("Chroma 查询失败: %s", exc)
            return RAGContext(chunks=[])

        chunks = []
        if results and results.get("documents"):
            for doc_list in results["documents"]:
                if doc_list:
                    chunks.extend([d for d in doc_list if d])

        # 去重并保持顺序
        seen = set()
        unique = []
        for ch in chunks:
            if ch not in seen:
                seen.add(ch)
                unique.append(ch)

        logger.debug("Chroma 检索 '%s...' -> %d 条结果", query[:20], len(unique))
        return RAGContext(chunks=unique)

    def delete(self) -> None:
        """删除 collection 和持久化数据。"""
        if self._collection is not None and self._client is not None:
            try:
                self._client.delete_collection(name=self.collection_name)
            except Exception as exc:
                logger.warning("删除 collection 失败: %s", exc)
            self._collection = None

        # 删除持久化目录
        if self.persist_dir.exists():
            try:
                shutil.rmtree(self.persist_dir)
                logger.info("删除向量数据目录: %s", self.persist_dir)
            except Exception as exc:
                logger.warning("删除目录失败: %s", exc)

        self._client = None

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息。"""
        if self._collection is None:
            return {"count": 0, "model": self.embedding_model}
        return {
            "count": self._collection.count(),
            "model": self.embedding_model,
            "persist_dir": str(self.persist_dir),
        }
