"""ZhiXia RAG 模块 — 基于 Chroma 向量数据库的知识检索

设计目标：
1. 轻量：RK3588 可运行，支持轻量级中文嵌入模型
2. 插卡友好：向量数据随卡片目录存储，拔卡即清除
3. 延迟加载：chromadb 和嵌入模型首次使用时才加载
4. 优雅回退：未安装依赖时自动降级到关键词检索

核心组件：
    ChromaStore      — Chroma 向量存储封装
    MarkdownSplitter — Markdown 文档智能切分
    Embedder         — 嵌入模型懒加载管理器

使用示例：
    from zhixia.rag import ChromaStore, MarkdownSplitter

    splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_file(Path("history.md"))

    store = ChromaStore(
        persist_dir=Path(".cache/vectors/my_knowledge"),
        collection_name="my_knowledge",
    )
    store.build_index(docs)
    results = store.query("知识库查询示例", top_k=3)
"""

from zhixia.rag.chroma_store import ChromaStore
from zhixia.rag.document_loader import MarkdownSplitter

__all__ = ["ChromaStore", "MarkdownSplitter"]
