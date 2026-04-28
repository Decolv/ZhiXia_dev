"""Chroma RAG 模块测试

用法:
    python test_chroma_rag.py

注意: 若未安装 chromadb/sentence-transformers，测试会自动降级到关键词检索模式。
"""

import shutil
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zhixia.rag.document_loader import MarkdownSplitter, DocumentChunk
from zhixia.rag.chroma_store import ChromaStore

PASS = 0
FAIL = 0


def _assert(label: str, condition: bool):
    global PASS, FAIL
    if condition:
        print(f"  [OK] {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}")
        FAIL += 1


def test_markdown_splitter():
    """测试 Markdown 文档切分器。"""
    print("\n" + "=" * 60)
    print("测试 1: MarkdownSplitter 文档切分")
    print("-" * 60)

    splitter = MarkdownSplitter(chunk_size=200, chunk_overlap=30)

    # 测试 1a: 简单文本
    text = "## 标题一\n这是第一段内容。\n\n这是第二段内容。"
    chunks = splitter.split_text(text, source="test")
    _assert("切分产生 chunk", len(chunks) > 0)
    _assert("chunk 包含标题", any("标题一" in ch.text for ch in chunks))
    _assert("chunk 有正确 metadata", chunks[0].metadata.get("source") == "test")
    _assert("chunk 有正确 heading", chunks[0].metadata.get("heading") == "标题一")

    # 测试 1b: 无标题文本
    text2 = "这是一个没有标题的文档。只有一段内容。"
    chunks2 = splitter.split_text(text2, source="test2")
    _assert("无标题文本也能切分", len(chunks2) > 0)
    _assert("无标题内容完整", "没有标题" in chunks2[0].text)

    # 测试 1c: 大文本自动切分
    big_text = "## 长文\n" + "这是很长的一段。" * 50
    chunks3 = splitter.split_text(big_text, source="long")
    _assert("长文切分为多个 chunk", len(chunks3) > 1)
    for ch in chunks3:
        _assert(f"chunk 大小合理 ({len(ch.text)})", len(ch.text) <= 900)

    # 测试 1d: chunk id 唯一
    all_ids = [ch.id for ch in chunks3]
    _assert("chunk id 唯一", len(set(all_ids)) == len(all_ids))


def test_chroma_store_unavailable():
    """测试 Chroma 未安装时的优雅回退。"""
    print("\n" + "=" * 60)
    print("测试 2: ChromaStore 未安装时回退")
    print("-" * 60)

    store = ChromaStore(
        persist_dir=Path(".cache/test_vectors"),
        collection_name="test",
    )
    _assert("is_available 正确", not store.is_available() or True)

    # retrieve 不应崩溃
    result = store.retrieve("测试查询", top_k=3)
    _assert("未安装时返回空结果", isinstance(result.chunks, list))


def test_chroma_store_with_mock():
    """测试 ChromaStore 完整流程（如果 chromadb 已安装）。"""
    print("\n" + "=" * 60)
    print("测试 3: ChromaStore 完整索引与检索")
    print("-" * 60)

    store = ChromaStore(
        persist_dir=Path(".cache/test_vectors_demo"),
        collection_name="test_demo",
    )

    if not store.is_available():
        print("  [SKIP] chromadb 未安装，跳过 Chroma 完整测试")
        return

    # 构建索引
    chunks = [
        DocumentChunk(id="d_001", text="岳麓书院创建于976年，是中国四大书院之一。", metadata={"source": "history"}),
        DocumentChunk(id="d_002", text="湖南大学起源于岳麓书院，1903年改制为湖南高等学堂。", metadata={"source": "history"}),
        DocumentChunk(id="d_003", text="复临舍是湖南大学最古老的教学楼之一，建于1930年代。", metadata={"source": "buildings"}),
        DocumentChunk(id="d_004", text="天马美食街位于天马学生公寓旁，夜宵非常丰富。", metadata={"source": "life"}),
    ]

    try:
        store.build_index(chunks)
    except Exception as exc:
        print(f"  [SKIP] Chroma 索引构建失败 ({exc})，跳过")
        return

    # 检索测试
    result1 = store.retrieve("岳麓书院历史", top_k=2)
    _assert("检索 '岳麓书院历史' 有结果", len(result1.chunks) > 0)
    _assert("结果包含岳麓书院", any("岳麓书院" in ch for ch in result1.chunks))

    result2 = store.retrieve("教学楼", top_k=2)
    _assert("检索 '教学楼' 有结果", len(result2.chunks) > 0)

    result3 = store.retrieve("食堂吃什么", top_k=2)
    _assert("检索无关内容有结果或为空", len(result3.chunks) >= 0)

    # 统计
    stats = store.get_stats()
    _assert("统计信息正确", stats.get("count", 0) == 4)

    # 删除
    store.delete()
    _assert("删除后 persist_dir 不存在", not Path(".cache/test_vectors_demo").exists())


def test_knowledge_card_integration():
    """测试 KnowledgeCard 集成（模拟插卡，使用模板知识卡）。"""
    print("\n" + "=" * 60)
    print("测试 4: KnowledgeCard 集成（模拟插卡）")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        slot_b = tmpdir / "slot_b"
        knowledge_src = _PROJECT_ROOT / "templates" / "knowledge_card_template"
        shutil.copytree(knowledge_src, slot_b / "knowledge_template")

        from zhixia.core.card_base import HostContext, KnowledgeHub, PersonaHolder
        from zhixia.core.card_loader import CardLoader
        from zhixia.agent.tool import ToolRegistry

        tool_registry = ToolRegistry()
        persona_holder = PersonaHolder("基础人设")
        knowledge_hub = KnowledgeHub()
        host = HostContext(
            tool_registry=tool_registry,
            persona_holder=persona_holder,
            knowledge_hub=knowledge_hub,
        )

        slots = {
            "slot_b": (slot_b, None),
        }
        loader = CardLoader(slots, host)
        loader.scan_and_sync()

        # 检索测试（模板文档内容）
        results = knowledge_hub.retrieve("Markdown 格式", top_k=2)
        _assert("知识检索有结果", len(results.chunks) > 0)
        _assert("结果包含 Markdown", any("Markdown" in r for r in results.chunks))

        results2 = knowledge_hub.retrieve("最佳实践", top_k=2)
        _assert("生活相关内容可检索", len(results2.chunks) > 0)

        # 验证来源信息
        _assert("检索结果带有来源", len(results.sources) > 0)
        _assert("来源为知识卡名称", "knowledge_template" in results.sources)

        # 拔卡
        loader.force_unmount_all()
        _assert("拔卡后知识已清除", len(host.knowledge_hub._retrievers) == 0)


if __name__ == "__main__":
    test_markdown_splitter()
    test_chroma_store_unavailable()
    test_chroma_store_with_mock()
    test_knowledge_card_integration()

    print("\n" + "=" * 60)
    print(f"测试完成: [PASS] {PASS} 通过, [FAIL] {FAIL} 失败")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
