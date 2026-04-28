"""插卡系统综合测试 —— 验证热插拔、痕迹清除、Agent 动态组装、多卡搭配

用法:
    python test_card_system.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zhixia.agent.tool import ToolRegistry
from zhixia.config.settings import AppSettings
from zhixia.core.card_base import HostContext, KnowledgeHub, PersonaHolder
from zhixia.core.card_loader import CardLoader
from zhixia.core.host_orchestrator import HostOrchestrator
from zhixia.llm.base import LLMMessage
from zhixia.llm.rag.null_retriever import NullRAGRetriever

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


def test_slot_watcher():
    """测试槽位监视器：检测插卡/拔卡。"""
    print("\n" + "=" * 60)
    print("测试 1: SlotWatcher 变化检测")
    print("-" * 60)

    from zhixia.core.card_loader import SlotWatcher

    with tempfile.TemporaryDirectory() as tmpdir:
        slot = Path(tmpdir)
        watcher = SlotWatcher(slot, "skill")

        # 初始为空
        changed, card = watcher.detect_change()
        _assert("初始为空", changed and card is None)

        # 模拟插卡
        (slot / "manifest.json").write_text(
            '{"name":"test","display_name":"Test","version":"1.0","type":"skill"}'
        )
        changed, card = watcher.detect_change()
        _assert("检测到插卡", changed and card == slot)
        watcher.update_signature(card)

        # 再次检测（无变化）
        changed, card = watcher.detect_change()
        _assert("无变化不触发", not changed)

        # 模拟拔卡
        (slot / "manifest.json").unlink()
        changed, card = watcher.detect_change()
        _assert("检测到拔卡", changed and card is None)


def test_host_context():
    """测试主机上下文的组件。"""
    print("\n" + "=" * 60)
    print("测试 2: HostContext 组件")
    print("-" * 60)

    # PersonaHolder
    holder = PersonaHolder("基础人设")
    _assert("基础人设", holder.current_persona == "基础人设")

    holder.set_overlay("叠加人设", "test_card")
    _assert("叠加后人设", "叠加人设" in holder.current_persona)

    holder.clear_overlay()
    _assert("清除后恢复", holder.current_persona == "基础人设")

    # KnowledgeHub
    hub = KnowledgeHub()
    hub.register_retriever("test", NullRAGRetriever())
    _assert("注册检索器", len(hub._retrievers) == 1)

    hub.unregister_retriever("test")
    _assert("注销检索器", len(hub._retrievers) == 0)

    hub.register_assets("test", {"map": Path("/tmp/map.png")})
    _assert("注册资源", hub.get_asset("map") == Path("/tmp/map.png"))

    hub.clear_all()
    _assert("清空后", hub.get_asset("map") is None)


def test_card_mount_unmount():
    """测试完整插卡/拔卡流程（使用模板卡）。"""
    print("\n" + "=" * 60)
    print("测试 3: CardLoader 插卡/拔卡 + 痕迹清除")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        slot_a = tmpdir / "slot_a"
        slot_b = tmpdir / "slot_b"
        skill_src = _PROJECT_ROOT / "templates" / "skill_card_template"
        knowledge_src = _PROJECT_ROOT / "templates" / "knowledge_card_template"

        # 复制模板卡到槽位
        shutil.copytree(skill_src, slot_a / "skill_template")
        shutil.copytree(knowledge_src, slot_b / "knowledge_template")

        # 构建主机上下文
        tool_registry = ToolRegistry()
        persona_holder = PersonaHolder("基础人设")
        knowledge_hub = KnowledgeHub()
        host = HostContext(
            tool_registry=tool_registry,
            persona_holder=persona_holder,
            knowledge_hub=knowledge_hub,
        )

        slots = {
            "slot_a": (slot_a, None),
            "slot_b": (slot_b, None),
        }
        loader = CardLoader(slots, host)

        # 扫描并挂载
        changes = loader.scan_and_sync()
        _assert("Skill 卡挂载", "mounted:skill_template" in changes.values())
        _assert("Knowledge 卡挂载", "mounted:knowledge_template" in changes.values())

        # 验证工具已注册
        tools = [t.name for t in host.tool_registry.list_tools()]
        _assert("example_tool 已注册", "example_tool" in tools)

        # 验证人设已叠加
        _assert("人设已叠加", "示例助手" in host.persona_holder.current_persona)

        # 验证知识已注册
        _assert("知识检索器已注册", "knowledge_template" in host.knowledge_hub._retrievers)

        # 拔卡
        loader.force_unmount_all()
        _assert("工具已清除", len(host.tool_registry.list_tools()) == 0)
        _assert("人设已恢复", host.persona_holder.current_persona == "基础人设")
        _assert("知识已清除", len(host.knowledge_hub._retrievers) == 0)


def test_host_orchestrator_build_agent():
    """测试 HostOrchestrator 动态组装 Agent（使用模板卡）。"""
    print("\n" + "=" * 60)
    print("测试 4: HostOrchestrator Agent 动态组装")
    print("-" * 60)

    from zhixia.llm.base import LLMEngine

    class MockLLM(LLMEngine):
        @property
        def name(self):
            return "mock"

        def chat(self, messages, max_new_tokens=32):
            return "Mock回答"

        def set_system_prompt(self, prompt):
            pass

    config = AppSettings()
    config.agent.enabled = True
    config.agent.engine = "react"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        slot_a = tmpdir / "slot_a"
        skill_src = _PROJECT_ROOT / "templates" / "skill_card_template"
        shutil.copytree(skill_src, slot_a / "skill_template")

        orchestrator = HostOrchestrator(
            config=config,
            asr_engine=None,  # 测试中不需要
            llm_engine=MockLLM(),
            tts_engine=None,
            audio_player=None,
            slot_paths={"slot_a": (slot_a, None)},
        )
        orchestrator.initialize_slots()

        # 验证 Agent 已构建
        agent = orchestrator._get_or_build_agent()
        _assert("Agent 已构建", agent is not None)
        _assert("Agent 类型正确", "ReActAgent" in agent.agent.name)

        # 验证工具存在
        tools = orchestrator.host_context.tool_registry.list_tools()
        tool_names = [t.name for t in tools]
        _assert("Agent 包含 example_tool", "example_tool" in tool_names)

        # 拔卡后 Agent 应失效
        orchestrator.card_loader.force_unmount_all()
        orchestrator._invalidate_agent_cache()
        agent2 = orchestrator._get_or_build_agent()
        _assert("拔卡后 Agent 为空", agent2 is None)


def test_knowledge_retrieval():
    """测试知识检索功能（使用模板知识卡）。"""
    print("\n" + "=" * 60)
    print("测试 5: Knowledge 卡知识检索")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        slot_b = tmpdir / "slot_b"
        knowledge_src = _PROJECT_ROOT / "templates" / "knowledge_card_template"
        shutil.copytree(knowledge_src, slot_b / "knowledge_template")

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

        # 检索测试（模板文档中有 "Markdown"、"最佳实践" 等关键词）
        results = knowledge_hub.retrieve("Markdown 格式", top_k=2)
        _assert("检索有结果", len(results.chunks) > 0)
        _assert("结果包含 Markdown", any("Markdown" in r for r in results.chunks))

        results2 = knowledge_hub.retrieve("最佳实践", top_k=2)
        _assert("检索到相关内容", len(results2.chunks) > 0)

        loader.force_unmount_all()


def test_multi_knowledge_cards():
    """测试多张知识卡搭配一个技能卡（解耦核心验证）。"""
    print("\n" + "=" * 60)
    print("测试 6: 多张 Knowledge 卡 + 单张 Skill 卡搭配")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        slot_a = tmpdir / "slot_a"
        slot_b = tmpdir / "slot_b"
        slot_c = tmpdir / "slot_c"

        skill_src = _PROJECT_ROOT / "templates" / "skill_card_template"
        knowledge_src = _PROJECT_ROOT / "templates" / "knowledge_card_template"

        shutil.copytree(skill_src, slot_a / "skill_template")
        # 插两张知识卡到不同槽位
        shutil.copytree(knowledge_src, slot_b / "knowledge_b")
        shutil.copytree(knowledge_src, slot_c / "knowledge_c")
        # 修改 manifest 中的 name，使每张卡唯一（便于验证）
        manifest_b = slot_b / "knowledge_b" / "manifest.json"
        manifest_b.write_text(
            '{"name":"knowledge_b","display_name":"Knowledge B","version":"1.0","type":"knowledge"}',
            encoding="utf-8",
        )
        manifest_c = slot_c / "knowledge_c" / "manifest.json"
        manifest_c.write_text(
            '{"name":"knowledge_c","display_name":"Knowledge C","version":"1.0","type":"knowledge"}',
            encoding="utf-8",
        )

        tool_registry = ToolRegistry()
        persona_holder = PersonaHolder("基础人设")
        knowledge_hub = KnowledgeHub()
        host = HostContext(
            tool_registry=tool_registry,
            persona_holder=persona_holder,
            knowledge_hub=knowledge_hub,
        )

        slots = {
            "slot_a": (slot_a, None),
            "slot_b": (slot_b, None),
            "slot_c": (slot_c, None),
        }
        loader = CardLoader(slots, host)
        changes = loader.scan_and_sync()

        # 验证 3 张卡都挂载成功
        mounted = loader.get_mounted_names()
        _assert("Skill 卡已挂载", "skill_template" in mounted)
        _assert("Knowledge B 已挂载", "knowledge_b" in mounted)
        _assert("Knowledge C 已挂载", "knowledge_c" in mounted)
        _assert("共挂载 3 张卡", len(mounted) == 3)

        # 验证知识检索合并了多张卡的结果
        results = knowledge_hub.retrieve("Markdown", top_k=5)
        _assert("合并检索有结果", len(results.chunks) > 0)
        # 来源应该包含多张卡
        unique_sources = set(results.sources)
        _assert("结果来自多个知识卡", len(unique_sources) >= 2)

        # 拔卡后全部清除
        loader.force_unmount_all()
        _assert("所有知识已清除", len(host.knowledge_hub._retrievers) == 0)
        _assert("所有工具已清除", len(host.tool_registry.list_tools()) == 0)


if __name__ == "__main__":
    test_slot_watcher()
    test_host_context()
    test_card_mount_unmount()
    test_host_orchestrator_build_agent()
    test_knowledge_retrieval()
    test_multi_knowledge_cards()

    print("\n" + "=" * 60)
    print(f"测试完成: [PASS] {PASS} 通过, [FAIL] {FAIL} 失败")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
