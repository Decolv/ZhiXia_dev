"""插卡系统综合测试 —— 验证热插拔、痕迹清除、Agent 动态组装

用法:
    python test_card_system.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
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
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        slot = Path(tmpdir)
        watcher = SlotWatcher(slot, "skill")

        # 初始为空
        changed, card = watcher.detect_change()
        _assert("初始为空", changed and card is None)

        # 模拟插卡
        (slot / "manifest.json").write_text('{"name":"test","display_name":"Test","version":"1.0","type":"skill"}')
        changed, card = watcher.detect_change()
        _assert("检测到插卡", changed and card == slot)

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
    _assert("叠加后人设", holder.current_persona == "叠加人设")

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
    """测试完整插卡/拔卡流程。"""
    print("\n" + "=" * 60)
    print("测试 3: CardLoader 插卡/拔卡 + 痕迹清除")
    print("-" * 60)

    # 先模拟插卡
    import subprocess
    subprocess.run([sys.executable, str(_PROJECT_ROOT / "mount_cards.py"),
                    "--skill", "skills/hnu_freshman",
                    "--knowledge", "knowledge/hnu_campus"],
                   check=True, capture_output=True)

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
        "skill": (_PROJECT_ROOT / "cards" / "slot_a", "skill"),
        "knowledge": (_PROJECT_ROOT / "cards" / "slot_b", "knowledge"),
    }
    loader = CardLoader(slots, host)

    # 扫描并挂载
    changes = loader.scan_and_sync()
    _assert("Skill 卡挂载", "mounted:hnu_freshman" in changes.values())
    _assert("Knowledge 卡挂载", "mounted:hnu_campus" in changes.values())

    # 验证工具已注册
    tools = [t.name for t in host.tool_registry.list_tools()]
    _assert("campus_navigate 已注册", "campus_navigate" in tools)
    _assert("query_major 已注册", "query_major" in tools)
    _assert("campus_life_guide 已注册", "campus_life_guide" in tools)

    # 验证人设已叠加
    _assert("人设已叠加", "湖小助" in host.persona_holder.current_persona)

    # 验证知识已注册
    _assert("知识检索器已注册", "hnu_campus" in host.knowledge_hub._retrievers)

    # 拔卡
    loader.force_unmount_all()
    _assert("工具已清除", len(host.tool_registry.list_tools()) == 0)
    _assert("人设已恢复", host.persona_holder.current_persona == "基础人设")
    _assert("知识已清除", len(host.knowledge_hub._retrievers) == 0)

    print("  清理测试槽位...")
    subprocess.run([sys.executable, str(_PROJECT_ROOT / "mount_cards.py"), "--eject"],
                   check=True, capture_output=True)


def test_host_orchestrator_build_agent():
    """测试 HostOrchestrator 动态组装 Agent。"""
    print("\n" + "=" * 60)
    print("测试 4: HostOrchestrator Agent 动态组装")
    print("-" * 60)

    from zhixia.llm.base import LLMEngine

    class MockLLM(LLMEngine):
        @property
        def name(self): return "mock"
        def chat(self, messages, max_new_tokens=32): return "Mock回答"
        def set_system_prompt(self, prompt): pass

    config = AppSettings()
    config.agent.enabled = True
    config.agent.engine = "react"

    # 先插卡
    import subprocess
    subprocess.run([sys.executable, str(_PROJECT_ROOT / "mount_cards.py"),
                    "--skill", "skills/hnu_freshman"],
                   check=True, capture_output=True)

    orchestrator = HostOrchestrator(
        config=config,
        asr_engine=None,  # 测试中不需要
        llm_engine=MockLLM(),
        tts_engine=None,
        audio_player=None,
    )
    orchestrator.initialize_slots()

    # 验证 Agent 已构建
    agent = orchestrator._get_or_build_agent()
    _assert("Agent 已构建", agent is not None)
    _assert("Agent 类型正确", "ReActAgent" in agent.agent.name)

    # 验证工具存在
    tools = orchestrator.host_context.tool_registry.list_tools()
    tool_names = [t.name for t in tools]
    _assert("Agent 包含 campus_navigate", "campus_navigate" in tool_names)

    # 拔卡后 Agent 应失效
    orchestrator.card_loader.force_unmount_all()
    orchestrator._invalidate_agent_cache()
    agent2 = orchestrator._get_or_build_agent()
    _assert("拔卡后 Agent 为空", agent2 is None)

    print("  清理测试槽位...")
    subprocess.run([sys.executable, str(_PROJECT_ROOT / "mount_cards.py"), "--eject"],
                   check=True, capture_output=True)


def test_knowledge_retrieval():
    """测试知识检索功能。"""
    print("\n" + "=" * 60)
    print("测试 5: Knowledge 卡知识检索")
    print("-" * 60)

    # 先插卡
    import subprocess
    subprocess.run([sys.executable, str(_PROJECT_ROOT / "mount_cards.py"),
                    "--knowledge", "knowledge/hnu_campus"],
                   check=True, capture_output=True)

    tool_registry = ToolRegistry()
    persona_holder = PersonaHolder("基础人设")
    knowledge_hub = KnowledgeHub()
    host = HostContext(
        tool_registry=tool_registry,
        persona_holder=persona_holder,
        knowledge_hub=knowledge_hub,
    )

    slots = {
        "knowledge": (_PROJECT_ROOT / "cards" / "slot_b", "knowledge"),
    }
    loader = CardLoader(slots, host)
    loader.scan_and_sync()

    # 检索测试
    results = knowledge_hub.retrieve("岳麓书院的历史", top_k=2)
    _assert("检索到校史内容", len(results) > 0)
    _assert("内容包含岳麓书院", any("岳麓书院" in r for r in results))

    results2 = knowledge_hub.retrieve("食堂在哪里", top_k=2)
    _assert("检索到生活指南", len(results2) > 0)

    loader.force_unmount_all()

    print("  清理测试槽位...")
    subprocess.run([sys.executable, str(_PROJECT_ROOT / "mount_cards.py"), "--eject"],
                   check=True, capture_output=True)


if __name__ == "__main__":
    test_slot_watcher()
    test_host_context()
    test_card_mount_unmount()
    test_host_orchestrator_build_agent()
    test_knowledge_retrieval()

    print("\n" + "=" * 60)
    print(f"测试完成: [PASS] {PASS} 通过, [FAIL] {FAIL} 失败")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
