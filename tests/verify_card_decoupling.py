"""验证卡片与主机深度解耦

测试步骤：
1. 将 skills/hnu_freshman 复制到 cards/slot_a/
2. 临时隐藏 skills/hnu_freshman（模拟独立部署）
3. 使用 CardLoader 从 cards/slot_a/ 加载卡片
4. 验证工具模块的 __file__ 指向 cards/slot_a/ 而非 skills/
5. 恢复备份
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from zhixia.core.card_base import HostContext, PersonaHolder, KnowledgeHub
from zhixia.core.card_loader import CardLoader
from zhixia.agent.tool import ToolRegistry


def main():
    skill_source = PROJECT_ROOT / "skills" / "hnu_freshman"
    skill_backup = PROJECT_ROOT / "skills" / "hnu_freshman_backup"
    slot_a = PROJECT_ROOT / "cards" / "slot_a"
    slot_b = PROJECT_ROOT / "cards" / "slot_b"

    print("=" * 60)
    print("卡片解耦验证测试")
    print("=" * 60)

    # 1. 确保备份不存在
    if skill_backup.exists():
        shutil.rmtree(skill_backup)

    # 2. 复制卡片到槽位
    if slot_a.exists():
        shutil.rmtree(slot_a)
    slot_a.mkdir(parents=True, exist_ok=True)
    for item in skill_source.iterdir():
        dest = slot_a / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"[1/5] 已复制卡片到: {slot_a}")

    # 3. 隐藏源代码（关键步骤：模拟独立部署）
    shutil.move(str(skill_source), str(skill_backup))
    print(f"[2/5] 已隐藏源代码: {skill_source} -> {skill_backup}")

    # 4. 创建最小 HostContext 并尝试加载卡片
    host = HostContext(
        tool_registry=ToolRegistry(),
        persona_holder=PersonaHolder("你是助手"),
        knowledge_hub=KnowledgeHub(),
        card_root=Path(),
    )

    slots = {
        "skill": (slot_a, "skill"),
        "knowledge": (slot_b, "knowledge"),
    }
    loader = CardLoader(slots, host)

    try:
        changes = loader.scan_and_sync()
        print(f"[3/5] 槽位扫描结果: {changes}")

        mounted = loader.get_mounted_cards()
        if not mounted:
            print("[FAIL] 失败: 卡片未能从 cards/slot_a/ 加载")
            return False

        card = mounted[0]
        print(f"[4/5] 卡片加载成功: {card.name} v{card.manifest.version}")

        # 5. 验证工具模块来源
        # 触发 on_mount 来加载工具
        card.on_mount(host)

        tools = host.tool_registry.list_tools()
        print(f"    已注册工具: {[t.name for t in tools]}")

        # 检查第一个工具的模块来源
        if tools:
            tool_cls = tools[0].__class__
            tool_module_file = getattr(sys.modules.get(tool_cls.__module__, None), "__file__", None)
            if tool_module_file:
                tool_module_path = Path(tool_module_file).resolve()
                slot_a_resolved = slot_a.resolve()
                is_from_slot = str(tool_module_path).startswith(str(slot_a_resolved))
                if is_from_slot:
                    print(f"[PASS] 验证通过: 工具模块来自 cards/slot_a/")
                    print(f"    {tool_module_path}")
                else:
                    print(f"[FAIL] 验证失败: 工具模块仍来自 skills/ 或其他路径")
                    print(f"    实际: {tool_module_path}")
                    print(f"    期望前缀: {slot_a_resolved}")
                    return False
            else:
                print("⚠️ 无法确定工具模块来源（无 __file__）")

        # 6. 卸载卡片
        loader.force_unmount_all()
        print(f"[5/5] 卡片已卸载")

        print("\n" + "=" * 60)
        print("[OK] 全部验证通过！卡片与主机已深度解耦。")
        print("=" * 60)
        return True

    except Exception as exc:
        print(f"[ERROR] 验证异常: {exc}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 恢复备份
        if skill_backup.exists():
            if skill_source.exists():
                shutil.rmtree(skill_source)
            shutil.move(str(skill_backup), str(skill_source))
            print(f"[恢复] 已还原源代码: {skill_backup} -> {skill_source}")

        # 清空槽位
        if slot_a.exists():
            shutil.rmtree(slot_a)
            slot_a.mkdir(parents=True, exist_ok=True)
        print("[恢复] 已清空槽位")


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
