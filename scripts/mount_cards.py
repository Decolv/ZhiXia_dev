"""模拟插卡操作脚本

用法:
    # 插入到指定槽位（槽位名: slot_a / slot_b / slot_c / slot_d）
    python mount_cards.py --slot slot_a skills/your_skill
    python mount_cards.py --slot slot_b knowledge/your_knowledge
    python mount_cards.py --slot slot_c knowledge/another_knowledge

    # 只插 Skill 卡到 slot_a（向后兼容）
    python mount_cards.py --skill skills/your_skill

    # 只插 Knowledge 卡到 slot_b（向后兼容）
    python mount_cards.py --knowledge knowledge/your_knowledge

    # 拔卡（清空所有槽位）
    python mount_cards.py --eject
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLOTS = {
    "slot_a": PROJECT_ROOT / "cards" / "slot_a",
    "slot_b": PROJECT_ROOT / "cards" / "slot_b",
    "slot_c": PROJECT_ROOT / "cards" / "slot_c",
    "slot_d": PROJECT_ROOT / "cards" / "slot_d",
}


def clear_slot(slot_path: Path) -> None:
    """清空槽位。"""
    if not slot_path.exists():
        return
    for item in slot_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"[CLEAR] 已清空槽位: {slot_path}")


def mount_card(source: Path, slot_path: Path) -> None:
    """将卡片挂载到槽位。"""
    if not source.exists():
        print(f"❌ 卡片源目录不存在: {source}")
        sys.exit(1)

    clear_slot(slot_path)
    slot_path.mkdir(parents=True, exist_ok=True)

    # 复制卡片内容到槽位（保持目录结构）
    for item in source.iterdir():
        dest = slot_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"[MOUNT] 已插卡: {source.name} -> {slot_path}")


def main():
    parser = argparse.ArgumentParser(description="ZhiXia 插卡模拟器")
    parser.add_argument("--skill", type=str, help="Skill 卡源目录路径（插入到 slot_a）")
    parser.add_argument("--knowledge", type=str, help="Knowledge 卡源目录路径（插入到 slot_b）")
    parser.add_argument("--slot", nargs=2, metavar=("SLOT_NAME", "SOURCE"), action="append",
                        help="指定槽位插卡，例如 --slot slot_a skills/my_skill")
    parser.add_argument("--eject", action="store_true", help="拔卡（清空所有槽位）")
    args = parser.parse_args()

    if args.eject:
        for slot_path in SLOTS.values():
            clear_slot(slot_path)
        print("[DONE] 所有卡片已拔出")
        return

    # 处理 --slot 参数（新方式，支持任意槽位）
    if args.slot:
        for slot_name, source in args.slot:
            if slot_name not in SLOTS:
                print(f"❌ 未知槽位: {slot_name}。可用槽位: {', '.join(SLOTS.keys())}")
                sys.exit(1)
            mount_card(PROJECT_ROOT / source, SLOTS[slot_name])

    # 处理向后兼容的 --skill / --knowledge 参数
    if args.skill:
        mount_card(PROJECT_ROOT / args.skill, SLOTS["slot_a"])

    if args.knowledge:
        mount_card(PROJECT_ROOT / args.knowledge, SLOTS["slot_b"])

    if not args.skill and not args.knowledge and not args.slot:
        print("当前槽位状态:")
        for name, path in SLOTS.items():
            contents = list(path.iterdir()) if path.exists() else []
            status = f"{len(contents)} 个条目" if contents else "空"
            print(f"  {name}: {status}")
        print("\n使用 --slot <槽位> <目录> 插卡，--eject 拔卡")
        print("或 --skill / --knowledge 使用向后兼容方式")


if __name__ == "__main__":
    main()
