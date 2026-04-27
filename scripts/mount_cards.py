"""模拟插卡操作脚本

用法:
    # 插入 Skill 卡 + Knowledge 卡
    python mount_cards.py --skill skills/hnu_freshman --knowledge knowledge/hnu_campus

    # 只插入 Skill 卡
    python mount_cards.py --skill skills/hnu_freshman

    # 只插入 Knowledge 卡
    python mount_cards.py --knowledge knowledge/hnu_campus

    # 拔卡（清空所有槽位）
    python mount_cards.py --eject
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLOT_A = PROJECT_ROOT / "cards" / "slot_a"  # Skill 卡槽位
SLOT_B = PROJECT_ROOT / "cards" / "slot_b"  # Knowledge 卡槽位


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
    parser.add_argument("--skill", type=str, help="Skill 卡源目录路径")
    parser.add_argument("--knowledge", type=str, help="Knowledge 卡源目录路径")
    parser.add_argument("--eject", action="store_true", help="拔卡（清空所有槽位）")
    args = parser.parse_args()

    if args.eject:
        clear_slot(SLOT_A)
        clear_slot(SLOT_B)
        print("[DONE] 所有卡片已拔出")
        return

    if args.skill:
        mount_card(PROJECT_ROOT / args.skill, SLOT_A)

    if args.knowledge:
        mount_card(PROJECT_ROOT / args.knowledge, SLOT_B)

    if not args.skill and not args.knowledge:
        print("当前槽位状态:")
        print(f"  Slot A (Skill): {list(SLOT_A.iterdir()) if SLOT_A.exists() else '空'}")
        print(f"  Slot B (Knowledge): {list(SLOT_B.iterdir()) if SLOT_B.exists() else '空'}")
        print("\n使用 --skill 或 --knowledge 参数插卡，--eject 拔卡")


if __name__ == "__main__":
    main()
