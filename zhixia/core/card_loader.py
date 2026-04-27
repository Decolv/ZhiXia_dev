"""CardLoader — 卡片热插拔加载器 + 痕迹清除引擎

职责：
1. 扫描插卡槽位（文件系统目录）
2. 检测新卡插入 / 旧卡拔出
3. 动态导入卡片模块
4. 调用 on_mount() / on_unmount() 生命周期
5. 拔卡后彻底清除痕迹（sys.modules、工具注册表、知识库、人设）

槽位设计：
    cards/slot_a/  — 技能卡槽位（SkillCard）
    cards/slot_b/  — 知识卡槽位（KnowledgeCard）

插卡操作（对用户而言）：
    cp -r skills/hnu_freshman cards/slot_a/
    cp -r knowledge/hnu_campus cards/slot_b/

拔卡操作：
    rm -rf cards/slot_a/*
    rm -rf cards/slot_b/*
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from zhixia.core.card_base import CardBase, CardManifest, HostContext
from zhixia.core.user_profile import UserProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SlotWatcher — 槽位监视器
# ---------------------------------------------------------------------------

class SlotWatcher:
    """监视单个槽位目录的变化。

    通过对比目录签名（manifest.json 的 mtime + size）检测变化。
    """

    def __init__(self, slot_path: Path, slot_type: str) -> None:
        self.slot_path = slot_path
        self.slot_type = slot_type
        self._last_signature: Optional[str] = None
        self._current_card: Optional[CardBase] = None

    def detect_change(self) -> Tuple[bool, Optional[Path]]:
        """检测槽位是否有变化。

        Returns:
            (changed, card_root_or_none)
            changed=True, card_root=Path → 有新卡插入
            changed=True, card_root=None → 卡被拔出
            changed=False → 无变化
        """
        card_root = self._find_card_root()
        new_signature = self._compute_signature(card_root)

        if new_signature == self._last_signature:
            return False, None

        return True, card_root

    def update_signature(self, card_root: Optional[Path]) -> None:
        """更新签名（仅在挂载成功后调用）。"""
        self._last_signature = self._compute_signature(card_root)

    def set_current_card(self, card: CardBase) -> None:
        self._current_card = card

    def clear_current_card(self) -> None:
        self._current_card = None

    def _find_card_root(self) -> Optional[Path]:
        """查找槽位中的卡片根目录。

        策略：查找包含 manifest.json 的子目录。
        如果 slot_path 本身包含 manifest.json，也支持扁平结构。
        """
        manifest = self.slot_path / "manifest.json"
        if manifest.exists():
            return self.slot_path

        for subdir in self.slot_path.iterdir():
            if subdir.is_dir() and (subdir / "manifest.json").exists():
                return subdir
        return None

    def _compute_signature(self, card_root: Optional[Path]) -> str:
        """计算卡片签名（用于变化检测）。"""
        if card_root is None:
            return "empty"
        manifest = card_root / "manifest.json"
        if not manifest.exists():
            return "invalid"
        stat = manifest.stat()
        return f"{stat.st_mtime:.6f}:{stat.st_size}"


# ---------------------------------------------------------------------------
# CardLoader — 卡片加载器
# ---------------------------------------------------------------------------

class CardLoader:
    """卡片加载器：管理所有槽位的热插拔。

    Args:
        slots: 槽位配置 {"slot_id": (path, type)}
               例如 {"skill": (Path("cards/slot_a"), "skill")}
        host_context: 主机上下文，传给卡片的 on_mount/on_unmount
    """

    def __init__(
        self,
        slots: Dict[str, Tuple[Path, str]],
        host_context: HostContext,
    ) -> None:
        self.host = host_context
        self.watchers: Dict[str, SlotWatcher] = {}
        self.mounted_cards: Dict[str, CardBase] = {}
        self._module_prefix = "_zhixia_card"

        for slot_id, (path, slot_type) in slots.items():
            path.mkdir(parents=True, exist_ok=True)
            self.watchers[slot_id] = SlotWatcher(path, slot_type)
            logger.info("槽位就绪 [%s]: %s (类型: %s)", slot_id, path, slot_type)

    # -- 公共接口 --

    def scan_and_sync(self) -> Dict[str, Optional[str]]:
        """扫描所有槽位，同步卡片状态。

        Returns:
            变化记录 {"slot_id": "mounted:card_name" | "unmounted:card_name" | None}
        """
        changes = {}
        for slot_id, watcher in self.watchers.items():
            try:
                changed, card_root = watcher.detect_change()
                if not changed:
                    continue

                # 有变化：先卸载旧卡
                old_card = self.mounted_cards.get(slot_id)
                if old_card is not None:
                    self._unmount_card(slot_id, old_card)
                    changes[slot_id] = f"unmounted:{old_card.name}"

                # 再挂载新卡
                if card_root is not None:
                    new_card = self._mount_card(slot_id, card_root)
                    if new_card:
                        changes[slot_id] = f"mounted:{new_card.name}"
                    else:
                        changes[slot_id] = "mount_failed"
                else:
                    changes[slot_id] = "empty"
            except Exception as exc:
                logger.exception("[%s] 槽位同步失败: %s", slot_id, exc)
                changes[slot_id] = f"error:{exc}"

        return changes

    def force_unmount_all(self) -> None:
        """强制卸载所有卡片（关机/重置时使用）。"""
        for slot_id, card in list(self.mounted_cards.items()):
            self._unmount_card(slot_id, card)
        self._clear_all_traces()
        logger.info("所有卡片已卸载，痕迹已清除")

    def get_mounted_cards(self) -> List[CardBase]:
        """获取当前所有已挂载的卡片。"""
        return list(self.mounted_cards.values())

    def get_mounted_names(self) -> List[str]:
        return [c.name for c in self.mounted_cards.values()]

    def is_slot_empty(self, slot_id: str) -> bool:
        """检查槽位是否没有挂载卡片。
        
        注意：此方法仅检查内存中是否有挂载的卡片，
        不检查文件系统上槽位目录是否为空。
        """
        return slot_id not in self.mounted_cards

    # -- 内部方法：挂载 / 卸载 --

    def _mount_card(self, slot_id: str, card_root: Path) -> Optional[CardBase]:
        """挂载单张卡片。"""
        logger.info("开始挂载卡片 [%s]: %s", slot_id, card_root)

        manifest = CardManifest.load(card_root)
        if manifest is None:
            logger.error("[%s] manifest 加载失败", slot_id)
            return None

        slot_type = self.watchers[slot_id].slot_type
        if slot_type != manifest.type:
            logger.error(
                "[%s] 卡片类型不匹配: 槽位要求 '%s'，但卡片是 '%s'",
                slot_id, slot_type, manifest.type,
            )
            return None

        card_instance = self._import_and_instantiate(manifest, card_root, slot_id)
        if card_instance is None:
            return None

        old_card_root = self.host.card_root
        self.host.card_root = card_root

        self.host.user_profile = UserProfile(card_root=card_root)
        logger.info("[%s] 用户画像已加载", slot_id)

        try:
            card_instance.on_mount(self.host)
        except Exception as exc:
            logger.exception("[%s] on_mount 失败: %s", slot_id, exc)
            self.host.card_root = old_card_root
            self._cleanup_modules(card_root)
            return None

        self.mounted_cards[slot_id] = card_instance
        watcher = self.watchers[slot_id]
        watcher.set_current_card(card_instance)
        watcher.update_signature(card_root)

        logger.info("[%s] 卡片挂载成功: %s v%s", slot_id, manifest.name, manifest.version)
        return card_instance

    def _unmount_card(self, slot_id: str, card: CardBase) -> None:
        """卸载单张卡片。"""
        logger.info("开始卸载卡片 [%s]: %s", slot_id, card.name)

        if self.host.user_profile:
            self.host.user_profile.save()
            logger.info("[%s] 用户画像已保存", slot_id)
            self.host.user_profile = None

        try:
            card.on_unmount(self.host)
        except Exception as exc:
            logger.exception("[%s] on_unmount 失败: %s", slot_id, exc)

        self.host.persona_holder.clear_overlay(card.name)
        self.host.knowledge_hub.unregister_retriever(card.name)
        self.host.knowledge_hub.unregister_assets(card.name)
        for tool_name in list(card.registered_tool_names):
            self.host.tool_registry.unregister(tool_name)

        self._cleanup_modules(card.card_root)

        if slot_id in self.mounted_cards:
            del self.mounted_cards[slot_id]

        watcher = self.watchers.get(slot_id)
        if watcher:
            watcher.clear_current_card()

        old_card_root = self.host.card_root
        if old_card_root == card.card_root:
            self.host.card_root = Path()

        logger.info("[%s] 卡片卸载完成: %s", slot_id, card.name)

    # -- 动态导入 --

    def _import_and_instantiate(
        self, manifest: CardManifest, card_root: Path, slot_id: str
    ) -> Optional[CardBase]:
        """动态导入卡片模块并实例化。"""
        entry_file = card_root / manifest.entrypoint
        if not entry_file.exists():
            logger.error("入口文件不存在: %s", entry_file)
            return None

        module_name = f"{self._module_prefix}_{slot_id}_{manifest.name}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, entry_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法加载模块: {entry_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 查找 Card 子类
            card_cls = self._find_card_class(module)
            if card_cls is None:
                logger.error("[%s] 未找到 Card 子类", slot_id)
                return None

            return card_cls(manifest=manifest, card_root=card_root)

        except Exception as exc:
            logger.exception("[%s] 导入卡片失败: %s", slot_id, exc)
            if module_name in sys.modules:
                del sys.modules[module_name]
            return None

    def _find_card_class(self, module) -> Optional[type]:
        """在模块中查找 CardBase 的子类。"""
        import inspect

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, CardBase) and obj is not CardBase:
                return obj
        return None

    # -- 痕迹清除 --

    def _cleanup_modules(self, card_root: Path) -> None:
        """从 sys.modules 中清除与卡片相关的模块。

        使用精确路径匹配（兼容 Python 3.8）。
        同时清理子模块。
        """
        def _is_relative_to(path: Path, other: Path) -> bool:
            try:
                path.relative_to(other)
                return True
            except ValueError:
                return False

        to_remove = []
        root_resolved = card_root.resolve()

        for name, mod in list(sys.modules.items()):
            if not name.startswith(self._module_prefix):
                continue
            mod_file = getattr(mod, "__file__", None)
            if not mod_file:
                continue
            try:
                mod_path = Path(mod_file).resolve()
                if _is_relative_to(mod_path, root_resolved):
                    to_remove.append(name)
            except (ValueError, OSError):
                continue

        for name in to_remove:
            del sys.modules[name]
            logger.debug("清理模块: %s", name)

    def _clear_all_traces(self) -> None:
        """彻底清除所有痕迹（模块、缓存、人设、知识）。"""
        # 清理所有卡片模块
        to_remove = [name for name in sys.modules if name.startswith(self._module_prefix)]
        for name in to_remove:
            del sys.modules[name]

        # 清理主机上下文
        self.host.persona_holder.clear_overlay()
        self.host.knowledge_hub.clear_all()

        # 清理工具注册表
        self.host.tool_registry.clear()

        logger.info("所有痕迹已清除")
