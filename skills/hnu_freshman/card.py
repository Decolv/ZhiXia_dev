"""湖南大学新生助手 — Skill 卡入口

插卡时自动注册：
- 3 个工具：校园导航、专业查询、生活指南
- 1 个人设：湖小助角色
"""

import json
from pathlib import Path

from zhixia.agent.tool import ToolRegistry
from zhixia.core.card_base import CardManifest, HostContext, SkillCard

from skills.hnu_freshman.tools.campus_navigate import CampusNavigateTool
from skills.hnu_freshman.tools.life_guide import CampusLifeGuideTool
from skills.hnu_freshman.tools.major_query import MajorQueryTool


class HNUFreshmanSkill(SkillCard):
    """湖南大学新生助手技能卡。"""

    def on_mount(self, host: HostContext) -> None:
        """插卡时：注册工具 + 加载人设。"""
        # 注册工具
        host.tool_registry.register(CampusNavigateTool())
        host.tool_registry.register(MajorQueryTool())
        host.tool_registry.register(CampusLifeGuideTool())

        # 加载人设
        persona = self._load_persona()
        if persona:
            host.persona_holder.set_overlay(persona, self.name)

        print(f"[MOUNT] Skill 卡已插入: {self.display_name}")
        print(f"   工具: campus_navigate, query_major, campus_life_guide")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销工具 + 恢复人设。"""
        host.tool_registry.unregister("campus_navigate")
        host.tool_registry.unregister("query_major")
        host.tool_registry.unregister("campus_life_guide")
        host.persona_holder.clear_overlay()
        print(f"[UNMOUNT] Skill 卡已拔出: {self.display_name}")

    def get_tools(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(CampusNavigateTool())
        registry.register(MajorQueryTool())
        registry.register(CampusLifeGuideTool())
        return registry

    def get_persona(self) -> str:
        return self._load_persona() or ""

    def _load_persona(self) -> str:
        persona_path = self.card_root / "persona.json"
        if not persona_path.exists():
            return ""
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("persona", "")
        except Exception:
            return ""
