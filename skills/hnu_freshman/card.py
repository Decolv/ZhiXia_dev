"""湖南大学新生助手 — Skill 卡入口

插卡时自动注册：
- 3 个工具：校园导航、专业查询、生活指南
- 1 个人设：湖小助角色

注意：所有工具都使用LLM智能生成答案，需要注入LLM引擎。
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
        """插卡时：注册工具 + 加载人设。

        为所有工具注入LLM引擎和回调管理器，支持智能生成和思考播报。
        """
        # 尝试从host获取LLM引擎
        llm_engine = getattr(host, 'llm_engine', None)

        # 创建工具实例并注入LLM引擎
        campus_navigate_tool = CampusNavigateTool(llm_engine=llm_engine)
        major_query_tool = MajorQueryTool(llm_engine=llm_engine)
        life_guide_tool = CampusLifeGuideTool(llm_engine=llm_engine)

        # 注册工具
        host.tool_registry.register(campus_navigate_tool)
        host.tool_registry.register(major_query_tool)
        host.tool_registry.register(life_guide_tool)

        # 加载人设
        persona = self._load_persona()
        if persona:
            host.persona_holder.set_overlay(persona, self.name)

        print(f"[MOUNT] Skill 卡已插入: {self.display_name}")
        print(f"   工具: campus_navigate, query_major, campus_life_guide (均使用LLM智能生成)")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销工具 + 恢复人设。"""
        host.tool_registry.unregister("campus_navigate")
        host.tool_registry.unregister("query_major")
        host.tool_registry.unregister("campus_life_guide")
        host.persona_holder.clear_overlay()
        print(f"[UNMOUNT] Skill 卡已拔出: {self.display_name}")

    def get_tools(self) -> ToolRegistry:
        """获取工具注册表（用于预览，不注入LLM引擎）。"""
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
