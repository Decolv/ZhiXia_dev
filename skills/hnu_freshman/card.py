"""湖南大学新生助手 — Skill 卡入口

插卡时自动注册：
- 3 个工具：校园导航、专业查询、生活指南
- 1 个人设：湖小助角色
- 导航响应后处理器：处理导航界面展示
- Agent 配置：ToolCallingAgent

注意：所有工具都在插卡时创建和注册，拔卡时自动注销。
工具使用LLM智能生成答案，需要注入LLM引擎。
"""

import json
from pathlib import Path
from typing import Optional

from zhixia.agent.tool import ToolRegistry
from zhixia.core.card_base import CardManifest, HostContext, SkillCard

from skills.hnu_freshman.tools.campus_navigate import CampusNavigateTool
from skills.hnu_freshman.tools.life_guide import CampusLifeGuideTool
from skills.hnu_freshman.tools.major_query import MajorQueryTool


class HNUFreshmanSkill(SkillCard):
    """湖南大学新生助手技能卡。

    工具生命周期：
    - 创建：在 on_mount() 插卡时
    - 注册：在 on_mount() 插卡时
    - 注销：在 on_unmount() 拔卡时
    """

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        super().__init__(manifest, card_root)
        # 工具实例在插卡时创建
        self._tools_created = False

    def on_mount(self, host: HostContext) -> None:
        """插卡时：创建工具 + 注册工具 + 加载人设 + 注册响应处理器 + 配置Agent。"""
        if self._tools_created:
            return  # 防止重复注册

        # 尝试从host获取LLM引擎
        llm_engine = getattr(host, 'llm_engine', None)

        # 创建工具实例并注入LLM引擎（插卡时创建）
        campus_navigate_tool = CampusNavigateTool(llm_engine=llm_engine)
        major_query_tool = MajorQueryTool(llm_engine=llm_engine)
        life_guide_tool = CampusLifeGuideTool(llm_engine=llm_engine)

        # 注册工具到主机（插卡时注册）
        host.tool_registry.register(campus_navigate_tool)
        host.tool_registry.register(major_query_tool)
        host.tool_registry.register(life_guide_tool)

        self._tools_created = True

        # 加载人设
        persona = self._load_persona()
        if persona:
            host.persona_holder.set_overlay(persona, self.name)

        # 配置 Agent 类型为 ToolCalling（更适合工具调用场景）
        host.agent_configurator.set_agent_type("tool_calling")
        host.agent_configurator.set_max_iterations(3)

        # 注册导航响应后处理器
        if host.display:
            from skills.hnu_freshman.nav_processor import NavResponseProcessor
            self._nav_processor = NavResponseProcessor(
                display=host.display,
                nav_data_provider=campus_navigate_tool,
            )
            host.register_response_processor(self._nav_processor)

        print(f"[MOUNT] Skill 卡已插入: {self.display_name}")
        print(f"   工具: campus_navigate, query_major, campus_life_guide (均使用LLM智能生成)")
        print(f"   Agent: ToolCallingAgent")
        print(f"   响应处理器: NavResponseProcessor")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销工具 + 恢复人设 + 注销响应处理器。"""
        # 注销工具（拔卡时注销）
        host.tool_registry.unregister("campus_navigate")
        host.tool_registry.unregister("query_major")
        host.tool_registry.unregister("campus_life_guide")
        host.persona_holder.clear_overlay()

        # 注销导航响应后处理器并清理资源
        if hasattr(self, '_nav_processor') and self._nav_processor:
            self._nav_processor.cleanup()
            host.unregister_response_processor("nav_response_processor")
            self._nav_processor = None

        # 重置 Agent 配置
        host.agent_configurator.clear()

        self._tools_created = False

        print(f"[UNMOUNT] Skill 卡已拔出: {self.display_name}")

    def get_tools(self) -> ToolRegistry:
        """获取工具列表预览（仅用于元数据展示，不创建实际工具实例）。"""
        registry = ToolRegistry()
        # 仅返回工具名称和描述，不创建实例
        # 实际工具在 on_mount() 插卡时创建和注册
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
