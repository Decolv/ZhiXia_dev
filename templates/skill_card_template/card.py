"""Skill 卡模板 —— 最小可运行示例

展示自包含导入规范：
1. 卡片根目录通过 sys.path 加入 Python 路径
2. 工具导入使用基于卡片目录的绝对路径（无前缀）
3. 不依赖项目源代码中的 skills/ 路径
"""

import json
import sys
from pathlib import Path
from typing import Optional

from zhixia.agent.tool import Tool, ToolRegistry
from zhixia.core.card_base import CardManifest, HostContext, SkillCard

# ============================================================
# 自包含导入关键代码（所有 Skill 卡必须包含）
# ============================================================
_CARD_ROOT = Path(__file__).parent.resolve()
if str(_CARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_CARD_ROOT))

from tools.example_tool import ExampleTool
# ============================================================


class SkillTemplate(SkillCard):
    """技能卡模板。

    工具生命周期：
    - 创建：在 on_mount() 插卡时
    - 注册：在 on_mount() 插卡时
    - 注销：在 on_unmount() 拔卡时
    """

    def __init__(self, manifest: CardManifest, card_root: Path) -> None:
        super().__init__(manifest, card_root)
        self._tools_created = False

    def on_mount(self, host: HostContext) -> None:
        """插卡时：创建工具 + 注册工具 + 加载人设。"""
        if self._tools_created:
            return  # 防止重复注册

        llm_engine = host.llm_engine

        # 创建工具实例并注入LLM引擎（插卡时创建）
        example_tool = ExampleTool(llm_engine=llm_engine)

        # 注册工具到主机（插卡时注册）
        host.tool_registry.register(example_tool)
        self.registered_tool_names = ["example_tool"]

        self._tools_created = True

        # 加载人设
        persona = self._load_persona()
        if persona:
            host.persona_holder.set_overlay(persona, self.name)

        print(f"[MOUNT] Skill 卡已插入: {self.display_name}")
        print(f"   工具: example_tool")

    def on_unmount(self, host: HostContext) -> None:
        """拔卡时：注销工具 + 恢复人设。"""
        for tool_name in list(self.registered_tool_names):
            host.tool_registry.unregister(tool_name)
        self.registered_tool_names = []

        host.persona_holder.clear_overlay(self.name)

        self._tools_created = False

        print(f"[UNMOUNT] Skill 卡已拔出: {self.display_name}")

    def get_tools(self) -> ToolRegistry:
        """获取工具列表预览（仅用于元数据展示，不创建实际工具实例）。"""
        registry = ToolRegistry()
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
