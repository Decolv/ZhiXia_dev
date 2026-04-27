"""Tool 实现与注册表 — 对应 LangChain Tool / StructuredTool / ToolRegistry

提供两种创建工具的方式：
1. 继承 BaseTool（适合复杂工具）
2. @tool 装饰器（适合快速把函数变成工具）

示例：
    @tool("weather", "查询指定城市的天气")
    def get_weather(city: str) -> str:
        return f"{city}今天晴，25°C"

    registry = ToolRegistry()
    registry.register(get_weather)
    print(registry.format_tool_descriptions())
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from zhixia.agent.base import BaseTool


class Tool(BaseTool):
    """基于函数的工具封装 — 对应 LangChain Tool。

    Args:
        name: 工具名（LLM 通过这个名字调用）。
        description: 工具描述（写入 prompt）。
        func: 实际执行的可调用对象。
        args_schema: 可选的参数 schema 类（用于结构化参数）。
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        args_schema: Optional[type] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema

    def _run(self, tool_input: Union[str, dict], **kwargs: Any) -> str:
        if isinstance(tool_input, dict):
            return self.func(**tool_input, **kwargs)
        return self.func(tool_input, **kwargs)


class ToolRegistry:
    """工具注册表 — 管理可用工具，为 Agent 生成工具描述文本。"""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """注册单个工具，支持链式调用。"""
        if not tool.name:
            raise ValueError("工具必须设置 name")
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已存在")
        self._tools[tool.name] = tool
        return self

    def register_from_callable(
        self,
        func: Callable[..., str],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "ToolRegistry":
        """将普通函数封装为 Tool 并注册。"""
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        return self.register(Tool(tool_name, tool_desc, func))

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def format_tool_descriptions(self) -> str:
        """生成 prompt 中使用的工具描述文本。"""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def to_json_schemas(self) -> List[Dict[str, Any]]:
        """生成所有工具的 JSON Schema 列表（用于结构化工具调用）。"""
        from zhixia.agent.tool_agent import ToolSchemaBuilder

        return ToolSchemaBuilder.build_all(self)

    def get_tool_names(self) -> List[str]:
        """获取所有已注册工具的名称列表。"""
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """装饰器：将函数快速转换为 Tool 对象。

    用法：
        @tool("weather", "查询天气")
        def get_weather(city: str) -> str:
            ...

        # 也可以不传参数，自动从函数名/文档推导
        @tool
        def get_time() -> str:
            '''获取当前时间'''
            ...
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()
        return Tool(tool_name, tool_desc, func)

    # 支持无参数调用: @tool
def _tool_wrapper(func_or_name=None, description=None):
    if callable(func_or_name):
        # @tool  (无括号)
        return Tool(func_or_name.__name__, (func_or_name.__doc__ or "").strip(), func_or_name)
    # @tool("name", "desc") 或 @tool()
    return tool(func_or_name, description)

# 重新导出更灵活的装饰器
# 为了简化，我们保持上面的 `tool` 只能带参数使用
# 如果用户想要无参数，可以直接：
#   @tool()
#   def foo(): ...
# 但这样写比较怪。我们改成兼容模式：


def tool(
    name: Optional[Union[str, Callable]] = None,
    description: Optional[str] = None,
) -> Union[Tool, Callable]:
    """兼容两种写法的装饰器。

    @tool
    def foo(): ...

    @tool("foo", "desc")
    def foo(): ...
    """
    if callable(name):
        # 第一种情况：@tool （name 实际上是 func）
        func = name
        return Tool(func.__name__, (func.__doc__ or "").strip(), func)

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()
        return Tool(tool_name, tool_desc, func)
    return decorator
