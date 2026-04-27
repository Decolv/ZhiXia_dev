"""ZhiXia Agent — 遵循 LangChain 架构风格的通用 Agent 骨架

核心组件：
    BaseTool        → 工具抽象（name / description / args_schema / _run）
    ToolRegistry    → 工具注册表
    BaseAgent       → Agent 决策核心（plan → AgentAction | AgentFinish）
    AgentExecutor   → 执行循环（ReAct loop）
    ReActAgent      → ReAct 风格的 Agent 实现

典型使用（见 Pipeline 集成）：
    registry = ToolRegistry()
    registry.register(WeatherTool())

    agent = ReActAgent(llm_engine=llm, tools=registry, memory=memory)
    executor = AgentExecutor(agent=agent, max_iterations=5)

    result = executor.run("今天北京天气怎么样？")
    # result → AgentFinish(return_values={"text": "今天北京晴，25°C..."})
"""

from zhixia.agent.base import (
    AgentAction,
    AgentFinish,
    AgentStep,
    BaseAgent,
    BaseTool,
)
from zhixia.agent.executor import AgentExecutor
from zhixia.agent.parser import ReActOutputParser
from zhixia.agent.prompts import ReActPromptTemplate
from zhixia.agent.tool import Tool, ToolRegistry, tool
from zhixia.agent.react_agent import ReActAgent

__all__ = [
    "AgentAction",
    "AgentFinish",
    "AgentStep",
    "BaseAgent",
    "BaseTool",
    "Tool",
    "ToolRegistry",
    "tool",
    "AgentExecutor",
    "ReActAgent",
    "ReActOutputParser",
    "ReActPromptTemplate",
]
