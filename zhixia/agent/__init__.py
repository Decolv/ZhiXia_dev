"""ZhiXia Agent — 遵循 LangChain 架构风格的通用 Agent 骨架

核心组件：
    BaseTool        → 工具抽象（name / description / args_schema / _run）
    ToolRegistry    → 工具注册表
    BaseAgent       → Agent 决策核心（plan → AgentAction | AgentFinish）
    AgentExecutor   → 执行循环（ReAct loop / ToolCalling loop）
    ReActAgent      → ReAct 风格的 Agent 实现
    ToolCallingAgent→ 结构化工具调用 Agent（现代 LLM 推荐）
    AgentState      → 不可变执行状态
    Runnable        → LCEL 管道协议（invoke / stream / | 组合）
    CallbackManager → 可观测性回调系统

典型使用（见 Pipeline 集成）：
    from zhixia.agent import ToolRegistry, ReActAgent, ToolCallingAgent, AgentExecutor, AgentRunner
    from zhixia.agent.callbacks import CallbackManager, LoggingHandler, StreamingDisplayHandler

    # 1. 注册工具
    registry = ToolRegistry()
    registry.register(WeatherTool())

    # 2. 创建 Agent（二选一）
    agent = ReActAgent(llm_engine=llm, tools=registry)
    # 或
    agent = ToolCallingAgent(llm_engine=llm, tools=registry)

    # 3. 创建执行器
    executor = AgentExecutor(agent=agent, tools=registry, max_iterations=5)

    # 4. 运行
    runner = AgentRunner(executor)
    result = runner.run("今天北京天气怎么样？")
    print(result["text"])

    # 5. 流式运行（带回调）
    callbacks = CallbackManager([StreamingDisplayHandler()])
    for chunk in runner.stream("今天北京天气怎么样？"):
        print(chunk, end="")

    # 6. LCEL 管道组合（高级）
    from zhixia.agent.runnable import RunnableLambda
    chain = RunnableLambda(lambda x: {"input": x}) | agent | RunnableLambda(lambda d: d.return_values.get("text", ""))
"""

from zhixia.agent.base import (
    AgentAction,
    AgentFinish,
    AgentStep,
    BaseAgent,
    BaseTool,
)
from zhixia.agent.callbacks import (
    BaseCallbackHandler,
    CallbackManager,
    LoggingHandler,
    StreamingDisplayHandler,
)
from zhixia.agent.executor import AgentExecutor, AgentRunner
from zhixia.agent.parser import ReActOutputParser
from zhixia.agent.prompts import ReActPromptTemplate
from zhixia.agent.react_agent import ReActAgent
from zhixia.agent.runnable import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
    RunnableSequence,
)
from zhixia.agent.state import AgentState
from zhixia.agent.tool import Tool, ToolRegistry, tool
from zhixia.agent.tool_agent import ToolCallingAgent

__all__ = [
    # 数据结构
    "AgentAction",
    "AgentFinish",
    "AgentStep",
    "AgentState",
    # 抽象基类
    "BaseAgent",
    "BaseTool",
    # 工具
    "Tool",
    "ToolRegistry",
    "tool",
    # Agent 实现
    "ReActAgent",
    "ToolCallingAgent",
    "AgentExecutor",
    "AgentRunner",
    # 解析器 / Prompt
    "ReActOutputParser",
    "ReActPromptTemplate",
    # Runnable / LCEL
    "Runnable",
    "RunnableConfig",
    "RunnableSequence",
    "RunnableMap",
    "RunnableLambda",
    # 回调
    "BaseCallbackHandler",
    "CallbackManager",
    "LoggingHandler",
    "StreamingDisplayHandler",
]
