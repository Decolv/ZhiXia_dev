"""Agent 核心抽象基类 —— 对应 LangChain BaseAgent / BaseTool

设计原则：
1. 最小依赖，不引入 langchain 包，保持项目轻量。
2. 保留 LangChain 的核心语义：AgentAction / AgentFinish / AgentStep / BaseTool。
3. 所有 Agent 交互都是同步的（语音场景下工具调用通常很快）。
4. **新增**：BaseAgent 继承 Runnable 协议，支持 invoke / stream / LCEL 管道组合。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from zhixia.agent.runnable import Runnable, RunnableConfig


# ---------------------------------------------------------------------------
# 数据结构 — Agent 的中间决策产物
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """Agent 决定执行某个工具时的动作描述。

    Attributes:
        tool: 要调用的工具名称。
        tool_input: 传给工具的参数（通常是 str 或 dict）。
        thought: Agent 的推理过程（ReAct 中的 Thought）。
        log: 原始 LLM 输出片段（用于调试或展示）。
    """

    tool: str
    tool_input: Union[str, dict] = ""
    thought: str = ""
    log: str = ""


@dataclass
class AgentFinish:
    """Agent 决定不再调用工具，直接给出最终答案。

    Attributes:
        return_values: 最终输出内容。至少包含 "text" 键。
                       若启用结构化输出，可额外包含 "emotion" / "metadata"。
        log: 原始 LLM 输出片段。
    """

    return_values: Dict[str, Any] = field(default_factory=dict)
    log: str = ""


@dataclass
class AgentStep:
    """ReAct 循环中的单步记录：Action + 执行后得到的 Observation。"""

    action: AgentAction
    observation: str = ""


# Union 类型，plan() 的返回值
AgentDecision = Union[AgentAction, AgentFinish]


# ---------------------------------------------------------------------------
# Tool 抽象
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """工具抽象基类 —— 对应 LangChain BaseTool。

    子类需要实现：
        - name: 工具唯一标识
        - description: 工具功能描述（会被写入 prompt，影响 LLM 决策）
        - _run(args) -> str: 同步执行逻辑

    可选覆盖：
        - args_schema: 参数格式说明（默认无 schema，LLM 自由输入字符串）
    """

    name: str = ""
    description: str = ""
    args_schema: Optional[type] = None

    def run(self, tool_input: Union[str, dict], **kwargs: Any) -> str:
        """外部调用入口，内部转给 _run。可在此添加通用日志/异常处理。"""
        try:
            return self._run(tool_input, **kwargs)
        except Exception as exc:
            # 工具异常不应导致 Agent 崩溃，返回错误描述作为 Observation
            return f"工具执行出错: {type(exc).__name__}: {exc}"

    @abstractmethod
    def _run(self, tool_input: Union[str, dict], **kwargs: Any) -> str:
        """子类实现的具体执行逻辑。"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def to_json_schema(self) -> Dict[str, Any]:
        """生成工具的 JSON Schema 描述（用于结构化工具调用）。"""
        from zhixia.agent.tool_agent import ToolSchemaBuilder

        return ToolSchemaBuilder.build(self)


# ---------------------------------------------------------------------------
# Agent 抽象 —— 继承 Runnable 协议
# ---------------------------------------------------------------------------

class BaseAgent(Runnable[Any, AgentDecision], ABC):
    """Agent 决策核心 —— 对应 LangChain BaseAgent，同时兼容 Runnable 协议。

    职责：
        1. 接收用户输入 + 当前 intermediate_steps（历史 Thought/Action/Observation）。
        2. 组装 prompt 并调用 LLM。
        3. 解析 LLM 输出，返回 AgentAction（继续调工具）或 AgentFinish（结束）。

    子类需要实现：
        - plan → AgentDecision
        - input_keys / return_values

    **新增**：继承 Runnable 后，可以直接：
        result = agent.invoke({"input": "...", "intermediate_steps": [...]})
        # 或管道组合
        chain = prompt | agent | executor
    """

    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks=None,
        **kwargs: Any,
    ) -> AgentDecision:
        """核心决策方法。

        Args:
            intermediate_steps: 当前已完成的 ReAct 步骤列表。
            callbacks: 预留扩展（如日志、观测）。
            **kwargs: 额外上下文，如 user_input, rag_context 等。

        Returns:
            AgentAction: 需要调用工具。
            AgentFinish: 已得到最终答案。
        """
        ...

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Agent plan() 需要的外部输入键名列表，如 ["input"] """
        ...

    @property
    @abstractmethod
    def return_values(self) -> List[str]:
        """AgentFinish.return_values 中期望包含的键名，如 ["text", "emotion"] """
        ...

    # -- Runnable 协议实现 --

    def _invoke(self, input: Any, config: RunnableConfig) -> AgentDecision:
        """将 Runnable 的 invoke 路由到 plan()。"""
        if isinstance(input, dict):
            intermediate_steps = input.get("intermediate_steps", [])
            return self.plan(intermediate_steps, callbacks=config.callbacks, **input)
        # 如果是 AgentState，提取所需信息
        from zhixia.agent.state import AgentState

        if isinstance(input, AgentState):
            return self.plan(
                input.intermediate_steps,
                callbacks=config.callbacks,
                input=input.last_user_input,
                messages=input.to_messages_for_llm(),
                **input.metadata,
            )
        raise ValueError(f"Agent 不支持的输入类型: {type(input)}")
