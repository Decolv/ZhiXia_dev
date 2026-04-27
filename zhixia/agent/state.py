"""AgentState — Agent 执行状态管理

参照 LangChain 的 AgentState / AgentScratchpad 设计，但做了轻量化和流式优化：
1. 状态是不可变的（dataclass，修改时创建新实例），便于追踪和回滚。
2. 支持将状态序列化为 prompt 上下文（scratchpad）。
3. 支持工具调用结果（Observation）直接追加到消息列表。

核心类：
    AgentState       — 完整的 Agent 执行状态
    AgentStatus      — 状态机枚举：thinking / tool_call / finished / error

使用场景：
    state = AgentState()
    state = state.add_message(LLMMessage(role="user", content="今天北京天气？"))
    state = state.add_step(AgentAction(...), "晴天，25°C")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from zhixia.agent.base import AgentAction, AgentFinish, AgentStep
from zhixia.llm.base import LLMMessage

AgentStatus = Literal["idle", "thinking", "tool_call", "finished", "error"]


@dataclass(frozen=True)
class AgentState:
    """Agent 执行的不可变状态。

    Attributes:
        messages: 完整的对话消息历史（含 system / user / assistant / tool）。
        intermediate_steps: ReAct 循环中的 Thought/Action/Observation 记录。
        status: 当前状态机状态。
        iteration: 当前迭代次数（防止无限循环）。
        metadata: 额外的运行时元数据（如 rag_context, emotion 等）。
    """

    messages: List[LLMMessage] = field(default_factory=list)
    intermediate_steps: List[AgentStep] = field(default_factory=list)
    status: AgentStatus = "idle"
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- 不可变更新方法（返回新实例） --

    def add_message(self, message: LLMMessage) -> AgentState:
        """追加单条消息。"""
        return self._replace(messages=[*self.messages, message])

    def add_messages(self, messages: List[LLMMessage]) -> AgentState:
        """批量追加消息。"""
        return self._replace(messages=[*self.messages, *messages])

    def add_step(self, action: AgentAction, observation: str) -> AgentState:
        """追加一个完整的 ReAct 步骤（Action + Observation）。"""
        step = AgentStep(action=action, observation=observation)
        return self._replace(
            intermediate_steps=[*self.intermediate_steps, step],
            iteration=self.iteration + 1,
        )

    def set_status(self, status: AgentStatus) -> AgentState:
        """更新状态机状态。"""
        return self._replace(status=status)

    def set_metadata(self, **kwargs: Any) -> AgentState:
        """更新元数据。"""
        return self._replace(metadata={**self.metadata, **kwargs})

    def with_tool_result(self, action: AgentAction, observation: str) -> AgentState:
        """将工具调用结果格式化为 message 并追加到对话历史。

        兼容 OpenAI function calling 格式：
            assistant: {"tool_calls": [{...}]}
            tool: {"role": "tool", "content": observation}
        """
        tool_msg = LLMMessage(
            role="tool",
            content=observation,
        )
        return self.add_step(action, observation).add_message(tool_msg)

    def with_finish(self, finish: AgentFinish) -> AgentState:
        """Agent 完成时的状态更新。"""
        assistant_msg = LLMMessage(
            role="assistant",
            content=finish.return_values.get("text", ""),
        )
        return (
            self.add_message(assistant_msg)
            .set_status("finished")
        )

    # -- 辅助属性 --

    @property
    def last_user_input(self) -> str:
        """获取最后一条 user 消息的内容。"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""

    @property
    def has_steps(self) -> bool:
        return len(self.intermediate_steps) > 0

    @property
    def scratchpad_text(self) -> str:
        """将 intermediate_steps 格式化为 ReAct scratchpad 文本。"""
        lines = []
        for step in self.intermediate_steps:
            action = step.action
            lines.append(f"Thought: {action.thought}")
            lines.append(f"Action: {action.tool}")
            lines.append(f"Action Input: {action.tool_input}")
            lines.append(f"Observation: {step.observation}")
            lines.append("")
        return "\n".join(lines)

    @property
    def scratchpad_for_prompt(self) -> str:
        """用于 prompt 模板的 scratchpad（末尾追加 Thought 引导）。"""
        text = self.scratchpad_text
        if text:
            text += "Thought: "
        return text

    def to_messages_for_llm(self) -> List[LLMMessage]:
        """将当前状态转换为可直接送入 LLM 的消息列表。

        包含：
        1. 原始 system / user / assistant 对话消息
        2. 工具调用相关消息（assistant tool_calls + tool results）
        """
        return list(self.messages)

    # -- 内部辅助 --

    def _replace(self, **kwargs: Any) -> AgentState:
        """创建状态的不可变拷贝。"""
        current = {
            "messages": self.messages,
            "intermediate_steps": self.intermediate_steps,
            "status": self.status,
            "iteration": self.iteration,
            "metadata": self.metadata,
        }
        current.update(kwargs)
        return AgentState(**current)

    def __repr__(self) -> str:
        return (
            f"AgentState(status={self.status!r}, iteration={self.iteration}, "
            f"steps={len(self.intermediate_steps)}, messages={len(self.messages)})"
        )
