"""ReAct Agent 实现 — Thought → Action → Observation → ... → Final Answer

对应 LangChain 的 ReActSingleActionAgent / ZeroShotAgent。
特点：
- 单 Action：每次 LLM 调用只决策一个 Action。
- 纯字符串交互：兼容任何遵循 base.py 接口的 LLMEngine。
- 无外部依赖：不依赖 langchain 包。
- **新增**：继承 Runnable 协议，支持 invoke / stream / LCEL 管道组合。
- **新增**：支持流式 thought 输出（通过 callback 实时展示思考过程）。
"""

from typing import Any, List, Optional

from zhixia.agent.base import (
    AgentAction,
    AgentDecision,
    AgentFinish,
    AgentStep,
    BaseAgent,
)
from zhixia.agent.parser import ReActOutputParser
from zhixia.agent.prompts import ReActPromptTemplate
from zhixia.agent.runnable import RunnableConfig
from zhixia.agent.tool import ToolRegistry
from zhixia.llm.base import LLMEngine, LLMMessage


class ReActAgent(BaseAgent):
    """ReAct 风格的 Agent。

    Args:
        llm_engine: 符合 LLMEngine 接口的语言模型。
        tools: 工具注册表。
        prompt_template: 可选的自定义 prompt 模板。
        output_parser: 可选的自定义输出解析器。
        max_new_tokens: Agent 每次决策的 max_new_tokens（默认 256）。
        stop_sequences: LLM 生成时的停止序列（遇到 Observation 前停止）。
    """

    def __init__(
        self,
        llm_engine: LLMEngine,
        tools: ToolRegistry,
        prompt_template: Optional[ReActPromptTemplate] = None,
        output_parser: Optional[ReActOutputParser] = None,
        max_new_tokens: int = 256,
        stop_sequences: Optional[List[str]] = None,
    ) -> None:
        self.llm = llm_engine
        self.tools = tools
        self.prompt_template = prompt_template or ReActPromptTemplate()
        self.output_parser = output_parser or ReActOutputParser()
        self.max_new_tokens = max_new_tokens
        self.stop_sequences = stop_sequences or ["Observation:"]

    @property
    def name(self) -> str:
        return "ReActAgent"

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def return_values(self) -> List[str]:
        return ["text"]

    def plan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks=None,
        **kwargs: Any,
    ) -> AgentDecision:
        """ReAct 核心决策：组装 prompt → 调用 LLM → 解析输出。

        Args:
            intermediate_steps: 历史步骤（用于构建 scratchpad）。
            **kwargs: 必须包含 "input"（用户当前问题）。
                      可选包含 "system_prompt" / "rag_context" 等。
        """
        user_input = kwargs.get("input", "")
        if not user_input:
            raise ValueError("ReActAgent.plan() 需要 kwargs['input']")

        # 1. 构建 scratchpad
        scratchpad = self.prompt_template.build_scratchpad(intermediate_steps)
        # 如果 scratchpad 非空，说明前面还有步骤，需要继续拼接
        if scratchpad:
            scratchpad += "Thought: "

        # 2. 组装完整 prompt
        tool_desc = self.tools.format_tool_descriptions()
        prompt_text = self.prompt_template.format(
            tool_descriptions=tool_desc,
            input=user_input,
            agent_scratchpad=scratchpad,
        )

        # 3. 可选：追加结构化输出格式指令
        messages = [LLMMessage(role="user", content=prompt_text)]

        # 4. 调用 LLM（非流式，因为 Agent 决策需要完整输出）
        response = self.llm.chat(messages, max_new_tokens=self.max_new_tokens)

        # 5. 解析输出
        decision = self.output_parser.parse(response)

        # 6. 如果决策是 Action，但工具不存在 → 转为 FinalAnswer（容错）
        if isinstance(decision, AgentAction):
            if decision.tool not in self.tools:
                return AgentFinish(
                    return_values={
                        "text": f"抱歉，我没有 '{decision.tool}' 这个工具。"
                        f"让我直接回答：{decision.thought or '我不太清楚该怎么帮你。'}"
                    },
                    log=response,
                )
        return decision

    # -- Runnable 协议增强：流式思考 --

    def stream_plan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks=None,
        **kwargs: Any,
    ):
        """流式决策：yield 思考过程中的 token，最终返回决策结果。

        注意：这要求底层 LLM 支持 stream_chat。Agent 的决策结果
        只有在 LLM 输出完整后才能真正确定，所以 stream 主要用于
        实时展示 "Agent 正在思考..." 的过程。
        """
        user_input = kwargs.get("input", "")
        if not user_input:
            raise ValueError("ReActAgent.stream_plan() 需要 kwargs['input']")

        scratchpad = self.prompt_template.build_scratchpad(intermediate_steps)
        if scratchpad:
            scratchpad += "Thought: "

        tool_desc = self.tools.format_tool_descriptions()
        prompt_text = self.prompt_template.format(
            tool_descriptions=tool_desc,
            input=user_input,
            agent_scratchpad=scratchpad,
        )

        messages = [LLMMessage(role="user", content=prompt_text)]

        # 流式收集
        buffer = []
        for token in self.llm.stream_chat(messages, max_new_tokens=self.max_new_tokens):
            buffer.append(token)
            # 如果 token 包含 Thought 开头，可以实时 yield
            if "Thought" in token or buffer:
                # 简单策略：yield 所有 token
                yield token

        response = "".join(buffer)
        decision = self.output_parser.parse(response)

        # 容错处理
        if isinstance(decision, AgentAction) and decision.tool not in self.tools:
            decision = AgentFinish(
                return_values={
                    "text": f"抱歉，我没有 '{decision.tool}' 这个工具。"
                    f"让我直接回答：{decision.thought or '我不太清楚该怎么帮你。'}"
                },
                log=response,
            )
        yield decision
