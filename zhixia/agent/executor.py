"""AgentExecutor — Agent 执行引擎（ReAct / ToolCalling 循环）

参照 LangChain AgentExecutor，但做了流式化和状态机重构：
1. 支持流式输出：每步决策、每次工具调用都可以实时观测。
2. 状态驱动：使用不可变的 AgentState，便于调试和恢复。
3. 双模式兼容：既可以执行 ReActAgent，也可以执行 ToolCallingAgent。
4. 容错设计：工具调用失败不会导致崩溃，错误信息作为 Observation 返回。

执行流程：
    state = AgentState(messages=[...])
    for step in executor.stream(state):
        # step 是 AgentState 的中间状态，可实时展示
        print(step.status)
    # 最终 state.status == "finished"

公共接口：
    executor.invoke(state) -> AgentState
    executor.stream(state) -> Iterator[AgentState]
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional, Union

from zhixia.agent.base import AgentAction, AgentDecision, AgentFinish, BaseAgent
from zhixia.agent.callbacks import CallbackManager
from zhixia.agent.runnable import RunnableConfig
from zhixia.agent.runnable import Runnable
from zhixia.agent.state import AgentState
from zhixia.agent.tool import ToolRegistry

logger = logging.getLogger(__name__)


class AgentExecutor(Runnable[AgentState, AgentState]):
    """Agent 执行引擎。

    Args:
        agent: 决策核心（ReActAgent 或 ToolCallingAgent）。
        tools: 工具注册表。
        max_iterations: 最大 ReAct 循环轮数（防止无限循环）。
        early_stopping_method: 达到 max_iterations 时的处理策略。
            - "force": 强制返回当前累积的文本作为最终答案。
            - "raise": 抛出异常（默认）。
    """

    def __init__(
        self,
        agent: BaseAgent,
        tools: ToolRegistry,
        max_iterations: int = 5,
        early_stopping_method: str = "raise",
    ) -> None:
        self.agent = agent
        self.tools = tools
        self.max_iterations = max_iterations
        self.early_stopping_method = early_stopping_method

    @property
    def name(self) -> str:
        return f"AgentExecutor({self.agent.__class__.__name__})"

    def _invoke(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """非流式执行：运行完整循环后返回最终状态。"""
        final_state = state
        for step_state in self._run_loop(final_state, config):
            final_state = step_state
        return final_state

    def _stream(self, state: AgentState, config: RunnableConfig) -> Iterator[AgentState]:
        """流式执行：每完成一步 yield 一次当前状态。"""
        yield from self._run_loop(state, config)

    def _run_loop(
        self, state: AgentState, config: RunnableConfig
    ) -> Iterator[AgentState]:
        """核心执行循环。

        每次迭代：
        1. Agent.plan() → AgentDecision (Action 或 Finish)
        2. 如果是 Finish → 结束
        3. 如果是 Action → 执行工具 → Observation → 更新 state → 继续
        """
        callbacks = config.callbacks
        current = state.set_status("thinking")

        for iteration in range(self.max_iterations):
            current = current.set_status("thinking")
            yield current

            # ---- 1. Agent 决策 ----
            user_input = current.last_user_input
            try:
                decision: AgentDecision = self.agent.plan(
                    current.intermediate_steps,
                    input=user_input,
                    messages=current.to_messages_for_llm(),
                )
            except Exception as exc:
                logger.exception("Agent 决策异常")
                callbacks.on_chain_error("agent_loop", exc)
                current = current.set_status("error")
                yield current
                raise RuntimeError(f"Agent 决策失败: {exc}") from exc

            # ---- 2. 处理决策结果 ----
            if isinstance(decision, AgentFinish):
                callbacks.on_agent_finish("agent_loop", decision)
                current = current.with_finish(decision)
                yield current
                return

            # AgentAction
            action: AgentAction = decision
            callbacks.on_agent_thought("agent_loop", action.thought)
            callbacks.on_agent_action("agent_loop", action)
            current = current.set_status("tool_call")
            yield current

            # ---- 3. 执行工具 ----
            tool_run_id = callbacks.on_tool_start(action.tool, action.tool_input)
            try:
                tool = self.tools.get(action.tool)
                if tool is None:
                    observation = f"错误：没有名为 '{action.tool}' 的工具。可用工具: {list(self.tools.list_tools())}"
                    logger.warning(observation)
                else:
                    observation = tool.run(action.tool_input)
                callbacks.on_tool_end(tool_run_id, observation)
            except Exception as exc:
                observation = f"工具执行出错: {type(exc).__name__}: {exc}"
                logger.exception("工具执行异常: %s", action.tool)
                callbacks.on_tool_error(tool_run_id, exc)

            # ---- 4. 更新状态 ----
            current = current.with_tool_result(action, observation)
            logger.debug(
                "Step %d: %s -> %s",
                iteration + 1,
                action.tool,
                observation[:100],
            )

        # ---- 达到最大迭代次数 ----
        if self.early_stopping_method == "force":
            # 强制收尾：把所有中间步骤的 observation 拼接作为答案
            texts = []
            for step in current.intermediate_steps:
                texts.append(step.observation)
            final_text = "\n".join(texts) if texts else "抱歉，我思考了太久，但没能找到答案。"
            finish = AgentFinish(return_values={"text": final_text})
            callbacks.on_agent_finish("agent_loop", finish)
            current = current.with_finish(finish)
            yield current
        else:
            current = current.set_status("error")
            yield current
            raise RuntimeError(
                f"Agent 达到最大迭代次数 ({self.max_iterations}) 仍未完成。"
            )


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------

class AgentRunner:
    """高层封装：将 Agent + Executor + 初始状态组装为可直接调用的对象。

    使用示例：
        runner = AgentRunner(agent_executor)
        result = runner.run("今天北京天气怎么样？")
        print(result["text"])

        # 流式
        for chunk in runner.stream("今天北京天气怎么样？"):
            print(chunk, end="")
    """

    def __init__(
        self,
        executor: AgentExecutor,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.executor = executor
        self.system_prompt = system_prompt or (
            "你是「小匣」，一个温暖有趣的智能助手。"
        )

    def run(
        self,
        user_input: str,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """单次运行，返回最终结果的字典。"""
        config = config or RunnableConfig()
        state = self._build_initial_state(user_input)
        final_state = self.executor.invoke(state, config)

        # 提取最终答案
        if final_state.status == "finished" and final_state.intermediate_steps:
            # 最后一步的 observation 或 finish 中的文本
            for msg in reversed(final_state.messages):
                if msg.role == "assistant":
                    return {"text": msg.content, "status": "finished"}

        # 兜底：返回最后一条 assistant 消息或空
        for msg in reversed(final_state.messages):
            if msg.role == "assistant":
                return {"text": msg.content, "status": final_state.status}
        return {"text": "", "status": final_state.status}

    def stream(
        self,
        user_input: str,
        config: Optional[RunnableConfig] = None,
    ) -> Iterator[str]:
        """流式运行：yield Agent 的思考过程和最终答案文本片段。

        注意：这不是真正的 token 级流式，而是 step 级流式。
        要获得 token 级流式，需要在 LLM 层实现。
        """
        config = config or RunnableConfig()
        state = self._build_initial_state(user_input)

        for step_state in self.executor.stream(state, config):
            if step_state.status == "thinking":
                # 可以 yield 思考状态的标记
                pass
            elif step_state.status == "tool_call":
                # 工具调用中
                pass
            elif step_state.status == "finished":
                # 找到最终答案并 yield
                for msg in reversed(step_state.messages):
                    if msg.role == "assistant":
                        yield msg.content
                        return

    def _build_initial_state(self, user_input: str) -> AgentState:
        """根据用户输入构建初始 AgentState。"""
        from zhixia.llm.base import LLMMessage

        messages = []
        if self.system_prompt:
            messages.append(LLMMessage(role="system", content=self.system_prompt))
        messages.append(LLMMessage(role="user", content=user_input))
        return AgentState(messages=messages)
