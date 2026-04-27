"""Callbacks 回调系统 —— Agent 执行过程的可观测性

参照 LangChain Callbacks / Tracing 设计，但极度精简：
1. 事件驱动：Agent 思考、工具调用、LLM 生成等关键节点触发回调。
2. 流式友好：支持在回调中实时更新 UI（如 display 模块）。
3. 零依赖：不依赖 langchain-core。

事件类型：
    on_chain_start / on_chain_stream / on_chain_end / on_chain_error
    on_tool_start / on_tool_end / on_tool_error
    on_llm_start / on_llm_new_token / on_llm_end / on_llm_error
    on_agent_action / on_agent_finish / on_agent_thought

使用示例：
    manager = CallbackManager(handlers=[DisplayHandler(), LoggingHandler()])
    config = RunnableConfig(callbacks=manager)
    result = agent.invoke(input, config)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseCallbackHandler — 回调处理器基类
# ---------------------------------------------------------------------------

class BaseCallbackHandler(ABC):
    """单个回调处理器。

    子类选择性地覆盖感兴趣的事件方法即可。
    所有方法都有默认空实现，不强制覆盖。
    """

    # -- Chain 事件 --
    def on_chain_start(
        self, runnable_name: str, input: Any, run_id: str, **kwargs: Any
    ) -> None:
        pass

    def on_chain_stream(self, run_id: str, chunk: Any, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, run_id: str, output: Any, **kwargs: Any) -> None:
        pass

    def on_chain_error(self, run_id: str, error: Exception, **kwargs: Any) -> None:
        pass

    # -- Tool 事件 --
    def on_tool_start(
        self, tool_name: str, tool_input: Any, run_id: str, **kwargs: Any
    ) -> None:
        pass

    def on_tool_end(self, run_id: str, output: str, **kwargs: Any) -> None:
        pass

    def on_tool_error(self, run_id: str, error: Exception, **kwargs: Any) -> None:
        pass

    # -- LLM 事件 --
    def on_llm_start(
        self, messages: List[Any], run_id: str, **kwargs: Any
    ) -> None:
        pass

    def on_llm_new_token(self, run_id: str, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_end(self, run_id: str, output: str, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, run_id: str, error: Exception, **kwargs: Any) -> None:
        pass

    # -- Agent 专用事件 --
    def on_agent_thought(self, run_id: str, thought: str, **kwargs: Any) -> None:
        """Agent 产生 Thought（可用于流式展示思考过程）。"""
        pass

    def on_thinking_start(self, run_id: str, **kwargs: Any) -> None:
        """Agent 开始思考时调用。"""
        pass

    def on_thinking_end(self, run_id: str, **kwargs: Any) -> None:
        """Agent 结束思考时调用。"""
        pass

    def on_agent_action(
        self, run_id: str, action: Any, **kwargs: Any
    ) -> None:
        """Agent 决定调用工具。"""
        pass

    def on_agent_finish(
        self, run_id: str, finish: Any, **kwargs: Any
    ) -> None:
        """Agent 完成，给出最终答案。"""
        pass


# ---------------------------------------------------------------------------
# CallbackManager — 管理多个 Handler 的聚合器
# ---------------------------------------------------------------------------

class CallbackManager:
    """回调管理器：将事件广播给所有注册的 Handler。"""

    def __init__(self, handlers: Optional[List[BaseCallbackHandler]] = None) -> None:
        self.handlers = handlers or []
        self._run_counter = 0

    def _next_run_id(self) -> str:
        self._run_counter += 1
        return f"run_{self._run_counter}_{time.time():.6f}"

    # -- Chain 事件广播 --

    def on_chain_start(self, runnable: Any, input: Any, config: Any) -> str:
        run_id = self._next_run_id()
        name = getattr(runnable, "name", runnable.__class__.__name__)
        for h in self.handlers:
            try:
                h.on_chain_start(name, input, run_id)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)
        return run_id

    def on_chain_stream(self, run_id: str, chunk: Any, config: Any) -> None:
        for h in self.handlers:
            try:
                h.on_chain_stream(run_id, chunk)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_chain_end(self, run_id: str, output: Any, config: Any) -> None:
        for h in self.handlers:
            try:
                h.on_chain_end(run_id, output)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_chain_error(self, run_id: str, error: Exception, config: Any) -> None:
        for h in self.handlers:
            try:
                h.on_chain_error(run_id, error)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    # -- Tool 事件广播 --

    def on_tool_start(self, tool_name: str, tool_input: Any) -> str:
        run_id = self._next_run_id()
        for h in self.handlers:
            try:
                h.on_tool_start(tool_name, tool_input, run_id)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)
        return run_id

    def on_tool_end(self, run_id: str, output: str) -> None:
        for h in self.handlers:
            try:
                h.on_tool_end(run_id, output)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_tool_error(self, run_id: str, error: Exception) -> None:
        for h in self.handlers:
            try:
                h.on_tool_error(run_id, error)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    # -- LLM 事件广播 --

    def on_llm_start(self, messages: List[Any]) -> str:
        run_id = self._next_run_id()
        for h in self.handlers:
            try:
                h.on_llm_start(messages, run_id)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)
        return run_id

    def on_llm_new_token(self, run_id: str, token: str) -> None:
        for h in self.handlers:
            try:
                h.on_llm_new_token(run_id, token)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_llm_end(self, run_id: str, output: str) -> None:
        for h in self.handlers:
            try:
                h.on_llm_end(run_id, output)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_llm_error(self, run_id: str, error: Exception) -> None:
        for h in self.handlers:
            try:
                h.on_llm_error(run_id, error)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    # -- Agent 事件广播 --

    def on_agent_thought(self, run_id: str, thought: str) -> None:
        for h in self.handlers:
            try:
                h.on_agent_thought(run_id, thought)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_thinking_start(self, run_id: str) -> None:
        for h in self.handlers:
            try:
                h.on_thinking_start(run_id)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_thinking_end(self, run_id: str) -> None:
        for h in self.handlers:
            try:
                h.on_thinking_end(run_id)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_agent_action(self, run_id: str, action: Any) -> None:
        for h in self.handlers:
            try:
                h.on_agent_action(run_id, action)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)

    def on_agent_finish(self, run_id: str, finish: Any) -> None:
        for h in self.handlers:
            try:
                h.on_agent_finish(run_id, finish)
            except Exception:
                logger.debug("Callback handler error", exc_info=True)


# ---------------------------------------------------------------------------
# 内置 Handler 实现
# ---------------------------------------------------------------------------

class LoggingHandler(BaseCallbackHandler):
    """将 Agent 执行过程记录到日志。"""

    def on_chain_start(self, runnable_name: str, input: Any, run_id: str, **kwargs: Any) -> None:
        logger.info("[%s] Chain start: %s", run_id, runnable_name)

    def on_tool_start(self, tool_name: str, tool_input: Any, run_id: str, **kwargs: Any) -> None:
        logger.info("[%s] Tool call: %s(%s)", run_id, tool_name, tool_input)

    def on_tool_end(self, run_id: str, output: str, **kwargs: Any) -> None:
        logger.info("[%s] Tool result: %s", run_id, output[:200])

    def on_agent_thought(self, run_id: str, thought: str, **kwargs: Any) -> None:
        logger.info("[%s] Thought: %s", run_id, thought[:200])

    def on_agent_action(self, run_id: str, action: Any, **kwargs: Any) -> None:
        from zhixia.agent.base import AgentAction

        if isinstance(action, AgentAction):
            logger.info("[%s] Action: %s(%s)", run_id, action.tool, action.tool_input)

    def on_agent_finish(self, run_id: str, finish: Any, **kwargs: Any) -> None:
        from zhixia.agent.base import AgentFinish

        if isinstance(finish, AgentFinish):
            text = finish.return_values.get("text", "")
            logger.info("[%s] Finish: %s", run_id, text[:200])


class StreamingDisplayHandler(BaseCallbackHandler):
    """流式处理 Handler：将 Agent 思考过程和最终结果实时输出。

    可接入项目的 Display 模块，实现"小匣正在思考..."的实时展示。
    """

    def __init__(self, display=None, show_thoughts: bool = True) -> None:
        self.display = display
        self.show_thoughts = show_thoughts
        self._thought_buffer = ""

    def on_thinking_start(self, run_id: str, **kwargs: Any) -> None:
        """开始思考时更新显示状态。"""
        if self.display:
            from zhixia.display.base import DisplayPayload
            self.display.update_thinking(True)
            self.display.show(DisplayPayload(
                text="",
                emotion="thinking",
                is_thinking=True,
                thinking_text="正在思考..."
            ))
        if self.show_thoughts:
            print("\n[思考] 开始思考...")

    def on_thinking_end(self, run_id: str, **kwargs: Any) -> None:
        """结束思考时更新显示状态。"""
        if self.display:
            self.display.update_thinking(False)
        if self.show_thoughts:
            print("\n[思考] 思考完成")

    def on_agent_thought(self, run_id: str, thought: str, **kwargs: Any) -> None:
        self._thought_buffer += thought
        if self.show_thoughts:
            print(f"[思考] {thought}", end="", flush=True)

    def on_agent_action(self, run_id: str, action: Any, **kwargs: Any) -> None:
        from zhixia.agent.base import AgentAction

        if isinstance(action, AgentAction):
            action_msg = f"正在调用 {action.tool} 工具..."
            if self.display:
                from zhixia.display.base import DisplayPayload
                self.display.show(DisplayPayload(
                    text="",
                    emotion="working",
                    is_thinking=True,
                    thinking_text=action_msg
                ))
            if self.show_thoughts:
                print(f"\n[工具] 调用 {action.tool}({action.tool_input})")

    def on_agent_finish(self, run_id: str, finish: Any, **kwargs: Any) -> None:
        from zhixia.agent.base import AgentFinish

        if isinstance(finish, AgentFinish):
            text = finish.return_values.get("text", "")
            if self.display:
                from zhixia.display.base import DisplayPayload
                self.display.show(DisplayPayload(
                    text=text,
                    emotion="neutral",
                    is_thinking=False,
                    thinking_text=""
                ))
            if self.show_thoughts:
                print(f"\n[回答] {text}")
