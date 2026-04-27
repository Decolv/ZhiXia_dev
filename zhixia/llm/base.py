"""LLM 引擎抽象基类

新增：支持工具绑定的抽象方法，为 ToolCallingAgent 提供底层支持。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional


@dataclass
class LLMMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str


@dataclass
class StructuredOutput:
    """解析后的 LLM 输出"""
    text: str
    emotion: str = "neutral"
    metadata: dict = field(default_factory=dict)


class LLMEngine(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def chat(self, messages: List[LLMMessage], max_new_tokens: int = 32) -> str:
        """返回原始 LLM 输出字符串。"""

    def stream_chat(self, messages: List[LLMMessage], max_new_tokens: int = 32) -> Generator[str, None, None]:
        """流式输出 token，默认回退到非流式实现。子类应覆盖此方法。"""
        yield self.chat(messages, max_new_tokens)

    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        ...

    def shutdown(self) -> None:
        """释放资源，默认空操作。"""

    # -- 工具调用扩展（可选实现） --

    def bind_tools(self, tools: List[Any]) -> "LLMEngine":
        """绑定工具描述，返回支持工具调用的包装器。

        默认实现：返回自身（文本模式，由上层 Agent 在 prompt 中注入工具描述）。
        支持原生 function calling 的子类（如 CloudLLM）应覆盖此方法。
        """
        return self

    def chat_with_tools(
        self,
        messages: List[LLMMessage],
        tool_schemas: List[Dict[str, Any]],
        max_new_tokens: int = 32,
    ) -> str:
        """使用原生 tool calling 调用 LLM。

        默认回退到普通 chat（由上层 BoundLLM 处理 prompt 注入）。
        """
        return self.chat(messages, max_new_tokens)
