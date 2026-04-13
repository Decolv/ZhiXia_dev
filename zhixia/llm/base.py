"""LLM 引擎抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, List


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
