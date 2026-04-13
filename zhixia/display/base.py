"""显示输出抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DisplayPayload:
    text: str
    emotion: str = "neutral"
    is_thinking: bool = False
    metadata: Dict = field(default_factory=dict)


class DisplayOutput(ABC):

    @abstractmethod
    def show(self, payload: DisplayPayload) -> None:
        """更新显示内容。"""

    @abstractmethod
    def clear(self) -> None:
        """清除显示。"""

    def update_thinking(self, is_thinking: bool) -> None:
        """显示/隐藏思考指示器。默认空操作。"""
