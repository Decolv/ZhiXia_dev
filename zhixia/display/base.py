"""显示输出抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class DisplayPayload:
    text: str
    emotion: str = "neutral"
    is_thinking: bool = False
    thinking_text: str = ""  # 思考内容，用于播报模型思考过程
    # 图片支持：可以是单个图片路径或图片路径列表
    images: Optional[List[Union[str, Path]]] = None
    # 图片标题/说明
    image_captions: Optional[List[str]] = None
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
