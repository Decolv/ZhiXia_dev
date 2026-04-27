"""显示输出抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
    # Live2D 眼睛控制
    eye_state: str = ""  # 眼睛状态：auto/neutral/thinking/happy/working/sad/surprised
    blink_override: Optional[bool] = None  # 是否强制眨眼
    # 导航界面控制
    show_nav_ui: bool = False  # 是否展示导航界面
    nav_data: Optional[Dict[str, Any]] = None  # 导航数据（地点信息、路线等）
    nav_completed: bool = False  # 导航是否完成（恢复眼睛）
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
        pass

    def set_eye_state(self, state: str) -> None:
        """设置眼睛状态。默认空操作。"""
        pass

    def set_eye_emotion(self, emotion: str) -> None:
        """设置眼睛情绪表情。默认空操作。"""
        pass

    def force_eye_blink(self) -> None:
        """强制眼睛眨眼。默认空操作。"""
        pass

    def show_navigation_ui(self, nav_data: Dict[str, Any]) -> None:
        """展示导航界面。"""
        pass

    def hide_navigation_ui(self) -> None:
        """隐藏导航界面。"""
        pass
