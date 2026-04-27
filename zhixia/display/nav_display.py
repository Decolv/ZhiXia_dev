"""导航 Display 实现

集成 NavUIRenderer 到 DisplayOutput 接口，
支持导航界面的显示/隐藏，以及与 Live2D 眼睛的联动。
"""

import logging
from typing import Any, Dict

from zhixia.display.base import DisplayOutput, DisplayPayload
from zhixia.display.live2d_display import Live2dEyeDisplay
from zhixia.display.nav_ui import NavUIRenderer

logger = logging.getLogger(__name__)


class NavDisplay(DisplayOutput):
    """导航 Display 实现。

    管理 Live2D 眼睛和导航界面的切换：
    - 正常状态：显示 Live2D 眼睛
    - 导航状态：隐藏眼睛，显示导航界面
    - 导航完成：隐藏导航界面，恢复眼睛
    """

    def __init__(
        self,
        eye_display: Live2dEyeDisplay,
        nav_renderer: NavUIRenderer,
    ) -> None:
        self.eye_display = eye_display
        self.nav_renderer = nav_renderer
        self._showing_nav = False

    # -- DisplayOutput 接口 --

    def show(self, payload: DisplayPayload) -> None:
        """根据 DisplayPayload 更新显示状态。"""
        # 处理导航界面控制
        if payload.show_nav_ui and payload.nav_data:
            self.show_navigation_ui(payload.nav_data)
        elif payload.nav_completed:
            self.hide_navigation_ui()

        # 始终将常规显示传递给眼睛
        if not self._showing_nav:
            self.eye_display.show(payload)

    def clear(self) -> None:
        """清除显示。"""
        self.eye_display.clear()
        self.nav_renderer.hide()
        self._showing_nav = False

    def update_thinking(self, is_thinking: bool, thinking_text: str = "") -> None:
        """更新思考状态。"""
        self.eye_display.update_thinking(is_thinking, thinking_text)

    def set_eye_state(self, state: str) -> None:
        """设置眼睛状态。"""
        if not self._showing_nav:
            self.eye_display.set_eye_state(state)

    def set_eye_emotion(self, emotion: str) -> None:
        """设置眼睛情绪。"""
        if not self._showing_nav:
            self.eye_display.set_eye_emotion(emotion)

    def force_eye_blink(self) -> None:
        """强制眨眼。"""
        if not self._showing_nav:
            self.eye_display.force_eye_blink()

    # -- 导航界面专用方法 --

    def show_navigation_ui(self, nav_data: Dict[str, Any]) -> None:
        """展示导航界面，暂停眼睛显示。"""
        self._showing_nav = True
        self.nav_renderer.show(nav_data)
        # 暂停眼睛（不停止线程，只是隐藏窗口）
        logger.info("导航界面已展示，目的地: %s", nav_data.get("destination", "未知"))

    def hide_navigation_ui(self) -> None:
        """隐藏导航界面，恢复眼睛显示。"""
        self._showing_nav = False
        self.nav_renderer.hide()
        # 恢复眼睛并眨眼
        self.eye_display.force_eye_blink()
        logger.info("导航界面已关闭，恢复眼睛显示")

    def stop(self) -> None:
        """停止所有显示。"""
        self.eye_display.stop()
        self.nav_renderer.stop()
