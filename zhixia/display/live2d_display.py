"""Live2D 眼睛 Display 实现

集成 Live2dEyeRenderer 到 DisplayOutput 接口，
支持情绪到眼睛状态的自动映射和联动。
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from zhixia.display.base import DisplayOutput, DisplayPayload
from zhixia.display.live2d_eyes import Live2dEyeRenderer

logger = logging.getLogger(__name__)

# 情绪到眼睛状态的映射
EMOTION_TO_EYE_STATE: Dict[str, str] = {
    "neutral": "neutral",
    "thinking": "thinking",
    "happy": "happy",
    "joy": "happy",
    "excited": "surprised",
    "working": "working",
    "busy": "working",
    "sad": "sad",
    "sorry": "sad",
    "surprised": "surprised",
    "shocked": "surprised",
    "angry": "working",
    "love": "happy",
    "warm": "happy",
}


class Live2dEyeDisplay(DisplayOutput):
    """Live2D 眼睛显示实现。

    在独立窗口中展示动态眼睛，根据情绪和交互状态自动变化。
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        window_width: int = 300,
        window_height: int = 200,
        auto_start: bool = True,
    ) -> None:
        self.renderer = Live2dEyeRenderer(
            config_path=config_path,
            window_width=window_width,
            window_height=window_height,
        )
        self._is_thinking = False
        self._current_emotion = "neutral"

        if auto_start:
            self.start()

    def start(self) -> None:
        """启动眼睛渲染器。"""
        self.renderer.start()
        self.renderer.set_state("neutral")
        logger.info("Live2D 眼睛显示已启动")

    def stop(self) -> None:
        """停止眼睛渲染器。"""
        self.renderer.stop()
        logger.info("Live2D 眼睛显示已停止")

    def show(self, payload: DisplayPayload) -> None:
        """根据 DisplayPayload 更新眼睛状态。"""
        emotion = payload.emotion or "neutral"
        self._current_emotion = emotion

        # 优先使用显式指定的 eye_state
        if payload.eye_state and payload.eye_state != "auto":
            self.set_eye_state(payload.eye_state)
        else:
            # 根据情绪自动映射
            if payload.is_thinking:
                self.set_eye_state("thinking")
            else:
                eye_state = EMOTION_TO_EYE_STATE.get(emotion, "neutral")
                self.set_eye_state(eye_state)

        # 强制眨眼
        if payload.blink_override is True:
            self.force_eye_blink()

    def clear(self) -> None:
        """重置眼睛状态到默认。"""
        self._is_thinking = False
        self._current_emotion = "neutral"
        self.renderer.set_state("neutral")

    def update_thinking(self, is_thinking: bool, thinking_text: str = "") -> None:
        """更新思考状态。"""
        self._is_thinking = is_thinking
        if is_thinking:
            self.set_eye_state("thinking")
        else:
            # 恢复到情绪对应的状态
            eye_state = EMOTION_TO_EYE_STATE.get(self._current_emotion, "neutral")
            self.set_eye_state(eye_state)

    def set_eye_state(self, state: str) -> None:
        """设置眼睛状态。"""
        self.renderer.set_state(state)
        logger.debug("眼睛状态: %s", state)

    def set_eye_emotion(self, emotion: str) -> None:
        """设置眼睛情绪（自动映射到状态）。"""
        self._current_emotion = emotion
        state = EMOTION_TO_EYE_STATE.get(emotion, "neutral")
        self.set_eye_state(state)

    def force_eye_blink(self) -> None:
        """强制眨眼。"""
        self.renderer.force_blink()
        logger.debug("强制眨眼")
