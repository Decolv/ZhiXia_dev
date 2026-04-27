"""空显示实现（日志输出，预留接口）"""

import logging

from zhixia.display.base import DisplayOutput, DisplayPayload

logger = logging.getLogger(__name__)


class NullDisplay(DisplayOutput):
    """空显示实现，用于日志输出和接口预留。

    支持思考状态显示和播报，为实际显示实现提供参考。
    """

    def __init__(self):
        self._is_thinking = False
        self._current_thinking_text = ""

    def show(self, payload: DisplayPayload) -> None:
        """显示内容，支持思考文本播报。"""
        if payload.is_thinking and payload.thinking_text:
            logger.debug("Display [thinking] [emotion=%s]: %s", payload.emotion, payload.thinking_text)
            self._current_thinking_text = payload.thinking_text
        else:
            logger.debug("Display [emotion=%s]: %s", payload.emotion, payload.text)
            if payload.thinking_text:
                logger.debug("Display [thinking_text]: %s", payload.thinking_text)

    def clear(self) -> None:
        """清除显示内容。"""
        self._is_thinking = False
        self._current_thinking_text = ""

    def update_thinking(self, is_thinking: bool, thinking_text: str = "") -> None:
        """更新思考状态和内容。

        Args:
            is_thinking: 是否正在思考
            thinking_text: 思考内容文本，用于播报给用户
        """
        self._is_thinking = is_thinking
        if is_thinking:
            if thinking_text:
                logger.debug("Display: thinking... [%s]", thinking_text)
                self._current_thinking_text = thinking_text
            else:
                logger.debug("Display: thinking...")
        else:
            self._current_thinking_text = ""
            logger.debug("Display: thinking finished")

    def set_eye_state(self, state: str) -> None:
        """设置眼睛状态（空实现）。"""
        logger.debug("Eye state set to: %s", state)

    def set_eye_emotion(self, emotion: str) -> None:
        """设置眼睛情绪（空实现）。"""
        logger.debug("Eye emotion set to: %s", emotion)

    def force_eye_blink(self) -> None:
        """强制眼睛眨眼（空实现）。"""
        pass
