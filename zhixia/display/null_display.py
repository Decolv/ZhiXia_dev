"""空显示实现（日志输出，预留接口）"""

import logging

from zhixia.display.base import DisplayOutput, DisplayPayload

logger = logging.getLogger(__name__)


class NullDisplay(DisplayOutput):

    def show(self, payload: DisplayPayload) -> None:
        logger.debug("Display [emotion=%s]: %s", payload.emotion, payload.text)

    def clear(self) -> None:
        pass

    def update_thinking(self, is_thinking: bool) -> None:
        if is_thinking:
            logger.debug("Display: thinking...")
