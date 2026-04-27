"""导航响应后处理器 — 解析导航数据并展示导航界面

此处理器由导航卡片注册，主机仅负责调用。
负责：
1. 解析 __NAV_DATA__ 标记
2. 获取导航数据
3. 展示导航界面
4. 定时自动关闭
"""

import logging
import re
import threading
from typing import Any, Dict, Optional, Tuple

from zhixia.core.card_base import ResponsePostProcessor
from zhixia.display.base import DisplayOutput

logger = logging.getLogger(__name__)


class NavResponseProcessor(ResponsePostProcessor):
    """导航响应后处理器。

    由校园导航卡片注册到主机，处理导航相关的响应。
    """

    def __init__(self, display: DisplayOutput, nav_data_provider: Any) -> None:
        """
        Args:
            display: 显示输出接口
            nav_data_provider: 提供导航数据的对象（需有 _build_nav_data 方法）
        """
        self.display = display
        self.nav_data_provider = nav_data_provider
        self._auto_hide_timer: Optional[threading.Timer] = None

    @property
    def name(self) -> str:
        return "nav_response_processor"

    def process(self, response_text: str) -> Tuple[str, bool]:
        """处理响应，解析导航数据并展示界面。

        Returns:
            (cleaned_text, is_handled)
        """
        if "__NAV_DATA__" not in response_text:
            return response_text, False

        # 解析导航标记
        nav_match = re.search(r'__NAV_DATA__(.*?)__', response_text)
        if not nav_match:
            return response_text, False

        location_name = nav_match.group(1)
        nav_data = self._get_nav_data(location_name)

        if nav_data:
            # 展示导航界面
            self.display.show_navigation_ui(nav_data)

            # 5秒后自动关闭
            self._cancel_auto_hide()
            self._auto_hide_timer = threading.Timer(
                5.0, self.display.hide_navigation_ui
            )
            self._auto_hide_timer.daemon = True
            self._auto_hide_timer.start()

            logger.info("导航界面已展示: %s", location_name)

        # 清理元数据标记
        cleaned = re.sub(r'__NAV_DATA__.*?__\n\n', '', response_text)
        return cleaned, True

    def _get_nav_data(self, location_name: str) -> Optional[Dict[str, str]]:
        """从导航工具获取导航数据。"""
        if hasattr(self.nav_data_provider, "_build_nav_data"):
            return self.nav_data_provider._build_nav_data(location_name)
        return None

    def _cancel_auto_hide(self) -> None:
        """取消之前的自动关闭定时器。"""
        if self._auto_hide_timer and self._auto_hide_timer.is_alive():
            self._auto_hide_timer.cancel()
            self._auto_hide_timer = None

    def cleanup(self) -> None:
        """清理资源。"""
        self._cancel_auto_hide()
        self.display.hide_navigation_ui()
