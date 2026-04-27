"""导航界面渲染器 — 使用 Pygame 渲染动画式导航界面

在独立窗口中展示校园导航信息，支持：
- 路线动画效果
- 地图标注
- 信息逐行展示
- 目的地标记闪烁
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from zhixia.display.pygame_manager import PygameManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NavUIRenderer — 导航界面渲染器
# ---------------------------------------------------------------------------

class NavUIRenderer:
    """导航界面渲染器。

    使用 Pygame 在独立窗口中渲染动画式导航界面。
    """

    # 导航界面配色
    BG_COLOR = (15, 25, 35)
    CARD_BG = (30, 45, 60)
    TEXT_COLOR = (220, 230, 240)
    ACCENT_COLOR = (80, 180, 255)
    HIGHLIGHT_COLOR = (100, 220, 180)
    DIM_TEXT_COLOR = (140, 160, 180)
    PATH_COLOR = (80, 180, 255)
    ARROW_COLOR = (255, 200, 80)

    def __init__(
        self,
        window_width: int = 600,
        window_height: int = 500,
    ) -> None:
        self.window_width = window_width
        self.window_height = window_height

        # 状态
        self.current_nav_data: Optional[Dict[str, Any]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 动画状态
        self._animation_progress = 0.0
        self._line_index = 0
        self._line_reveal_progress = 0.0
        self._blink_phase = 0.0
        self._start_time = 0.0

    def show(self, nav_data: Dict[str, Any]) -> None:
        """展示导航界面。"""
        with self._lock:
            self.current_nav_data = nav_data
            self._animation_progress = 0.0
            self._line_index = 0
            self._line_reveal_progress = 0.0
            self._start_time = time.monotonic()

    def hide(self) -> None:
        """隐藏导航界面。"""
        with self._lock:
            self.current_nav_data = None
            self._animation_progress = 0.0

    def is_visible(self) -> bool:
        """检查导航界面是否正在显示。"""
        with self._lock:
            return self.current_nav_data is not None

    def start(self) -> None:
        """启动渲染线程。"""
        if self._running:
            return

        if not PygameManager.init():
            logger.error("Pygame 初始化失败，导航渲染器不可用。")
            return

        os.environ["SDL_VIDEO_WINDOW_POS"] = "center"

        import pygame
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.NOFRAME | pygame.RESIZABLE
        )
        pygame.display.set_caption("知匣 - 导航")

        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True, name="NavUIRenderer")
        self._thread.start()
        logger.info("导航界面渲染器已启动")

    def stop(self) -> None:
        """停止渲染线程。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        PygameManager.quit()
        logger.info("导航界面渲染器已停止")

    def _render_loop(self) -> None:
        """渲染主循环。"""
        import pygame
        clock = pygame.time.Clock()

        # 加载字体
        try:
            font_large = pygame.font.SysFont("microsoftyahei, simhei, arial", 28, bold=True)
            font_medium = pygame.font.SysFont("microsoftyahei, simhei, arial", 20)
            font_small = pygame.font.SysFont("microsoftyahei, simhei, arial", 16)
        except Exception:
            font_large = pygame.font.Font(None, 36)
            font_medium = pygame.font.Font(None, 24)
            font_small = pygame.font.Font(None, 20)

        try:
            while self._running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                        break

                # 仅在锁内复制数据
                with self._lock:
                    nav_data = None
                    if self.current_nav_data is not None:
                        nav_data = dict(self.current_nav_data)

                # 在锁外执行渲染
                if nav_data is None:
                    self.screen.fill(self.BG_COLOR)
                    self._draw_waiting_text(self.screen, font_medium)
                else:
                    elapsed = time.monotonic() - self._start_time
                    self._update_animation(elapsed)
                    self.screen.fill(self.BG_COLOR)
                    self._draw_nav_ui(self.screen, nav_data, font_large, font_medium, font_small, elapsed)

                pygame.display.flip()
                clock.tick(30)
        except Exception as exc:
            logger.error("NavUI渲染循环异常: %s", exc)
        finally:
            self._running = False

    def _update_animation(self, elapsed: float) -> None:
        """更新动画状态。"""
        # 整体动画进度 (前3秒)
        self._animation_progress = min(1.0, elapsed / 3.0)

        # 逐行显示
        total_lines = 5  # 标题 + 目的地 + 位置 + 路线 + 周边
        line_duration = 0.5
        line_index_float = elapsed / line_duration
        self._line_index = min(total_lines, int(line_index_float))
        self._line_reveal_progress = line_index_float - int(line_index_float)

        # 闪烁相位
        self._blink_phase = elapsed * 3.0

    def _draw_waiting_text(self, screen, font) -> None:
        """绘制待机文字。"""
        import pygame

        text = font.render("等待导航请求...", True, self.DIM_TEXT_COLOR)
        text_rect = text.get_rect(center=(self.window_width // 2, self.window_height // 2))
        screen.blit(text, text_rect)

    def _draw_nav_ui(self, screen, nav_data: Dict, font_large, font_medium, font_small, elapsed: float) -> None:
        """绘制导航界面。"""
        import pygame

        padding = 30
        card_width = self.window_width - padding * 2
        card_x = padding

        # 计算卡片高度和位置
        destination = nav_data.get("destination", "未知地点")
        area = nav_data.get("area", "")
        description = nav_data.get("description", "")
        route = nav_data.get("route", "")
        nearby = nav_data.get("nearby", "")
        walk_time = nav_data.get("walk_time", "")

        # 构建行列表
        lines = []
        lines.append(("title", "校园导航", self.ACCENT_COLOR, font_large, True))
        if area:
            lines.append(("area", f"📍 {destination} · {area}", self.TEXT_COLOR, font_medium, False))
        else:
            lines.append(("dest", f"📍 {destination}", self.TEXT_COLOR, font_medium, False))
        if description:
            lines.append(("desc", description, self.DIM_TEXT_COLOR, font_small, False))
        lines.append(("divider", "", self.ACCENT_COLOR, None, False))
        if route:
            lines.append(("route_icon", "🚶", self.TEXT_COLOR, font_medium, False))
            lines.append(("route", route, self.TEXT_COLOR, font_medium, False))
        if walk_time:
            lines.append(("time", f"⏱ {walk_time}", self.HIGHLIGHT_COLOR, font_medium, False))
        if nearby:
            lines.append(("divider2", "", self.ACCENT_COLOR, None, False))
            lines.append(("nearby_label", f"🏪 周边: {nearby}", self.DIM_TEXT_COLOR, font_small, False))

        # 计算总高度
        current_y = 30
        line_heights = []
        for line_type, text, color, font, is_bold in lines:
            if line_type == "divider" or line_type == "divider2":
                h = 15
            else:
                h = font.size(text)[1] + 8
            line_heights.append(h)

        total_height = sum(line_heights)
        card_height = total_height + 40

        # 绘制背景卡片（带圆角）
        card_y = (self.window_height - card_height) // 2
        self._draw_rounded_rect(screen, card_x, card_y, card_width, card_height, self.CARD_BG, 12)

        # 绘制顶部标题栏
        title_y = card_y + 15
        pygame.draw.line(screen, self.ACCENT_COLOR,
                        (card_x + 15, title_y + 25),
                        (card_x + card_width - 15, title_y + 25), 2)

        # 绘制路线图（简单的路线可视化）
        if self._line_index >= 4:
            path_y = card_y + card_height - 50
            progress = min(1.0, self._animation_progress * 1.5)
            self._draw_path_visualization(screen, card_x + 20, path_y, card_width - 40, progress)

        # 逐行绘制文本
        draw_y = title_y + 35
        for i, (line_type, text, color, font, is_bold) in enumerate(lines):
            if i > self._line_index:
                continue

            if line_type == "divider" or line_type == "divider2":
                pygame.draw.line(screen, self.ACCENT_COLOR,
                               (card_x + 15, draw_y),
                               (card_x + card_width - 15, draw_y), 1)
                draw_y += 15
                continue

            # 逐行淡入效果
            alpha = 1.0
            if i == self._line_index:
                alpha = self._line_reveal_progress

            if font:
                text_surface = font.render(text, True, color)
                text_rect = text_surface.get_rect(x=card_x + 20, y=int(draw_y))
                screen.blit(text_surface, text_rect)

            draw_y += line_heights[i]

        # 目的地标记闪烁效果
        blink_alpha = 0.5 + 0.5 * math.sin(self._blink_phase)
        marker_color = (
            int(self.ACCENT_COLOR[0] * blink_alpha),
            int(self.ACCENT_COLOR[1] * blink_alpha),
            int(self.ACCENT_COLOR[2] * blink_alpha)
        )
        pygame.draw.circle(screen, marker_color, (card_x + card_width - 30, card_y + 30), 8)

    def _draw_rounded_rect(self, screen, x: int, y: int, w: int, h: int, color: tuple, radius: int) -> None:
        """绘制圆角矩形。"""
        import pygame

        # 简化实现：使用矩形代替圆角
        pygame.draw.rect(screen, color, (x, y, w, h), border_radius=radius)

    def _draw_path_visualization(self, screen, x: int, y: int, width: int, progress: float) -> None:
        """绘制简单的路线可视化。"""
        import pygame

        # 路径线条
        end_x = x + int(width * progress)
        pygame.draw.line(screen, self.PATH_COLOR, (x, y), (end_x, y), 4)

        # 路径上的点
        points = 5
        for i in range(points):
            px = x + int(width * i / (points - 1))
            if px <= end_x:
                alpha = 1.0 if px < end_x else progress
                point_color = (
                    int(self.PATH_COLOR[0] * alpha),
                    int(self.PATH_COLOR[1] * alpha),
                    int(self.PATH_COLOR[2] * alpha)
                )
                pygame.draw.circle(screen, point_color, (px, y), 6)

        # 终点箭头
        if progress > 0.1:
            pygame.draw.circle(screen, self.ARROW_COLOR, (end_x, y), 10)
            # 箭头三角形
            arrow_points = [
                (end_x + 8, y),
                (end_x - 2, y - 8),
                (end_x - 2, y + 8)
            ]
            pygame.draw.polygon(screen, self.ARROW_COLOR, arrow_points)
