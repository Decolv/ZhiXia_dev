"""Live2D 眼睛渲染器 — 使用 Pygame 渲染动态眼睛动画

在独立窗口中展示可爱的眼睛，支持：
- 6种情绪状态切换
- 自动眨眼动画
- 瞳孔跟随/转动动画
- 透明度/缩放控制
"""

import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_MODEL_CONFIG = Path(__file__).parent.parent.parent / "assets" / "live2d" / "eyes" / "model.json"


# ---------------------------------------------------------------------------
# 眼睛状态配置
# ---------------------------------------------------------------------------

@dataclass
class EyeStateConfig:
    """单个眼睛状态的配置。"""
    eye_open: float = 1.0        # 眼睛开合度 (0.0=闭眼, 1.0=正常, 1.5=睁大)
    blink_interval_ms: int = 3000  # 眨眼间隔（毫秒）
    pupil_movement: str = "slow"   # 瞳孔移动模式: slow/fast/vertical/down/none
    eye_curve: bool = False        # 是否使用弯月形（开心表情）


# 预设状态配置
PRESET_STATES: Dict[str, EyeStateConfig] = {
    "neutral": EyeStateConfig(eye_open=1.0, blink_interval_ms=3000, pupil_movement="slow"),
    "thinking": EyeStateConfig(eye_open=0.7, blink_interval_ms=4000, pupil_movement="vertical"),
    "happy": EyeStateConfig(eye_open=1.2, blink_interval_ms=5000, pupil_movement="slow", eye_curve=True),
    "working": EyeStateConfig(eye_open=1.0, blink_interval_ms=2500, pupil_movement="fast"),
    "sad": EyeStateConfig(eye_open=0.6, blink_interval_ms=5000, pupil_movement="down"),
    "surprised": EyeStateConfig(eye_open=1.5, blink_interval_ms=1500, pupil_movement="fast"),
}


# ---------------------------------------------------------------------------
# Live2dEyeRenderer — 眼睛渲染器
# ---------------------------------------------------------------------------

class Live2dEyeRenderer:
    """Live2D 风格眼睛渲染器。

    使用 Pygame 在独立窗口中渲染动态眼睛，支持多种情绪状态。
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        window_width: int = 300,
        window_height: int = 200,
    ) -> None:
        self.config_path = config_path or DEFAULT_MODEL_CONFIG
        self.window_width = window_width
        self.window_height = window_height

        # 加载配置
        self.config = self._load_config()

        # 窗口参数
        self.bg_color = tuple(self.config.get("bg_color", [20, 20, 30]))
        self.eye_color = tuple(self.config.get("eye_color", [100, 150, 255]))
        self.pupil_color = tuple(self.config.get("pupil_color", [30, 30, 50]))
        self.highlight_color = tuple(self.config.get("highlight_color", [255, 255, 255]))
        self.eye_size = self.config.get("eye_size", 40)
        self.eye_spacing = self.config.get("eye_spacing", 80)

        # 状态
        self.current_state = "neutral"
        self.target_open = 1.0
        self.current_open = 1.0
        self.is_blinking = False
        self.blink_progress = 0.0

        # 瞳孔位置
        self.pupil_x = 0.0
        self.pupil_y = 0.0
        self.pupil_target_x = 0.0
        self.pupil_target_y = 0.0

        # 计时器
        self.last_blink_time = time.monotonic()
        self.blink_duration = self.config.get("blink_duration_ms", 120) / 1000.0

        # 线程控制
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Pygame 初始化延迟到 start()

    def _load_config(self) -> Dict:
        """加载模型配置文件。"""
        try:
            if self.config_path and self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("config", {})
        except Exception as exc:
            logger.warning("加载眼睛配置失败: %s，使用默认值", exc)
        return {}

    def set_state(self, state: str) -> None:
        """设置眼睛状态。"""
        with self._lock:
            if state in PRESET_STATES:
                self.current_state = state
                cfg = PRESET_STATES[state]
                self.target_open = cfg.eye_open
            else:
                logger.warning("未知眼睛状态: %s", state)

    def force_blink(self) -> None:
        """强制眨眼。"""
        with self._lock:
            if not self.is_blinking:
                self.is_blinking = True
                self.blink_progress = 0.0

    def set_pupil_position(self, x: float, y: float) -> None:
        """设置瞳孔目标位置 (-1.0 ~ 1.0)。"""
        with self._lock:
            self.pupil_target_x = max(-1.0, min(1.0, x))
            self.pupil_target_y = max(-1.0, min(1.0, y))

    def start(self) -> None:
        """启动渲染线程。"""
        if self._running:
            return

        # 延迟导入 pygame（避免无 GUI 环境报错）
        try:
            import pygame
        except ImportError:
            logger.error("pygame 未安装，眼睛渲染器不可用。安装命令: pip install pygame")
            return

        os.environ["SDL_VIDEO_WINDOW_POS"] = "center"
        pygame.init()

        # 创建无边框透明窗口
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.NOFRAME | pygame.RESIZABLE
        )
        pygame.display.set_caption("知匣 - 小匣")
        pygame.display.set_icon(self._create_icon(pygame))

        # 设置窗口透明度
        try:
            pygame.display.get_wm_info()
        except Exception:
            pass

        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True, name="Live2dEyeRenderer")
        self._thread.start()
        logger.info("Live2D 眼睛渲染器已启动")

    def stop(self) -> None:
        """停止渲染线程。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
        logger.info("Live2D 眼睛渲染器已停止")

    def _create_icon(self, pygame) -> "pygame.Surface":
        """创建窗口图标（简单圆形眼睛）。"""
        size = 32
        icon = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        pygame.draw.circle(icon, (*self.eye_color, 255), (center, center), size // 2 - 2)
        pygame.draw.circle(icon, (*self.pupil_color, 255), (center, center), size // 4)
        pygame.draw.circle(icon, (*self.highlight_color, 200), (center - 4, center - 4), 4)
        return icon

    def _render_loop(self) -> None:
        """渲染主循环。"""
        import pygame
        clock = pygame.time.Clock()
        font = None

        while self._running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break

            # 更新动画
            self._update_animation()

            # 渲染
            self.screen.fill(self.bg_color)
            self._draw_eyes(self.screen)

            pygame.display.flip()
            clock.tick(60)  # 60 FPS

    def _update_animation(self) -> None:
        """更新动画状态。"""
        now = time.monotonic()

        with self._lock:
            # 平滑过渡到目标开合度
            target = self.target_open
            if self.is_blinking:
                # 眨眼动画
                self.blink_progress += 1.0 / 60.0 / self.blink_duration
                if self.blink_progress >= 1.0:
                    self.is_blinking = False
                    self.blink_progress = 0.0
                    self.last_blink_time = now
                else:
                    # 使用正弦曲线模拟眨眼
                    blink_factor = math.sin(self.blink_progress * math.pi)
                    target = max(0.05, self.target_open * (1.0 - blink_factor))

            # 平滑插值
            self.current_open += (target - self.current_open) * 0.15

            # 瞳孔移动
            cfg = PRESET_STATES.get(self.current_state, PRESET_STATES["neutral"])
            self._update_pupil(cfg, now)

    def _update_pupil(self, cfg: EyeStateConfig, now: float) -> None:
        """更新瞳孔位置。"""
        movement = cfg.pupil_movement
        speed = 0.02  # 基础速度

        if movement == "slow":
            self.pupil_target_x = math.sin(now * 0.5) * 0.3
            self.pupil_target_y = math.cos(now * 0.3) * 0.2
            speed = 0.03
        elif movement == "fast":
            self.pupil_target_x = math.sin(now * 2.0) * 0.5
            self.pupil_target_y = math.cos(now * 1.5) * 0.4
            speed = 0.08
        elif movement == "vertical":
            self.pupil_target_y = math.sin(now * 0.8) * 0.4
            speed = 0.04
        elif movement == "down":
            self.pupil_target_y = 0.3
            self.pupil_target_x = math.sin(now * 0.3) * 0.1
            speed = 0.02
        elif movement == "none":
            self.pupil_target_x = 0.0
            self.pupil_target_y = 0.0

        # 平滑插值到目标位置
        self.pupil_x += (self.pupil_target_x - self.pupil_x) * speed
        self.pupil_y += (self.pupil_target_y - self.pupil_y) * speed

    def _draw_eyes(self, screen) -> None:
        """绘制眼睛。"""
        import pygame

        center_x = self.window_width // 2
        center_y = self.window_height // 2
        half_spacing = self.eye_spacing // 2

        cfg = PRESET_STATES.get(self.current_state, PRESET_STATES["neutral"])

        for side in [-1, 1]:  # 左眼和右眼
            eye_x = center_x + side * half_spacing
            eye_y = center_y

            # 眼睛高度
            eye_height = self.eye_size * self.current_open
            eye_width = self.eye_size * 0.8

            # 瞳孔偏移
            pupil_offset_x = self.pupil_x * self.eye_size * 0.3
            pupil_offset_y = self.pupil_y * eye_height * 0.3

            if cfg.eye_curve and self.current_open > 1.0:
                # 开心时绘制弯月形眼睛
                self._draw_curved_eye(screen, eye_x, eye_y, eye_width, eye_height, side)
            else:
                # 正常眼睛
                self._draw_normal_eye(screen, eye_x, eye_y, eye_width, eye_height, pupil_offset_x, pupil_offset_y)

    def _draw_normal_eye(self, screen, x: int, y: int, w: float, h: float, px: float, py: float) -> None:
        """绘制正常形状眼睛。"""
        import pygame

        # 眼白（椭圆）
        eye_rect = pygame.Rect(x - w / 2, y - h / 2, w, h)
        pygame.draw.ellipse(screen, (240, 240, 250), eye_rect)

        # 虹膜（圆形）
        iris_radius = min(w, h) * 0.45
        iris_center = (int(x + px), int(y + py))
        pygame.draw.circle(screen, self.eye_color, iris_center, int(iris_radius))

        # 瞳孔
        pupil_radius = iris_radius * 0.5
        pygame.draw.circle(screen, self.pupil_color, iris_center, int(pupil_radius))

        # 高光
        highlight_pos1 = (iris_center[0] - int(iris_radius * 0.3), iris_center[1] - int(iris_radius * 0.3))
        highlight_pos2 = (iris_center[0] + int(iris_radius * 0.1), iris_center[1] + int(iris_radius * 0.2))
        pygame.draw.circle(screen, self.highlight_color, highlight_pos1, int(iris_radius * 0.2))
        pygame.draw.circle(screen, (200, 200, 255, 150), highlight_pos2, int(iris_radius * 0.1))

        # 眼线
        pygame.draw.ellipse(screen, (50, 50, 70), eye_rect, 2)

    def _draw_curved_eye(self, screen, x: int, y: int, w: float, h: float, side: int) -> None:
        """绘制弯月形眼睛（开心表情）。"""
        import pygame

        # 上弧线
        start_angle = math.pi * 1.1 if side > 0 else math.pi * 1.9
        end_angle = math.pi * 1.9 if side > 0 else math.pi * 1.1

        pygame.draw.arc(screen, self.pupil_color,
                       pygame.Rect(x - w / 2, y - h / 3, w, h * 0.6),
                       start_angle, end_angle, 3)

        # 下弧线
        pygame.draw.arc(screen, self.pupil_color,
                       pygame.Rect(x - w / 2, y - h / 6, w, h * 0.5),
                       -start_angle + math.pi, -end_angle + math.pi, 2)
