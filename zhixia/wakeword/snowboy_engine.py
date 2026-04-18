"""Snowboy 唤醒词检测引擎

依赖: snowboy (或 seasalt-ai/snowboy 社区 fork)
安装: pip install snowboy
     或 pip install git+https://github.com/seasalt-ai/snowboy.git

模型: .umdl (通用模型) 或 .pmdl (个人训练模型)
      可训练: https://snowboy.kitt.ai
"""

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from zhixia.config.settings import WakeWordConfig
from zhixia.wakeword.base import WakeWordEngine, WakeWordResult

logger = logging.getLogger(__name__)


class SnowboyWakeWordEngine(WakeWordEngine):
    """基于 Snowboy 的唤醒词检测引擎。

    Snowboy 内部使用 PyAudio 管理麦克风，检测到唤醒词后调用回调。
    回调中需要调用 detector.terminate() 来停止监听。

    由于 PyAudio 设备独占特性，Snowboy 监听和 sounddevice 录音不能同时进行。
    本引擎在检测到唤醒词后自动终止 detector，释放麦克风，以便后续录音。
    """

    def __init__(self, config: WakeWordConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._detector = None
        self._should_stop = threading.Event()
        self._is_listening = False
        self._lock = threading.Lock()
        self._model_paths: list[str] = []

    @property
    def name(self) -> str:
        return "snowboy"

    def is_available(self) -> bool:
        try:
            import snowboydecoder

            del snowboydecoder
            return True
        except ImportError:
            logger.warning("snowboy 未安装，唤醒词检测不可用")
            return False

    def _resolve_model_paths(self) -> list[Path]:
        """解析模型路径，支持相对路径和绝对路径。"""
        paths = []
        for model_path_str in self._config.model_paths:
            path = Path(model_path_str)
            if not path.is_absolute():
                path = self._project_root / path
            if path.exists():
                paths.append(path)
            else:
                logger.warning("唤醒词模型不存在: %s", path)
        return paths

    def load_models(self) -> bool:
        """验证模型文件存在性。"""
        model_paths = self._resolve_model_paths()
        if not model_paths:
            logger.error("没有可用的唤醒词模型")
            return False
        self._model_paths = [str(p) for p in model_paths]
        logger.info("Snowboy 模型路径: %s", self._model_paths)
        return True

    def _create_detector(self):
        """创建 Snowboy HotwordDetector 实例。"""
        from snowboy import snowboydecoder

        sensitivity = self._config.sensitivity
        if isinstance(sensitivity, (int, float)):
            sensitivity = [str(sensitivity)] * len(self._model_paths)
        else:
            sensitivity = [str(s) for s in sensitivity]

        detector = snowboydecoder.HotwordDetector(
            self._model_paths,
            sensitivity=sensitivity,
            audio_gain=self._config.audio_gain,
            apply_frontend=self._config.apply_frontend,
        )
        return detector

    def start_listening(
        self,
        on_wake: Callable[[WakeWordResult], None],
        interrupt_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """开始监听唤醒词。

        阻塞调用，直到检测到唤醒词或外部中断。
        Snowboy 的 start() 是阻塞的，在回调中调用 terminate() 来结束。
        """
        if not self._model_paths:
            raise RuntimeError("模型未加载，请先调用 load_models()")

        with self._lock:
            if self._is_listening:
                logger.warning("已经在监听中")
                return
            self._is_listening = True
            self._should_stop.clear()

        try:
            self._detector = self._create_detector()
        except Exception as exc:
            logger.exception("创建 Snowboy 检测器失败")
            self._is_listening = False
            raise RuntimeError(f"Snowboy 检测器初始化失败: {exc}") from exc

        logger.info("开始监听唤醒词...")

        def _wrapped_callback():
            """Snowboy 回调函数。"""
            logger.info("检测到唤醒词!")

            result = WakeWordResult(
                detected=True,
                keyword_index=0,
                keyword_name=(
                    self._config.keyword_names[0]
                    if self._config.keyword_names
                    else "snowboy"
                ),
            )

            try:
                on_wake(result)
            except Exception:
                logger.exception("唤醒回调异常")

            # 必须在回调中终止 detector，否则 start() 不会返回
            self._should_stop.set()
            if self._detector:
                self._detector.terminate()

        def _interrupt_check() -> bool:
            if self._should_stop.is_set():
                return True
            if interrupt_check is not None:
                return interrupt_check()
            return False

        try:
            self._detector.start(
                detected_callback=_wrapped_callback,
                interrupt_check=_interrupt_check,
                sleep_time=0.03,
            )
        finally:
            with self._lock:
                self._is_listening = False
                self._detector = None
            logger.debug("监听已停止")

    def stop_listening(self) -> None:
        """停止当前监听。"""
        logger.info("请求停止监听")
        self._should_stop.set()
        if self._detector:
            self._detector.terminate()
            self._detector = None

    def shutdown(self) -> None:
        """释放资源。"""
        self.stop_listening()
