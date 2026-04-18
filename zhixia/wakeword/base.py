"""唤醒词检测引擎抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class WakeWordResult:
    """唤醒检测结果"""

    detected: bool
    keyword_index: int = 0
    keyword_name: str = ""
    confidence: float = 0.0


class WakeWordEngine(ABC):
    """唤醒词检测引擎抽象基类。

    生命周期：
        1. 创建实例（配置参数）
        2. load_models() — 加载模型文件
        3. start_listening() — 开始监听（阻塞直到检测到或中断）
        4. stop_listening() — 停止监听
        5. shutdown() — 释放资源
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称。"""

    @abstractmethod
    def load_models(self) -> bool:
        """加载唤醒词模型。返回是否成功。"""

    @abstractmethod
    def start_listening(
        self,
        on_wake: Callable[[WakeWordResult], None],
        interrupt_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """开始监听唤醒词。

        这是阻塞调用，直到检测到唤醒词或外部中断。

        Args:
            on_wake: 检测到唤醒词时的回调函数
            interrupt_check: 可选的轮询函数，返回 True 时停止监听
        """

    @abstractmethod
    def stop_listening(self) -> None:
        """停止当前监听。"""

    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用（依赖库是否安装）。"""

    def shutdown(self) -> None:
        """释放资源，默认空操作。"""
