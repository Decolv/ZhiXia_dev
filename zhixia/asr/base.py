"""ASR 引擎抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ASRResult:
    text: str
    confidence: float = 1.0
    engine_name: str = ""


class ASREngine(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def transcribe(self, audio_path: Path) -> ASRResult:
        """将音频文件转写为文本。空 text 表示识别失败。"""
