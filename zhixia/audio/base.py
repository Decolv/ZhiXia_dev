"""音频播放抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class AudioPlayer(ABC):

    @abstractmethod
    def play(self, audio_path: Path, blocking: bool = True) -> bool:
        """播放音频文件。"""

    def play_bytes(self, wav_bytes: bytes, blocking: bool = True) -> bool:
        """从内存 bytes 播放 WAV 音频。默认回退到临时文件实现。"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = Path(f.name)
        try:
            return self.play(tmp, blocking=blocking)
        finally:
            tmp.unlink(missing_ok=True)

    @abstractmethod
    def is_available(self) -> bool:
        """检查播放器是否可用。"""
