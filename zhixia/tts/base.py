"""TTS 引擎抽象基类"""

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class TTSEngine(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> bool:
        """合成语音到文件。返回是否成功。"""

    def synthesize_to_bytes(self, text: str) -> Optional[bytes]:
        """合成语音到内存 bytes（WAV 格式）。默认回退到文件实现。子类应覆盖以减少 I/O。"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            if not self.synthesize(text, tmp):
                return None
            return tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)

    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用。"""
