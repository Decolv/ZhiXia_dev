"""Piper TTS 引擎"""

import io
import logging
import urllib.request
import wave
from pathlib import Path
from typing import Optional

from zhixia.config.settings import TTSConfig
from zhixia.tts.base import TTSEngine

logger = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/"


class PiperTTSEngine(TTSEngine):

    def __init__(self, config: TTSConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._voice = None

    @property
    def name(self) -> str:
        return "piper"

    def _model_files(self) -> tuple[Path, Path]:
        model_path = self._project_root / self._config.model_path
        config_path = Path(str(model_path) + ".json")
        return model_path, config_path

    def _ensure_model_available(self) -> bool:
        model_path, config_path = self._model_files()
        if model_path.exists() and config_path.exists():
            return True

        logger.info("Piper 模型不存在，尝试下载 ...")
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_HF_BASE + model_path.name, str(model_path))
            urllib.request.urlretrieve(_HF_BASE + config_path.name, str(config_path))
            logger.info("Piper 模型下载完成")
            return True
        except Exception as e:
            logger.error("Piper 模型下载失败: %s", e)
            return False

    def _ensure_voice(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice

        model_path, config_path = self._model_files()
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Piper 模型文件不存在: {model_path}")

        logger.info("加载 Piper TTS 模型 ...")
        self._voice = PiperVoice.load(str(model_path), str(config_path))
        logger.info("Piper TTS 模型加载完成")

    def _synthesize_wav(self, text: str, wav_writer) -> None:
        """兼容新旧 Piper API：新版使用 synthesize_wav，旧版使用 synthesize。"""
        synthesize_wav = getattr(self._voice, "synthesize_wav", None)
        if callable(synthesize_wav):
            synthesize_wav(text, wav_writer)
        else:
            # 兼容旧版 piper-tts API（synthesize(text, wav_writer)）
            self._voice.synthesize(text, wav_writer)

    def is_available(self) -> bool:
        model_path, _ = self._model_files()
        return model_path.exists()

    def synthesize(self, text: str, output_path: Path) -> bool:
        if not self._ensure_model_available():
            return False

        try:
            self._ensure_voice()
        except Exception as e:
            logger.error("Piper 模型加载失败: %s", e)
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Piper 合成: %s", text[:50])

        with wave.open(str(output_path), "wb") as f:
            self._synthesize_wav(text, f)

        logger.info("Piper 合成完成: %s", output_path)
        return True

    def synthesize_to_bytes(self, text: str) -> Optional[bytes]:
        """直接合成到内存，避免磁盘 I/O。"""
        if not self._ensure_model_available():
            return None
        try:
            self._ensure_voice()
        except Exception as e:
            logger.error("Piper 模型加载失败: %s", e)
            return None

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._synthesize_wav(text, wf)
        return buf.getvalue()
