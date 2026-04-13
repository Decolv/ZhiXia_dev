"""Faster Whisper 语音识别引擎"""

import logging
from pathlib import Path

from zhixia.asr.base import ASREngine, ASRResult
from zhixia.config.settings import ASRConfig

logger = logging.getLogger(__name__)


class WhisperASREngine(ASREngine):

    def __init__(self, config: ASRConfig) -> None:
        self._config = config
        self._model = None

    @property
    def name(self) -> str:
        return "whisper"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info("加载 Faster Whisper 模型 (%s) ...", self._config.whisper_model)
        self._model = WhisperModel(
            model_size_or_path=self._config.whisper_model,
            device=self._config.whisper_device,
            compute_type=self._config.whisper_compute_type,
        )
        logger.info("Faster Whisper 模型加载完成")

    def transcribe(self, audio_path: Path) -> ASRResult:
        self._ensure_model()
        segments, _ = self._model.transcribe(
            str(audio_path),
            language=self._config.language,
            beam_size=1,
            vad_filter=False,
        )
        text = "".join(segment.text for segment in segments).strip()
        logger.info("Whisper 识别结果: %s", text)
        return ASRResult(text=text, engine_name=self.name)
