"""FunASR 语音识别引擎（INT8 量化）"""

import logging
from pathlib import Path

from zhixia.asr.base import ASREngine, ASRResult
from zhixia.config.settings import ASRConfig

logger = logging.getLogger(__name__)


class FunASREngine(ASREngine):

    def __init__(self, config: ASRConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._model = None

    @property
    def name(self) -> str:
        return "funasr"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel

        logger.info("加载 FunASR 模型 (INT8 量化) ...")
        try:
            self._model = AutoModel(
                model=self._config.model,
                vad_model=self._config.funasr_vad_model,
                punc_model=self._config.funasr_punc_model,
                disable_update=True,
                hub="ms",
                quantize=True,
                device="cpu",
            )
        except Exception:
            logger.warning("INT8 量化模型加载失败，回退到标准版")
            self._model = AutoModel(
                model=self._config.model,
                vad_model=self._config.funasr_vad_model,
                punc_model=self._config.funasr_punc_model,
                disable_update=True,
                hub="ms",
            )
        logger.info("FunASR 模型加载完成")

    def transcribe(self, audio_path: Path) -> ASRResult:
        self._ensure_model()
        result = self._model.generate(input=str(audio_path))
        text = ""
        confidence = 1.0
        if result and len(result) > 0:
            text = result[0].get("text", "")
            confidence = result[0].get("confidence", 1.0)
        text = text.strip()
        logger.info("ASR 识别结果: %s", text)
        return ASRResult(text=text, engine_name=self.name, confidence=confidence)
