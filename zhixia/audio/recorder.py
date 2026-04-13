"""音频录制"""

import logging
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioRecorder:

    def __init__(self, sample_rate: int = 16000, channels: int = 1, dtype: str = "int16") -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._sd = None

    def _ensure_sd(self):
        if self._sd is None:
            import sounddevice as sd
            self._sd = sd

    def ensure_input_device(self) -> None:
        self._ensure_sd()
        try:
            self._sd.query_devices(kind="input")
        except Exception as exc:
            raise RuntimeError("未检测到可用麦克风输入设备。") from exc

    def record_to_wav(self, seconds: float, output_path: Path) -> Path:
        if seconds <= 0:
            raise ValueError("录音时长必须大于 0 秒。")

        self._ensure_sd()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = int(seconds * self.sample_rate)

        logger.info("录音 %.1f 秒 ...", seconds)
        audio = self._sd.rec(
            frame_count,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
        )
        self._sd.wait()

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())

        logger.info("录音已保存: %s", output_path)
        return output_path
