from __future__ import annotations

import wave
from pathlib import Path

import sounddevice as sd


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, dtype: str = "int16") -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

    def ensure_input_device(self) -> None:
        try:
            sd.query_devices(kind="input")
        except Exception as exc:
            raise RuntimeError("未检测到可用麦克风输入设备。") from exc

    def record_to_wav(self, seconds: float, output_path: Path) -> Path:
        if seconds <= 0:
            raise ValueError("录音时长必须大于 0 秒。")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_count = int(seconds * self.sample_rate)

        audio = sd.rec(
            frame_count,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
        )
        sd.wait()

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())

        return output_path
