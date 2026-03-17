from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


class FasterWhisperSTT:
    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
    ) -> None:
        self.language = language
        self.model = WhisperModel(
            model_size_or_path=model_name,
            device=device,
            compute_type=compute_type,
        )

    def transcribe_wav(self, wav_path: Path) -> str:
        segments, _ = self.model.transcribe(
            str(wav_path),
            language=self.language,
            beam_size=1,
            vad_filter=False,
        )
        return "".join(segment.text for segment in segments).strip()
