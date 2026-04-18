"""配置管理 — 分层加载：代码默认值 + localconfig.json 用户覆盖"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ASRConfig:
    engine: str = "funasr"
    model: str = "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1"
    whisper_model: str = "tiny"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    language: str = "zh"
    input_audio: str = ""
    enable_recording: bool = False
    record_duration: float = 5.0
    record_sample_rate: int = 16000


@dataclass
class LLMConfig:
    engine: str = "rkllm"
    model_path: str = "models/Qwen3-1.7B-w8a8-rk3588.rkllm"
    max_context_len: int = 512
    max_new_tokens: int = 32
    temperature: float = 0.8
    top_p: float = 0.95
    system_prompt: str = "你是AI助手，用一句话简短回答。"
    enable_structured_output: bool = False


@dataclass
class RAGConfig:
    enabled: bool = False
    engine: str = "null"
    top_k: int = 3


@dataclass
class TTSConfig:
    engine: str = "piper"
    model_path: str = "models/piper/zh_CN-huayan-medium.onnx"
    speed: float = 1.0


@dataclass
class AudioConfig:
    output_dir: str = "output"
    output_format: str = "wav"
    sample_rate: int = 22050


@dataclass
class DisplayConfig:
    enabled: bool = False
    engine: str = "null"


@dataclass
class WakeWordConfig:
    enabled: bool = False
    engine: str = "snowboy"
    model_paths: list[str] = field(default_factory=list)
    keyword_names: list[str] = field(default_factory=lambda: ["zhixia"])
    sensitivity: float = 0.5
    audio_gain: float = 1.0
    apply_frontend: bool = False
    record_duration: float = 5.0
    wake_sound: str = "ding"  # "ding" | "tts" | "none"


@dataclass
class AppSettings:
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    log_level: str = "INFO"

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    config_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "localconfig")

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppSettings":
        """加载配置：代码默认值 + localconfig.json 覆盖"""
        settings = cls()

        if config_path is None:
            config_path = settings.config_dir / "localconfig.json"

        if not config_path.exists():
            return settings

        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)

        _deep_merge(settings, user_config)
        return settings


def _deep_merge(obj, data: dict) -> None:
    """将 data 中的值递归合并到 obj（dataclass 实例）中"""
    if not hasattr(obj, "__dataclass_fields__"):
        return

    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        field_type = obj.__dataclass_fields__[key].type
        target = getattr(obj, key)

        if isinstance(value, dict) and hasattr(target, "__dataclass_fields__"):
            _deep_merge(target, value)
        else:
            setattr(obj, key, value)
