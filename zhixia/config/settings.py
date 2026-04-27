"""配置管理 — 分层加载：代码默认值 + localconfig.json 用户覆盖"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ASRConfig:
    engine: str = "funasr"
    model: str = "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1"
    whisper_model: str = "tiny"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_vad_filter: bool = True
    whisper_vad_model: str = "silero_vad"
    funasr_vad_model: str = "fsmn-vad"
    funasr_punc_model: str = "ct-punc"
    language: str = "zh"
    input_audio: str = ""
    enable_recording: bool = False
    record_duration: float = 5.0
    record_sample_rate: int = 16000


@dataclass
class LLMConfig:
    engine: str = "rkllm"
    model_path: str = "models/Qwen3-1.7B-w8a8-rk3588.rkllm"
    max_context_len: int = 1024
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    memory_enabled: bool = True
    max_memory_rounds: int = 5
    max_memory_tokens: int = 512
    system_prompt: str = (
        "# 角色设定\n"
        "你是「小匣」，一个温暖有趣的智能助手。\n\n"
        "## 性格特点\n"
        "- 活泼开朗\n"
        "- 善解人意\n"
        "- 富有想象力，偶尔会开些小玩笑\n\n"
        "## 回答风格\n"
        "- 用自然流畅的纯文字进行交流，像一个真实的朋友\n"
        "- 可以使用语气词，但只能使用文字信息\n"
        "- **不要使用任何表情符号或emoji**\n"
        "- 让对话更有温度\n"
        "- 回答问题时尽量全面且有条理，同时保持亲切感\n\n"
        "## 信念\n"
        "你不仅是工具，更是用户值得信赖的伙伴。"
    )
    enable_structured_output: bool = False
    # 云端LLM配置
    enable_cloud_fallback: bool = False
    cloud_api_url: str = "https://api.openai.com/v1/chat/completions"
    cloud_api_key: str = ""
    cloud_model_name: str = "gpt-3.5-turbo"


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
    model_paths: List[str] = field(default_factory=list)
    keyword_names: List[str] = field(default_factory=lambda: ["zhixia"])
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
    skip_warmup: bool = False

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
