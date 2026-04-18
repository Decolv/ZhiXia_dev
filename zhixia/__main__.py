"""ZhiXia 语音助手入口：python -m zhixia"""

import logging
import os
import sys
from pathlib import Path

from zhixia.asr.funasr_engine import FunASREngine
from zhixia.asr.whisper_engine import WhisperASREngine
from zhixia.audio.player import ALSAAudioPlayer
from zhixia.config.settings import AppSettings
from zhixia.display.null_display import NullDisplay
from zhixia.llm.rkllm_engine import RKLLMEngine
from zhixia.llm.rag.null_retriever import NullRAGRetriever
from zhixia.pipeline.orchestrator import VoicePipeline
from zhixia.tts.piper_engine import PiperTTSEngine
from zhixia.utils.logging import setup_logging

# 添加项目根目录到 sys.path（支持从 IDE 运行）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 默认将 ModelScope 缓存固定到项目目录，和文档约定保持一致。
_DEFAULT_MODELSCOPE_CACHE = _PROJECT_ROOT / ".cache" / "modelscope"
os.environ.setdefault("MODELSCOPE_CACHE", str(_DEFAULT_MODELSCOPE_CACHE))
_DEFAULT_MODELSCOPE_CACHE.mkdir(parents=True, exist_ok=True)


def create_asr_engine(config):
    if config.asr.engine == "whisper":
        return WhisperASREngine(config.asr)
    else:
        return FunASREngine(config.asr, _PROJECT_ROOT)


def create_llm_engine(config):
    return RKLLMEngine(config)


def create_tts_engine(config):
    return PiperTTSEngine(config, _PROJECT_ROOT)


def create_rag_retriever(config):
    if not config.rag.enabled:
        return NullRAGRetriever()
    return NullRAGRetriever()  # 暂时只支持 null，后续可扩展


def create_display(config):
    return NullDisplay()


def main():
    logger = logging.getLogger(__name__)

    # 加载配置（支持通过环境变量覆盖，便于 PC/Linux 分离测试）
    config_override = os.environ.get("ZHIXIA_CONFIG", "").strip()
    config_path = None
    if config_override:
        config_path = Path(config_override)
        if not config_path.is_absolute():
            config_path = _PROJECT_ROOT / config_path
        print(f"🧩 使用配置文件: {config_path}")

    settings = AppSettings.load(config_path=config_path)

    # 设置日志
    setup_logging(settings.log_level)

    # 检查内存（仅在 RK3588 上有意义）
    device_config = getattr(settings, "device", None)
    if isinstance(device_config, dict) and device_config.get("memory_optimization"):
        from zhixia.utils.memory import check_memory
        mem_available = check_memory()
        if mem_available and mem_available < 2.0:
            logger.warning(f"可用内存不足: {mem_available:.2f} GB")

    # 创建引擎
    asr = create_asr_engine(settings)
    llm = create_llm_engine(settings.llm)
    tts = create_tts_engine(settings.tts)
    player = ALSAAudioPlayer()
    rag = create_rag_retriever(settings)
    display = create_display(settings.display)

    # 预热：提前加载模型，消除首次请求冷启动
    import time
    print("⏳ 预热模型中...")
    t0 = time.perf_counter()
    try:
        llm._ensure_model()
    except Exception:
        pass
    try:
        tts._ensure_voice()
    except Exception:
        pass
    print(f"✅ 模型预热完成 ({time.perf_counter()-t0:.2f}s)")

    # 创建管线
    pipeline = VoicePipeline(
        config=settings,
        asr_engine=asr,
        llm_engine=llm,
        tts_engine=tts,
        audio_player=player,
        rag_retriever=rag,
        display=display,
    )

    # 获取输入音频路径（录音模式 or 文件模式）
    input_audio: Path | None = None
    if settings.asr.enable_recording:
        # 录音模式
        from zhixia.audio.recorder import AudioRecorder
        recorder = AudioRecorder(sample_rate=settings.asr.record_sample_rate)
        try:
            recorder.ensure_input_device()
        except RuntimeError as e:
            logger.error("未检测到麦克风: %s", e)
            print(f"\n❌ 未检测到可用麦克风输入设备")
            print("请检查音频设备连接，或在配置中关闭录音模式 (asr.enable_recording=false)")
            sys.exit(1)

        output_dir = Path(settings.audio.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_audio = output_dir / "recorded_input.wav"

        duration = settings.asr.record_duration
        print(f"\n🎤 录音模式: 请在 {duration:.0f} 秒内说话...")
        try:
            recorder.record_to_wav(duration, input_audio)
            print(f"✅ 录音完成: {input_audio}")
        except Exception as e:
            logger.exception("录音失败")
            print(f"\n❌ 录音失败: {e}")
            sys.exit(1)
    else:
        # 文件模式
        input_audio = Path(settings.asr.input_audio) if settings.asr.input_audio else None
        if not input_audio or not input_audio.exists():
            logger.error(f"输入音频文件不存在: {input_audio}")
            print(f"\n❌ 输入音频文件不存在: {input_audio}")
            print("请在 localconfig.json 中配置 asr.input_audio，或开启录音模式 (asr.enable_recording=true)")
            sys.exit(1)

    # 处理音频
    try:
        pipeline.process_audio(input_audio)
    except Exception as e:
        logger.exception("管线处理失败")
        print(f"\n❌ 处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
