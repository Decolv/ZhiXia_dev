"""ZhiXia 语音助手入口：python -m zhixia"""

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from zhixia.asr.funasr_engine import FunASREngine
from zhixia.asr.whisper_engine import WhisperASREngine
from zhixia.audio.player import ALSAAudioPlayer
from zhixia.audio.recorder import AudioRecorder
from zhixia.config.settings import AppSettings
from zhixia.display.null_display import NullDisplay
from zhixia.llm.cloud_engine import CloudLLMEngine
from zhixia.llm.rkllm_engine import RKLLMEngine
from zhixia.llm.rag.null_retriever import NullRAGRetriever
from zhixia.pipeline.orchestrator import VoicePipeline
from zhixia.tts.piper_engine import PiperTTSEngine
from zhixia.utils.logging import setup_logging
from zhixia.utils.network import is_online

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
    """创建LLM引擎，根据网络状态自动选择云端或本地"""
    if config.enable_cloud_fallback:
        online = is_online(use_cache=True)
        if online:
            logger = logging.getLogger(__name__)
            logger.info("网络可用，使用云端LLM引擎")
            print("🌐 使用云端大模型")
            return CloudLLMEngine(config)
        else:
            logger = logging.getLogger(__name__)
            logger.info("网络不可用，回退到本地LLM引擎")
            print("📴 网络离线，使用本地模型")
            return RKLLMEngine(config)
    else:
        return RKLLMEngine(config)


def create_tts_engine(config):
    return PiperTTSEngine(config, _PROJECT_ROOT)


def create_rag_retriever(config):
    if not config.rag.enabled:
        return NullRAGRetriever()
    return NullRAGRetriever()  # 暂时只支持 null，后续可扩展


def create_display(config):
    return NullDisplay()


def create_agent_executor(config, llm_engine):
    """创建 AgentExecutor（如果配置启用了 Agent 模式）。"""
    if not getattr(config, "agent", None) or not config.agent.enabled:
        return None

    from zhixia.agent import (
        AgentExecutor,
        ReActAgent,
        ToolCallingAgent,
        ToolRegistry,
    )

    # 这里可以注册项目需要的工具
    tools = ToolRegistry()
    # 示例：注册天气工具（需要实际实现 WeatherTool）
    # from zhixia.agent.tools.weather import WeatherTool
    # tools.register(WeatherTool())

    if config.agent.engine == "tool_calling":
        agent = ToolCallingAgent(
            llm_engine=llm_engine,
            tools=tools,
            max_new_tokens=config.llm.max_new_tokens,
        )
    else:
        agent = ReActAgent(
            llm_engine=llm_engine,
            tools=tools,
            max_new_tokens=config.llm.max_new_tokens,
        )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=config.agent.max_iterations,
        early_stopping_method=config.agent.early_stopping_method,
    )


def create_wakeword_engine(config, project_root: Path):
    """创建唤醒词引擎。"""
    if not getattr(config, "wakeword", None) or not config.wakeword.enabled:
        return None

    if config.wakeword.engine == "snowboy":
        from zhixia.wakeword.snowboy_engine import SnowboyWakeWordEngine

        engine = SnowboyWakeWordEngine(config.wakeword, project_root)
        if not engine.is_available():
            logger.error("Snowboy 不可用。安装方式:")
            logger.error("  pip install snowboy")
            logger.error("  或: pip install git+https://github.com/seasalt-ai/snowboy.git")
            return None
        return engine

    logger.error("未知的唤醒词引擎: %s", config.wakeword.engine)
    return None


class WakeWordLoop:
    """唤醒词监听 + 语音交互主循环。

    流程：
        启动 → 加载唤醒词模型 → 开始监听
        检测到唤醒词 → 停止监听 → 播放提示音 → 录音 → Pipeline处理 → 回到监听
    """

    def __init__(
        self,
        config: AppSettings,
        wakeword_engine,
        pipeline: VoicePipeline,
        recorder: AudioRecorder,
        player: ALSAAudioPlayer,
    ) -> None:
        self.config = config
        self.wakeword = wakeword_engine
        self.pipeline = pipeline
        self.recorder = recorder
        self.player = player

        self._shutdown = threading.Event()
        self._output_dir = Path(config.audio.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 预合成提示音路径
        self._ding_wav = _PROJECT_ROOT / "assets" / "ding.wav"

    def _play_wake_sound(self) -> None:
        """播放唤醒提示音。"""
        wake_sound = getattr(self.config.wakeword, "wake_sound", "ding")

        if wake_sound == "none":
            return

        if wake_sound == "ding" and self._ding_wav.exists():
            self.player.play(self._ding_wav, blocking=True)
            return

        if wake_sound == "tts":
            try:
                wav = self.pipeline.tts_engine.synthesize_to_bytes("我在")
                if wav:
                    self.player.play_bytes(wav, blocking=True)
                    return
            except Exception:
                logger.exception("TTS 提示音合成失败")

        # 回退：静默（不播放提示音）
        logger.debug("无可用提示音")

    def _on_wake(self, result) -> None:
        """唤醒词检测回调。"""
        print(f"\n🔔 唤醒词检测: {result.keyword_name}")

    def _record_and_process(self) -> None:
        """录音并走 Pipeline 处理。"""
        input_audio = self._output_dir / "recorded_input.wav"
        duration = self.config.wakeword.record_duration

        print(f"\n🎤 请说话 ({duration:.0f} 秒)...")
        try:
            self.recorder.record_to_wav(duration, input_audio)
            print("✅ 录音完成")
        except Exception as exc:
            logger.exception("录音失败")
            print(f"\n❌ 录音失败: {exc}")
            return

        try:
            self.pipeline.process_audio(input_audio)
        except Exception as exc:
            logger.exception("Pipeline 处理失败")
            print(f"\n❌ 处理失败: {exc}")

    def run(self) -> None:
        """主循环。"""
        print("\n" + "=" * 70)
        print("🎙️  ZhiXia 语音助手 - 唤醒词模式")
        print("=" * 70)
        print(f"唤醒词引擎: {self.wakeword.name}")
        print("按 Ctrl+C 退出")
        print("-" * 70)

        if not self.wakeword.load_models():
            print("❌ 唤醒词模型加载失败")
            sys.exit(1)

        while not self._shutdown.is_set():
            print("\n👂 正在监听唤醒词...")
            try:
                self.wakeword.start_listening(
                    on_wake=self._on_wake,
                    interrupt_check=lambda: self._shutdown.is_set(),
                )
            except Exception as exc:
                logger.exception("监听异常")
                print(f"\n❌ 监听异常: {exc}")
                time.sleep(1)
                continue

            if self._shutdown.is_set():
                break

            # 检测到唤醒词，播放提示音
            self._play_wake_sound()

            # 录音 + Pipeline 处理
            self._record_and_process()

            # 短暂停顿后回到监听
            time.sleep(0.5)

        print("\n👋 已退出唤醒词模式")

    def stop(self) -> None:
        """请求停止循环。"""
        print("\n🛑 正在停止...")
        self._shutdown.set()
        self.wakeword.stop_listening()


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
    agent_executor = create_agent_executor(settings, llm)

    # 预热：提前加载模型并执行一次真实推理，消除首次请求冷启动
    if not settings.skip_warmup:
        import time

        print("⏳ 预热模型中...")
        t0 = time.perf_counter()

        # ASR: 加载模型到内存（不跑推理，保留流式扩展性）
        try:
            if hasattr(asr, "_ensure_model"):
                asr._ensure_model()
                print(f"  ASR 模型加载完成 ({time.perf_counter() - t0:.2f}s)")
        except Exception:
            pass

        # LLM: 加载模型 + 一次短推理完成缓存预热
        try:
            llm._ensure_model()
            t1 = time.perf_counter()
            # 极短 prompt 只生成 1~2 个 token，避免长回复耗时
            from zhixia.llm.base import LLMMessage
            list(llm.stream_chat([LLMMessage(role="user", content="只回复一个'好的'")]))
            print(f"  LLM 预热完成 ({time.perf_counter() - t1:.2f}s)")
        except Exception:
            pass

        # TTS: 加载模型 + 一次短合成完成 ONNX session 预热
        try:
            tts._ensure_voice()
            t2 = time.perf_counter()
            tts.synthesize_to_bytes("预热")
            print(f"  TTS 预热完成 ({time.perf_counter() - t2:.2f}s)")
        except Exception:
            pass

        print(f"✅ 全部预热完成 ({time.perf_counter() - t0:.2f}s)")
    else:
        print("⏩ 已跳过预热（skip_warmup=true）")

    # 创建管线
    pipeline = VoicePipeline(
        config=settings,
        asr_engine=asr,
        llm_engine=llm,
        tts_engine=tts,
        audio_player=player,
        rag_retriever=rag,
        display=display,
        agent_executor=agent_executor,
    )

    # 判断运行模式
    wakeword_engine = create_wakeword_engine(settings, _PROJECT_ROOT)

    if wakeword_engine is not None:
        # ===== 唤醒词模式 =====
        recorder = AudioRecorder(sample_rate=settings.asr.record_sample_rate)
        try:
            recorder.ensure_input_device()
        except RuntimeError as exc:
            logger.error("未检测到麦克风: %s", exc)
            print("\n❌ 未检测到可用麦克风输入设备")
            sys.exit(1)

        loop = WakeWordLoop(
            config=settings,
            wakeword_engine=wakeword_engine,
            pipeline=pipeline,
            recorder=recorder,
            player=player,
        )

        def _signal_handler(_signum, _frame):
            loop.stop()

        signal.signal(signal.SIGINT, _signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _signal_handler)

        try:
            loop.run()
        except KeyboardInterrupt:
            loop.stop()
        finally:
            wakeword_engine.shutdown()

    else:
        # ===== 单次运行模式（录音 or 文件）=====
        input_audio: Path | None = None
        if settings.asr.enable_recording:
            # 录音模式
            recorder = AudioRecorder(sample_rate=settings.asr.record_sample_rate)
            try:
                recorder.ensure_input_device()
            except RuntimeError as exc:
                logger.error("未检测到麦克风: %s", exc)
                print("\n❌ 未检测到可用麦克风输入设备")
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
            except Exception as exc:
                logger.exception("录音失败")
                print(f"\n❌ 录音失败: {exc}")
                sys.exit(1)
        else:
            # 文件模式
            input_audio = Path(settings.asr.input_audio) if settings.asr.input_audio else None
            if not input_audio or not input_audio.exists():
                logger.error("输入音频文件不存在: %s", input_audio)
                print(f"\n❌ 输入音频文件不存在: {input_audio}")
                print("请在 localconfig.json 中配置 asr.input_audio，或开启录音模式 (asr.enable_recording=true)")
                sys.exit(1)

        # 处理音频
        try:
            pipeline.process_audio(input_audio)
        except Exception as exc:
            logger.exception("管线处理失败")
            print(f"\n❌ 处理失败: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
