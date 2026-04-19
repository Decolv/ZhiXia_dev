"""公共测试函数库。

被 tests/quick_test.py 和 tests/test_pipeline_stages.ipynb 共用。
每个测试函数返回 (success, result, duration_ms, error)。
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# 将项目根目录加入 sys.path（支持从 tests/ 目录运行）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 默认 ModelScope 缓存
os.environ.setdefault("MODELSCOPE_CACHE", str(_PROJECT_ROOT / ".cache" / "modelscope"))

from zhixia.config.settings import AppSettings
from zhixia.llm.base import LLMMessage
from zhixia.llm.output_parser import StructuredOutput, parse_llm_output

logger = logging.getLogger(__name__)

TestResult = tuple[bool, Any, float, Optional[str]]


def test_config(config_path: Optional[Path] = None) -> TestResult:
    """测试配置加载。"""
    t0 = time.perf_counter()
    try:
        config = AppSettings.load(config_path=config_path)
        duration = (time.perf_counter() - t0) * 1000
        return True, config, duration, None
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        return False, None, duration, str(exc)


def test_asr(config: AppSettings, audio_path: Optional[Path] = None) -> TestResult:
    """测试 ASR：加载模型 + 转写音频。"""
    from zhixia.asr.funasr_engine import FunASREngine

    t0 = time.perf_counter()
    try:
        asr = FunASREngine(config.asr, config.project_root)
        asr._ensure_model()

        test_audio = audio_path
        if test_audio is None:
            test_audio = Path(config.asr.input_audio) if config.asr.input_audio else None
        if test_audio is None or not test_audio.exists():
            duration = (time.perf_counter() - t0) * 1000
            return True, f"模型加载成功 (无音频可转写: {test_audio})", duration, None

        result = asr.transcribe(test_audio)
        duration = (time.perf_counter() - t0) * 1000
        return True, result.text, duration, None
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        return False, None, duration, str(exc)


def test_llm(config: AppSettings) -> TestResult:
    """测试 LLM：加载模型 + 短流式推理（与生产路径一致）。"""
    from zhixia.llm.rkllm_engine import RKLLMEngine

    t0 = time.perf_counter()
    try:
        llm = RKLLMEngine(config.llm)
        llm._ensure_model()

        messages = [LLMMessage(role="user", content="只回复一个'好的'")]
        response = "".join(llm.stream_chat(messages, max_new_tokens=8))
        duration = (time.perf_counter() - t0) * 1000
        return True, response, duration, None
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        return False, None, duration, str(exc)


def test_tts(config: AppSettings) -> TestResult:
    """测试 TTS：加载模型 + 短合成。"""
    from zhixia.tts.piper_engine import PiperTTSEngine

    t0 = time.perf_counter()
    try:
        tts = PiperTTSEngine(config.tts, config.project_root)
        tts._ensure_voice()

        wav = tts.synthesize_to_bytes("你好，测试")
        if wav is None:
            return False, None, (time.perf_counter() - t0) * 1000, "合成返回 None"

        duration = (time.perf_counter() - t0) * 1000
        return True, f"WAV bytes: {len(wav)} bytes", duration, None
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        return False, None, duration, str(exc)


def test_pipeline(config: AppSettings, audio_path: Optional[Path] = None) -> TestResult:
    """测试完整 Pipeline：ASR → LLM → TTS → Play。"""
    from zhixia.asr.funasr_engine import FunASREngine
    from zhixia.audio.player import ALSAAudioPlayer
    from zhixia.display.null_display import NullDisplay
    from zhixia.llm.rag.null_retriever import NullRAGRetriever
    from zhixia.llm.rkllm_engine import RKLLMEngine
    from zhixia.pipeline.orchestrator import VoicePipeline
    from zhixia.tts.piper_engine import PiperTTSEngine

    t0 = time.perf_counter()
    try:
        # 确定测试音频
        test_audio = audio_path
        if test_audio is None:
            test_audio = Path(config.asr.input_audio) if config.asr.input_audio else None
        if test_audio is None or not test_audio.exists():
            return False, None, (time.perf_counter() - t0) * 1000, f"测试音频不存在: {test_audio}"

        # 创建各引擎（跳过预热，由测试本身完成）
        asr = FunASREngine(config.asr, config.project_root)
        llm = RKLLMEngine(config.llm)
        tts = PiperTTSEngine(config.tts, config.project_root)
        player = ALSAAudioPlayer()
        rag = NullRAGRetriever()
        display = NullDisplay()

        pipeline = VoicePipeline(
            config=config,
            asr_engine=asr,
            llm_engine=llm,
            tts_engine=tts,
            audio_player=player,
            rag_retriever=rag,
            display=display,
        )

        pipeline.process_audio(test_audio)
        duration = (time.perf_counter() - t0) * 1000
        return True, "Pipeline 执行完成", duration, None
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        return False, None, duration, str(exc)


def test_display() -> TestResult:
    """测试 Display 接口。"""
    from zhixia.display.base import DisplayPayload
    from zhixia.display.null_display import NullDisplay

    t0 = time.perf_counter()
    try:
        display = NullDisplay()
        payload = DisplayPayload(
            text="测试显示",
            emotion="happy",
            is_thinking=False,
            metadata={"test": True},
        )
        display.show(payload)
        display.update_thinking(True)
        display.clear()
        duration = (time.perf_counter() - t0) * 1000
        return True, "Display 接口正常", duration, None
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        return False, None, duration, str(exc)


def test_rag() -> TestResult:
    """测试 RAG retriever。"""
    from zhixia.llm.rag.null_retriever import NullRAGRetriever

    t0 = time.perf_counter()
    try:
        retriever = NullRAGRetriever()
        context = retriever.retrieve("测试查询", top_k=2)
        duration = (time.perf_counter() - t0) * 1000
        return True, f"chunks={len(context.chunks)}, source={context.source_description}", duration, None
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        return False, None, duration, str(exc)


def test_output_parser() -> TestResult:
    """测试输出解析器。"""
    t0 = time.perf_counter()
    try:
        cases: list[tuple[str, str]] = [
            ('{"text": "你好", "emotion": "happy"}', "标准JSON"),
            ('{"text": "回答", "emotion": "thinking", "detail": "思考中"}', "带元数据"),
            ("这是普通回复", "普通文本"),
            ('<think>思考</think>最终答案', "带思考标签"),
            ('前缀{"text": "提取", "emotion": "sad"}后缀', "JSON嵌入文本"),
            ('[emotion:angry]很生气', "情绪前缀"),
        ]

        results: list[tuple[str, StructuredOutput]] = []
        for raw, desc in cases:
            parsed = parse_llm_output(raw)
            results.append((desc, parsed))

        duration = (time.perf_counter() - t0) * 1000
        return True, results, duration, None
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        return False, None, duration, str(exc)
