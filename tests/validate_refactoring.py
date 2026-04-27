#!/usr/bin/env python3
"""验证重构是否成功的简单脚本"""

import sys
from pathlib import Path

def test_imports():
    """测试所有模块能否正确导入"""
    print("验证模块导入...")

    try:
        # 配置
        from zhixia.config.settings import AppSettings
        print("[OK] config.settings")

        # ASR
        from zhixia.asr.base import ASREngine, ASRResult
        from zhixia.asr.funasr_engine import FunASREngine
        from zhixia.asr.whisper_engine import WhisperASREngine
        print("[OK] asr.*")

        # LLM
        from zhixia.llm.base import LLMEngine, LLMMessage, StructuredOutput
        from zhixia.llm.rkllm_engine import RKLLMEngine
        from zhixia.llm.output_parser import parse_llm_output
        from zhixia.llm.rag.base import RAGRetriever, RAGContext
        from zhixia.llm.rag.null_retriever import NullRAGRetriever
        print("[OK] llm.*")

        # TTS
        from zhixia.tts.base import TTSEngine
        from zhixia.tts.piper_engine import PiperTTSEngine
        print("[OK] tts.*")

        # Audio
        from zhixia.audio.base import AudioPlayer
        from zhixia.audio.player import ALSAAudioPlayer
        from zhixia.audio.recorder import AudioRecorder
        print("[OK] audio.*")

        # Display
        from zhixia.display.base import DisplayOutput, DisplayPayload
        from zhixia.display.null_display import NullDisplay
        print("[OK] display.*")

        # Pipeline
        from zhixia.pipeline.orchestrator import VoicePipeline
        print("[OK] pipeline.orchestrator")

        # Utils
        from zhixia.utils.logging import setup_logging
        from zhixia.utils.memory import force_gc
        print("[OK] utils.*")

        print("\n[SUCCESS] 所有模块导入成功！")
        return True

    except ImportError as e:
        print(f"\n[ERROR] 导入失败: {e}")
        return False

def test_config():
    """测试配置加载"""
    print("\n验证配置加载...")
    try:
        from zhixia.config.settings import AppSettings
        config = AppSettings.load()
        print("[OK] 配置加载成功")
        print(f"   项目根目录: {config.project_root}")
        print(f"   LLM 引擎: {config.llm.engine}")
        print(f"   TTS 引擎: {config.tts.engine}")
        print(f"   ASR 引擎: {config.asr.engine}")
        return True
    except Exception as e:
        print(f"[ERROR] 配置加载失败: {e}")
        return False

def test_output_parser():
    """测试输出解析器"""
    print("\n验证输出解析器...")
    try:
        from zhixia.llm.output_parser import parse_llm_output

        # 测试 JSON 解析
        result = parse_llm_output('{"text": "你好", "emotion": "happy"}')
        assert result.text == "你好"
        assert result.emotion == "happy"
        print("[OK] JSON 解析正确")

        # 测试普通文本
        result = parse_llm_output("普通文本")
        assert result.text == "普通文本"
        assert result.emotion == "neutral"
        print("[OK] 普通文本解析正确")

        return True
    except Exception as e:
        print(f"[ERROR] 输出解析器测试失败: {e}")
        return False

def test_entry_point():
    """测试入口点（不需要实际运行）"""
    print("\n验证入口点...")
    try:
        import importlib.util
        spec = importlib.util.find_spec("zhixia.__main__")
        if spec is None:
            print("[ERROR] __main__.py 不存在")
            return False

        print("[OK] 主入口点存在")

        # 检查脚本是否能找到
        entry_script = Path(__file__).parent / "asr_llm_tts_piper.py"
        if not entry_script.exists():
            print("[ERROR] shim 脚本不存在")
            return False

        print("[OK] 向后兼容脚本存在")
        return True

    except Exception as e:
        print(f"[ERROR] 入口点验证失败: {e}")
        return False

def main():
    """主验证函数"""
    print("=" * 60)
    print("ZhiXia 重构验证")
    print("=" * 60)

    tests = [
        test_imports,
        test_config,
        test_output_parser,
        test_entry_point,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    if all(results):
        print("[SUCCESS] 验证全部通过！重构成功。")
        print("\n下一步：")
        print("1. 在开发机上运行 tests/test_pipeline_stages.ipynb")
        print("2. 在 RK3588 上部署并测试完整功能")
        return True
    else:
        print("[FAILED] 部分验证失败，请检查实现。")
        failed = [i+1 for i, success in enumerate(results) if not success]
        print(f"失败的测试: {failed}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)