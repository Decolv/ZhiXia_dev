#!/usr/bin/env python3
"""快速真机测试工具。

用法:
    python3 tests/quick_test.py check      # 环境检查
    python3 tests/quick_test.py asr        # ASR 测试
    python3 tests/quick_test.py llm        # LLM 测试
    python3 tests/quick_test.py tts        # TTS 测试
    python3 tests/quick_test.py pipeline   # 端到端 Pipeline 测试
    python3 tests/quick_test.py all        # 全部测试
    python3 tests/quick_test.py check asr  # 组合测试

环境变量:
    ZHIXIA_CONFIG    — 指定配置文件路径（默认 localconfig/localconfig.json）
    ZHIXIA_ALLOW_FAKE_LLM — PC测试时设为1
"""

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.test_core import (
    test_asr,
    test_config,
    test_display,
    test_llm,
    test_output_parser,
    test_pipeline,
    test_rag,
    test_tts,
)
from zhixia.config.settings import AppSettings
from zhixia.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# 简单的颜色标记（如果终端支持）
_PASS = "✅"
_FAIL = "❌"
_WARN = "⚠️"
_INFO = "ℹ️"


def _check_mark(success: bool) -> str:
    return _PASS if success else _FAIL


def _print_result(name: str, success: bool, result: any, duration_ms: float, error: str | None) -> None:
    mark = _check_mark(success)
    if success:
        print(f"  {mark} {name}: {result} ({duration_ms:.0f}ms)")
    else:
        print(f"  {mark} {name} ({duration_ms:.0f}ms)")
        if error:
            print(f"     错误: {error}")


def _load_config() -> AppSettings:
    config_override = os.environ.get("ZHIXIA_CONFIG", "").strip()
    config_path = None
    if config_override:
        config_path = Path(config_override)
        if not config_path.is_absolute():
            config_path = _PROJECT_ROOT / config_path
        print(f"{_INFO} 使用配置: {config_path}")
    else:
        config_path = _PROJECT_ROOT / "localconfig" / "localconfig.json"
    return AppSettings.load(config_path=config_path)


# ------------------------------------------------------------------
# 环境检查
# ------------------------------------------------------------------

def run_check(config: AppSettings) -> list[bool]:
    """环境检查：模型文件、依赖库、pip包。"""
    print("\n🔍 环境检查")
    print("-" * 60)
    results: list[bool] = []

    # 1. RKLLM 模型文件
    model_path = config.project_root / config.llm.model_path
    ok = model_path.exists()
    results.append(ok)
    print(f"  {_check_mark(ok)} LLM 模型: {model_path}")
    if not ok:
        print(f"     提示: RKLLM 模型文件不存在，真机上需要手动下载")

    # 2. TTS 模型文件
    tts_model_path = config.project_root / config.tts.model_path
    tts_config_path = Path(str(tts_model_path) + ".json")
    ok = tts_model_path.exists() and tts_config_path.exists()
    results.append(ok)
    print(f"  {_check_mark(ok)} TTS 模型: {tts_model_path}")
    if not ok:
        print(f"     提示: Piper 模型不存在，运行时会自动下载")

    # 3. librkllmrt.so
    so_paths = [
        config.project_root / "rknn_libs" / "librkllmrt.so",
        Path("/usr/lib/librkllmrt.so"),
        Path("/usr/local/lib/librkllmrt.so"),
    ]
    so_found = any(p.exists() for p in so_paths)
    results.append(so_found)
    print(f"  {_check_mark(so_found)} librkllmrt.so")
    if not so_found:
        print(f"     提示: 未找到 RKLLM 运行时库，真机需要安装")

    # 4. rkllm_inference.py
    rkllm_py = config.project_root / "rkllm_inference.py"
    ok = rkllm_py.exists()
    results.append(ok)
    print(f"  {_check_mark(ok)} rkllm_inference.py")

    # 5. 测试音频文件
    test_audio = Path(config.asr.input_audio) if config.asr.input_audio else None
    if test_audio and not test_audio.is_absolute():
        test_audio = config.project_root / test_audio
    audio_ok = test_audio is not None and test_audio.exists()
    results.append(audio_ok)
    if audio_ok:
        print(f"  {_check_mark(True)} 测试音频: {test_audio}")
    else:
        print(f"  {_WARN} 测试音频: {test_audio} (不存在)")
        print(f"     提示: ASR / Pipeline 测试需要有效的音频文件")

    # 6. Python 依赖包
    packages = ["funasr", "piper", "zhixia"]
    for pkg in packages:
        spec = importlib.util.find_spec(pkg)
        ok = spec is not None
        results.append(ok)
        print(f"  {_check_mark(ok)} Python 包: {pkg}")

    # 7. 平台信息
    device_config = getattr(config, "device", None)
    if isinstance(device_config, dict):
        platform = device_config.get("platform", "unknown")
        npu = device_config.get("npu_enabled", False)
    else:
        platform = getattr(device_config, "platform", "unknown") if device_config else "unknown"
        npu = getattr(device_config, "npu_enabled", False) if device_config else False
    print(f"  {_INFO} 平台: {platform}, NPU: {npu}")

    # 8. Fake LLM 模式
    fake_llm = os.environ.get("ZHIXIA_ALLOW_FAKE_LLM", "").strip() == "1"
    if fake_llm:
        print(f"  {_WARN} Fake LLM 模式已启用 (ZHIXIA_ALLOW_FAKE_LLM=1)")

    return results


# ------------------------------------------------------------------
# 各模块测试
# ------------------------------------------------------------------

def run_asr(config: AppSettings) -> bool:
    print("\n🎤 ASR 测试")
    print("-" * 60)
    success, result, duration, error = test_asr(config)
    _print_result("ASR", success, result, duration, error)
    return success


def run_llm(config: AppSettings) -> bool:
    print("\n🧠 LLM 测试")
    print("-" * 60)
    success, result, duration, error = test_llm(config)
    _print_result("LLM", success, result, duration, error)
    return success


def run_tts(config: AppSettings) -> bool:
    print("\n🔊 TTS 测试")
    print("-" * 60)
    success, result, duration, error = test_tts(config)
    _print_result("TTS", success, result, duration, error)
    return success


def run_pipeline(config: AppSettings) -> bool:
    print("\n🔄 Pipeline 端到端测试")
    print("-" * 60)
    success, result, duration, error = test_pipeline(config)
    _print_result("Pipeline", success, result, duration, error)
    return success


def run_unit_tests() -> list[bool]:
    """纯代码单元测试（无模型加载）。"""
    print("\n🧪 单元测试")
    print("-" * 60)
    results = []

    # Display
    success, result, duration, error = test_display()
    results.append(success)
    _print_result("Display", success, result, duration, error)

    # RAG
    success, result, duration, error = test_rag()
    results.append(success)
    _print_result("RAG", success, result, duration, error)

    # Output Parser
    success, result, duration, error = test_output_parser()
    results.append(success)
    _print_result("Output Parser", success, f"{len(result)} cases passed", duration, error)
    if success:
        for desc, parsed in result:
            print(f"     • {desc}: text={parsed.text!r}, emotion={parsed.emotion!r}")

    return results


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ZhiXia 快速真机测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 tests/quick_test.py check      环境检查
  python3 tests/quick_test.py asr        ASR 测试
  python3 tests/quick_test.py llm        LLM 测试
  python3 tests/quick_test.py tts        TTS 测试
  python3 tests/quick_test.py pipeline   端到端测试
  python3 tests/quick_test.py all        全部测试
  python3 tests/quick_test.py check asr  组合测试
        """,
    )
    parser.add_argument(
        "commands",
        nargs="+",
        choices=["check", "asr", "llm", "tts", "pipeline", "all"],
        help="要执行的测试命令",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="指定配置文件路径",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )
    args = parser.parse_args()

    # 日志
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    # 确定要执行的命令
    commands = set(args.commands)
    if "all" in commands:
        commands = {"check", "asr", "llm", "tts", "pipeline"}

    print("=" * 60)
    print("🚀 ZhiXia 快速真机测试")
    print("=" * 60)
    print(f"项目目录: {_PROJECT_ROOT}")

    t_start = time.perf_counter()
    all_results: list[bool] = []

    # 加载配置（check 以外的测试需要）
    config = None
    if commands != {"check"}:
        if args.config:
            os.environ["ZHIXIA_CONFIG"] = str(args.config)
        config = _load_config()

    # 执行单元测试（无模型）
    if "all" in args.commands or any(c in commands for c in ["check", "pipeline"]):
        all_results.extend(run_unit_tests())

    # 环境检查
    if "check" in commands:
        check_results = run_check(config if config else _load_config())
        all_results.extend(check_results)

    # ASR
    if "asr" in commands:
        all_results.append(run_asr(config))

    # LLM
    if "llm" in commands:
        all_results.append(run_llm(config))

    # TTS
    if "tts" in commands:
        all_results.append(run_tts(config))

    # Pipeline
    if "pipeline" in commands:
        all_results.append(run_pipeline(config))

    # 汇总
    total = len(all_results)
    passed = sum(all_results)
    failed = total - passed
    total_duration = (time.perf_counter() - t_start) * 1000

    print("\n" + "=" * 60)
    print("📊 测试汇总")
    print("=" * 60)
    print(f"  总计: {total}")
    print(f"  {_PASS} 通过: {passed}")
    if failed > 0:
        print(f"  {_FAIL} 失败: {failed}")
    print(f"  ⏱️  总耗时: {total_duration / 1000:.1f}s")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
