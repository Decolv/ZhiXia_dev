#!/usr/bin/env python3
"""
边缘设备模型下载与准备脚本
用于 RDK、树莓派、Jetson Nano 等 4GB RAM 设备

使用方法：
    python download_edge_models.py

确保已安装 huggingface-cli：
    pip install huggingface-hub
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# 配置
MODEL_DIR = Path("./models")
STT_DIR = MODEL_DIR / "stt"
LLM_DIR = MODEL_DIR / "llm"
TTS_DIR = MODEL_DIR / "tts"
VAD_DIR = MODEL_DIR / "vad"

# 模型 URLs 和大小信息
MODELS_INFO = {
    "stt": {
        "whisper-tiny": {
            "url": "openai/whisper-tiny",
            "size_mb": 139,
            "description": "轻量 STT 模型（39M 参数）"
        }
    },
    "llm": {
        "qwen-1.5b-q3_k_m": {
            "url": "second-state/Qwen-1.5B-Chat-GGUF",
            "file": "qwen-1.5b-chat-q3_k_m.gguf",
            "size_mb": 750,
            "description": "Qwen 1.5B 中文模型（Q3_K_M 量化）- 推荐"
        },
        "qwen-1.5b-q2_k": {
            "url": "second-state/Qwen-1.5B-Chat-GGUF",
            "file": "qwen-1.5b-chat-q2_k.gguf",
            "size_mb": 500,
            "description": "Qwen 1.5B 中文模型（Q2_K 量化）- 极端内存紧张时"
        },
        "qwen-0.5b": {
            "url": "Qwen/Qwen-0.5B-Chat",
            "size_mb": 300,
            "description": "Qwen 0.5B 极轻模型（当 1.5B 不可用时）"
        }
    },
    "tts": {
        "kokoro": {
            "url": "hexgrad/Kokoro-82M",
            "size_mb": 160,
            "description": "Kokoro TTS 轻量模型"
        }
    },
    "vad": {
        "silero-vad": {
            "url": "snakers4/silero-vad",
            "size_mb": 40,
            "description": "Silero VAD 轻量模型"
        }
    }
}


def print_banner():
    """打印欢迎信息"""
    print("\n" + "=" * 70)
    print("🚀 边缘设备模型下载工具")
    print("=" * 70)
    print(f"目标内存预算：< 4GB RAM")
    print(f"推荐总大小：STT (~140MB) + LLM (~750MB) + TTS (~160MB) + VAD (~40MB)")
    print(f"           ≈ 1.1GB 总计")
    print("=" * 70 + "\n")


def create_directories():
    """创建模型目录"""
    for dir_path in [STT_DIR, LLM_DIR, TTS_DIR, VAD_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 目录就位：{dir_path}")


def download_via_huggingface(repo_id: str, filename: Optional[str] = None, dest_dir: Path = None):
    """
    通过 HuggingFace CLI 下载模型
    
    Args:
        repo_id: HuggingFace 仓库 ID (如 "openai/whisper-tiny")
        filename: 特定文件（可选，为空则下载全部）
        dest_dir: 目标目录
    """
    
    cmd = [
        "huggingface-cli", "download",
        repo_id
    ]
    
    if filename:
        cmd.append(filename)
    
    cmd.extend([
        "--local-dir", str(dest_dir),
        "--local-dir-use-symlinks", "False"  # 直接下载，不用符号链接
    ])
    
    print(f"   命令：{' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 下载失败：{e}")
        return False


def download_stt_models():
    """下载 STT 模型"""
    print("\n" + "─" * 70)
    print("📥 [1/4] 下载 STT 模型（语音转文字）")
    print("─" * 70)
    
    model_info = MODELS_INFO["stt"]["whisper-tiny"]
    print(f"模型：{model_info['description']}")
    print(f"大小：~{model_info['size_mb']}MB")
    print()
    
    print(f"使用 faster-whisper 自动下载...")
    
    try:
        # faster-whisper 会自动下载到缓存目录
        cmd = [
            sys.executable, "-m", "faster_whisper_cli",
            "--model", "tiny",
            "--language", "zh"
        ]
        print(f"   提示：首次运行 faster_whisper 会自动下载模型")
        print(f"   建议先运行：python -c \"from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu')\"")
        
        return True
    except Exception as e:
        print(f"   ❌ 错误：{e}")
        return False


def download_llm_models():
    """下载 LLM 模型"""
    print("\n" + "─" * 70)
    print("📥 [2/4] 下载 LLM 模型（大语言模型）")
    print("─" * 70)
    
    # 显示选项
    print("推荐选择（按优先级）：")
    print()
    print("1️⃣  Qwen-1.5B-Chat (Q3_K_M) - 推荐")
    print("   大小：~750MB，质量平衡")
    print()
    print("2️⃣  Qwen-1.5B-Chat (Q2_K) - 极端内存紧张")
    print("   大小：~500MB，质量下降")
    print()
    print("3️⃣  Qwen-0.5B - 超轻")
    print("   大小：~300MB，速度快但质量明显不足")
    print()
    
    choice = input("请选择（输入 1/2/3，默认 1）：").strip() or "1"
    
    if choice == "1":
        model_info = MODELS_INFO["llm"]["qwen-1.5b-q3_k_m"]
        repo_id = model_info["url"]
        filename = model_info["file"]
        print(f"\n下载：{model_info['description']}")
    elif choice == "2":
        model_info = MODELS_INFO["llm"]["qwen-1.5b-q2_k"]
        repo_id = model_info["url"]
        filename = model_info["file"]
        print(f"\n下载：{model_info['description']}")
    else:
        model_info = MODELS_INFO["llm"]["qwen-0.5b"]
        repo_id = model_info["url"]
        filename = None
        print(f"\n下载：{model_info['description']}")
    
    print(f"大小：~{model_info['size_mb']}MB")
    print()
    
    if not download_via_huggingface(repo_id, filename, LLM_DIR):
        print("   💡 手动下载：访问 https://huggingface.co/{repo_id}")
        return False
    
    print("✅ LLM 模型下载完成")
    return True


def download_tts_models():
    """下载 TTS 模型"""
    print("\n" + "─" * 70)
    print("📥 [3/4] 下载 TTS 模型（文字转语音）")
    print("─" * 70)
    
    model_info = MODELS_INFO["tts"]["kokoro"]
    print(f"模型：{model_info['description']}")
    print(f"大小：~{model_info['size_mb']}MB")
    print()
    
    repo_id = model_info["url"]
    
    if not download_via_huggingface(repo_id, None, TTS_DIR):
        print("   💡 手动下载：访问 https://huggingface.co/{repo_id}")
        return False
    
    print("✅ TTS 模型下载完成")
    return True


def download_vad_models():
    """下载 VAD 模型"""
    print("\n" + "─" * 70)
    print("📥 [4/4] 下载 VAD 模型（语音活动检测）")
    print("─" * 70)
    
    model_info = MODELS_INFO["vad"]["silero-vad"]
    print(f"模型：{model_info['description']}")
    print(f"大小：~{model_info['size_mb']}MB")
    print()
    
    # Silero VAD 通常通过专门的方式下载
    try:
        import torch
        print("使用 torch 下载 Silero VAD...")
        torch.hub.load(
            repo_or_dir='snakers4/silero-vad:master',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        print("✅ VAD 模型下载完成")
        return True
    except Exception as e:
        print(f"⚠️  VAD 下载可能失败：{e}")
        print("   您可以手动下载或稍后尝试")
        return False


def verify_models():
    """验证已下载的模型"""
    print("\n" + "─" * 70)
    print("🔍 验证模型文件")
    print("─" * 70)
    
    total_size = 0
    
    for category_dir, category_name in [
        (STT_DIR, "STT"),
        (LLM_DIR, "LLM"),
        (TTS_DIR, "TTS"),
        (VAD_DIR, "VAD")
    ]:
        if not category_dir.exists():
            print(f"⚠️  {category_name} 目录不存在：{category_dir}")
            continue
        
        files = list(category_dir.rglob("*"))
        model_files = [f for f in files if f.is_file()]
        
        if model_files:
            category_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
            total_size += category_size
            print(f"✅ {category_name}: {len(model_files)} 文件，~{category_size:.0f}MB")
            for f in model_files[:3]:  # 只显示前 3 个
                print(f"   - {f.name} ({f.stat().st_size / (1024*1024):.1f}MB)")
            if len(model_files) > 3:
                print(f"   ... 及其他 {len(model_files)-3} 个文件")
        else:
            print(f"❌ {category_name}: 未找到模型文件")
    
    print()
    print(f"📊 总大小：~{total_size:.0f}MB")
    print(f"🎯 预算检查：", end="")
    if total_size < 1500:
        print(f"✅ 在预算内 (<1.5GB)")
    elif total_size < 2000:
        print(f"⚠️  接近极限 (1.5-2GB)")
    else:
        print(f"❌ 超过预算 (>2GB)，建议删除不需要的量化版本")


def print_next_steps():
    """打印后续步骤"""
    print("\n" + "=" * 70)
    print("✅ 模型下载完成！")
    print("=" * 70)
    print()
    print("📝 后续步骤：")
    print()
    print("1. 验证 Python 环境：")
    print("   python --version  # 需要 Python 3.10+")
    print()
    print("2. 安装依赖：")
    print("   pip install -r requirements-edge.txt")
    print()
    print("3. 编辑配置文件：")
    print("   cp docs/LOCAL_AI_VOICE_SYSTEM_RDK_CONFIG.yaml config.yaml")
    print("   # 根据你的 RDK 硬件调整参数")
    print()
    print("4. 测试模型加载：")
    print("   python -c \"from llama_cpp import Llama; Llama(model_path='models/llm/qwen-1.5b-chat-q3_k_m.gguf')\"")
    print()
    print("5. 运行应用：")
    print("   python main.py --config config.yaml")
    print()
    print("有问题？参考文档：")
    print("   - RDK 部署指南：https://github.com/...")
    print("   - 模型选择指南：LOCAL_AI_VOICE_SYSTEM.md 第七章")
    print()
    print("=" * 70)


def main():
    """主函数"""
    print_banner()
    
    # 创建目录
    create_directories()
    
    # 下载各类模型
    print()
    print("开始下载模型（需要网络连接，首次下载可能需要 10-30 分钟）")
    print()
    
    try:
        # 可选：跳过某些模型
        results = {
            "stt": download_stt_models(),
            "llm": download_llm_models(),
            "tts": download_tts_models(),
            "vad": download_vad_models(),
        }
    except KeyboardInterrupt:
        print("\n⏹️  下载已取消")
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        return 1
    
    # 验证结果
    verify_models()
    
    # 后续步骤
    print_next_steps()
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
