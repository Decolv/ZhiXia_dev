#!/bin/bash
# 快速启动智能语音助手

echo "=================================="
echo "🎙️ 智能语音助手 - Piper版"
echo "=================================="
echo ""

# 检查Piper是否安装
if ! command -v piper &> /dev/null && ! python3 -c "import piper" 2>/dev/null; then
    echo "❌ Piper TTS未安装"
    echo ""
    echo "请先运行安装脚本:"
    echo "  bash install_fast_tts.sh"
    echo ""
    exit 1
fi

# 检查模型是否存在
if [ ! -f "models/piper/zh_CN-huayan-medium.onnx" ]; then
    echo "❌ Piper模型不存在"
    echo ""
    echo "请先运行安装脚本:"
    echo "  bash install_fast_tts.sh"
    echo ""
    exit 1
fi

if [ ! -f "models/Qwen3-1.7B-w8a8-rk3588.rkllm" ]; then
    echo "❌ RKLLM模型不存在"
    echo ""
    echo "请参考 README_RKLLM.md 获取模型"
    echo ""
    exit 1
fi

echo "✅ 环境检查通过"
echo ""
echo "启动语音助手..."
echo ""

# 运行主程序
python3 asr_llm_tts_piper.py

echo ""
echo "=================================="
echo "程序结束"
echo "=================================="
