#!/bin/bash
# ZhiXia 智能语音助手启动脚本（RK3588 优化版）

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}🎙️  ZhiXia 智能语音助手${NC}"
echo -e "${BLUE}==================================${NC}"
echo ""

# 检查 Python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 未找到 python3，请先安装 Python 3.9+${NC}"
    exit 1
fi

# 检查 Piper TTS
PIPER_OK=false
if command -v piper &> /dev/null || python3 -c "import piper" 2>/dev/null; then
    PIPER_OK=true
fi

if [ "$PIPER_OK" = false ]; then
    echo -e "${YELLOW}⚠️ Piper TTS 未安装${NC}"
    echo ""
    echo "请先运行安装脚本:"
    echo "  bash scripts/install_fast_tts.sh"
    echo ""
    read -rp "是否现在运行安装脚本? [y/N] " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        bash scripts/install_fast_tts.sh
    else
        echo -e "${RED}退出。请安装 Piper TTS 后再启动。${NC}"
        exit 1
    fi
fi

# 检查 Piper 模型
if [ ! -f "models/piper/zh_CN-huayan-medium.onnx" ]; then
    echo -e "${YELLOW}⚠️ Piper 模型不存在: models/piper/zh_CN-huayan-medium.onnx${NC}"
    echo ""
    echo "模型将尝试在首次运行时自动下载，或手动运行:"
    echo "  bash scripts/install_fast_tts.sh"
    echo ""
fi

# 检查 RKLLM 模型
if [ ! -f "models/Qwen3-1.7B-w8a8-rk3588.rkllm" ]; then
    echo -e "${YELLOW}⚠️ RKLLM 模型不存在: models/Qwen3-1.7B-w8a8-rk3588.rkllm${NC}"
    echo ""
    echo "请从以下渠道获取模型并放置到 models/ 目录:"
    echo "  1. 从 RKNN 官方 SDK 转换"
    echo "  2. 联系项目维护者获取预编译模型"
    echo ""
    read -rp "模型缺失，是否仍要启动（将使用回退模式或报错）? [y/N] " yn
    if [[ ! "$yn" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 RKNN 运行时库
if [ ! -f "rknn_libs/librkllmrt.so" ] && [ ! -f "/usr/lib/librkllmrt.so" ]; then
    echo -e "${YELLOW}⚠️ RKNN 运行时库未找到${NC}"
    echo ""
    echo "请确保以下之一存在:"
    echo "  - rknn_libs/librkllmrt.so (项目目录)"
    echo "  - /usr/lib/librkllmrt.so (系统目录)"
    echo ""
    echo "获取方式:"
    echo "  1. 从瑞芯微 RKNN SDK 复制"
    echo "  2. 运行: bash scripts/setup_rk3588.sh"
    echo ""
    read -rp "是否仍要启动? [y/N] " yn
    if [[ ! "$yn" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 ALSA
if ! command -v aplay &> /dev/null; then
    echo -e "${YELLOW}⚠️ 未找到 aplay，音频播放可能不可用${NC}"
    echo "建议安装: sudo apt-get install -y alsa-utils"
fi

echo -e "${GREEN}✅ 环境检查通过${NC}"
echo ""
echo -e "${GREEN}启动语音助手...${NC}"
echo ""

# 运行主程序
python3 -m zhixia

echo ""
echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}程序结束${NC}"
echo -e "${BLUE}==================================${NC}"
