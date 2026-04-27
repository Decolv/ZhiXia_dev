#!/bin/bash
# 安装Piper TTS

set -e

echo "=================================="
echo "安装Piper TTS"
echo "=================================="
echo ""

# 检测系统架构
ARCH=$(uname -m)
echo "检测到系统架构: $ARCH"
echo ""

# 创建模型目录
echo "创建模型目录..."
mkdir -p models/piper
echo "✅ 目录创建完成"
echo ""

# 安装Piper TTS
echo "正在安装Piper TTS..."

# 尝试pip安装
if pip3 install piper-tts 2>/dev/null; then
    echo "✅ Piper TTS安装成功 (Python包)"
else
    echo "⚠️ Python包安装失败，尝试二进制安装..."
    
    # 下载二进制版本
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        echo "下载ARM64版本..."
        cd /tmp
        wget -q --show-progress https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
        tar -xzf piper_arm64.tar.gz
        sudo mv piper/piper /usr/local/bin/
        sudo chmod +x /usr/local/bin/piper
        rm -rf piper piper_arm64.tar.gz
        cd -
        echo "✅ Piper二进制安装成功"
    else
        echo "❌ 不支持的架构: $ARCH"
        echo "请手动安装: pip3 install piper-tts"
        exit 1
    fi
fi

echo ""

# 下载中文模型
echo "下载Piper中文模型..."
cd models/piper

if [ ! -f "zh_CN-huayan-medium.onnx" ]; then
    echo "下载华研音色模型 (42MB)..."
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
    wget -q --show-progress https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
    echo "✅ 模型下载完成"
else
    echo "✅ 模型已存在"
fi

cd ../..

echo ""

# 安装音频播放器
echo "=================================="
echo "安装音频播放器"
echo "=================================="
echo ""

if ! command -v aplay &> /dev/null; then
    echo "安装aplay..."
    sudo apt-get update
    sudo apt-get install -y alsa-utils
    echo "✅ aplay安装完成"
else
    echo "✅ aplay已安装"
fi

echo ""

# 测试安装
echo "=================================="
echo "测试安装"
echo "=================================="
echo ""

echo "测试Piper..."
if command -v piper &> /dev/null; then
    piper --version
    echo "✅ Piper命令行可用"
elif python3 -c "import piper" 2>/dev/null; then
    echo "✅ Piper Python包可用"
else
    echo "❌ Piper未安装成功"
    exit 1
fi

echo ""
echo "=================================="
echo "安装完成！"
echo "=================================="
echo ""
echo "推荐使用以下方式启动语音助手:"
echo "  python -m zhixia"
echo ""
