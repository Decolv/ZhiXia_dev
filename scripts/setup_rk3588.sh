#!/bin/bash
# ZhiXia RK3588 一键部署脚本
# 在 Debian/Ubuntu ARM64 系统上运行，自动配置运行环境

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  ZhiXia RK3588 部署脚本${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# 检查架构
check_arch() {
    local arch
    arch=$(uname -m)
    log_info "检测到系统架构: $arch"
    if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
        log_warn "非 ARM64 架构 ($arch)，此脚本主要为 RK3588 设计"
        read -rp "是否继续? [y/N] " yn
        if [[ ! "$yn" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 检查 Debian/Ubuntu
check_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log_info "操作系统: $NAME $VERSION_ID"
    else
        log_warn "无法检测操作系统"
    fi
}

# 安装系统依赖
install_system_deps() {
    log_info "安装系统依赖..."
    local deps=(
        python3
        python3-pip
        python3-venv
        alsa-utils
        libsndfile1
        portaudio19-dev
        wget
        curl
        git
    )

    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y "${deps[@]}"
    elif command -v apk &> /dev/null; then
        sudo apk add --no-cache alsa-utils wget curl git
    else
        log_warn "不支持的包管理器，请手动安装: ${deps[*]}"
    fi
    log_ok "系统依赖安装完成"
}

# 创建虚拟环境（可选）
setup_venv() {
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        log_info "虚拟环境已存在: $PROJECT_ROOT/.venv"
        return
    fi

    read -rp "是否创建 Python 虚拟环境 (.venv)? [Y/n] " yn
    if [[ ! "$yn" =~ ^[Nn]$ ]]; then
        log_info "创建虚拟环境..."
        python3 -m venv "$PROJECT_ROOT/.venv"
        log_ok "虚拟环境创建完成"
    fi
}

# 获取 Python 解释器路径
get_python() {
    if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
        echo "$PROJECT_ROOT/.venv/bin/python"
    else
        echo "python3"
    fi
}

# 安装 Python 依赖
install_python_deps() {
    local py
    py=$(get_python)
    log_info "使用 Python: $py"

    log_info "升级 pip..."
    "$py" -m pip install --upgrade pip setuptools wheel

    log_info "安装项目依赖（含 ASR FunASR + TTS Piper）..."
    "$py" -m pip install -e "$PROJECT_ROOT[asr-funasr,tts-piper,dev]"

    log_ok "Python 依赖安装完成"
}

# 创建目录结构
setup_dirs() {
    log_info "创建项目目录..."
    mkdir -p "$PROJECT_ROOT/models/piper"
    mkdir -p "$PROJECT_ROOT/models/snowboy"
    mkdir -p "$PROJECT_ROOT/output"
    mkdir -p "$PROJECT_ROOT/rknn_libs"
    mkdir -p "$PROJECT_ROOT/.cache/modelscope"
    log_ok "目录创建完成"
}

# 安装 Piper 模型
install_piper_model() {
    local model_dir="$PROJECT_ROOT/models/piper"
    local model_name="zh_CN-huayan-medium"

    if [ -f "$model_dir/${model_name}.onnx" ]; then
        log_ok "Piper 模型已存在"
        return
    fi

    log_info "下载 Piper 中文语音模型 (~42MB)..."
    local base_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/"

    if ! wget -q --show-progress "${base_url}${model_name}.onnx" -O "$model_dir/${model_name}.onnx"; then
        log_error "Piper 模型下载失败"
        log_info "请手动下载并放置到: $model_dir/"
        return 1
    fi
    wget -q --show-progress "${base_url}${model_name}.onnx.json" -O "$model_dir/${model_name}.onnx.json"

    log_ok "Piper 模型下载完成"
}

# 检查 RKNN 库
check_rknn_libs() {
    local lib_paths=(
        "$PROJECT_ROOT/rknn_libs/librkllmrt.so"
        "/usr/lib/librkllmrt.so"
        "/usr/local/lib/librkllmrt.so"
    )

    for path in "${lib_paths[@]}"; do
        if [ -f "$path" ]; then
            log_ok "找到 RKLLM 运行时库: $path"
            return
        fi
    done

    log_warn "未找到 RKLLM 运行时库 (librkllmrt.so)"
    echo ""
    echo "请从 RKNN SDK 复制库文件到项目目录:"
    echo "  cp /path/to/rknn_sdk/rknn_libs/* $PROJECT_ROOT/rknn_libs/"
    echo ""
    echo "或安装到系统目录:"
    echo "  sudo cp /path/to/rknn_sdk/rknn_libs/librkllmrt.so /usr/lib/"
    echo "  sudo ldconfig"
    echo ""
}

# 配置环境变量建议
suggest_env() {
    local env_file="$PROJECT_ROOT/.env"
    cat > "$env_file" << 'EOF'
# ZhiXia 环境变量配置（RK3588）
# 请 source 此文件或将其加入 ~/.bashrc

# ModelScope 缓存目录
export MODELSCOPE_CACHE="$PROJECT_ROOT/.cache/modelscope"

# RKNN 库路径（如果库在项目目录中）
if [ -d "$PROJECT_ROOT/rknn_libs" ]; then
    export LD_LIBRARY_PATH="$PROJECT_ROOT/rknn_libs:$LD_LIBRARY_PATH"
fi

# 允许使用 Fake LLM（PC 测试模式，RK3588 生产环境不要设置）
# export ZHIXIA_ALLOW_FAKE_LLM=1

# 指定配置文件（可选）
# export ZHIXIA_CONFIG="localconfig/localconfig.json"
EOF
    sed -i "s|\$PROJECT_ROOT|$PROJECT_ROOT|g" "$env_file"
    log_ok "环境变量模板已写入: $env_file"
}

# 打印完成信息
print_finish() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  ZhiXia 部署完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${CYAN}后续步骤:${NC}"
    echo ""
    echo "1. 激活环境（如创建了虚拟环境）:"
    echo "   source $PROJECT_ROOT/.venv/bin/activate"
    echo ""
    echo "2. 加载环境变量:"
    echo "   source $PROJECT_ROOT/.env"
    echo ""
    echo "3. 获取 RKLLM 模型文件 (~2.2GB):"
    echo "   将 Qwen3-1.7B-w8a8-rk3588.rkllm 放置到:"
    echo "   $PROJECT_ROOT/models/"
    echo ""
    echo "4. 运行环境检查:"
    echo "   python3 tests/quick_test.py check"
    echo ""
    echo "5. 启动语音助手:"
    echo "   bash run.sh"
    echo "   # 或"
    echo "   python3 -m zhixia"
    echo ""
    echo -e "${YELLOW}提示:${NC}"
    echo "  • 首次启动会自动下载 FunASR 模型（约 200MB）"
    echo "  • 如需使用唤醒词，请放置 snowboy 模型到 models/snowboy/"
    echo "  • 详细配置请编辑: localconfig/localconfig.json"
    echo ""
}

# 主流程
main() {
    print_banner
    check_arch
    check_os

    read -rp "开始部署 ZhiXia 到 RK3588? [Y/n] " yn
    if [[ "$yn" =~ ^[Nn]$ ]]; then
        exit 0
    fi

    install_system_deps
    setup_venv
    setup_dirs
    install_python_deps
    install_piper_model
    check_rknn_libs
    suggest_env
    print_finish
}

main "$@"
