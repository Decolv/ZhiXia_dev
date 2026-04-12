# 目录放置位置声明

## 模型文件放置位置

### 1. models/ 目录
- **用途**: 存放RKLLM模型文件
- **位置**: 项目根目录下的 `models/` 文件夹
- **推荐模型**: `Qwen3-1.7B-w8a8-rk3588.rkllm`
- **大小**: 约2.2GB
- **获取方式**: 从RKLLM官方渠道下载或通过 `convert_to_rkllm.py` 转换

### 2. asset/ 目录
- **用途**: 存放ChatTTS等模型文件
- **位置**: 项目根目录下的 `asset/` 文件夹
- **包含文件**:
  - Decoder.safetensors (约100MB)
  - DVAE.safetensors (约58MB)
  - Embed.safetensors (约139MB)
  - Vocos.safetensors (约52MB)
  - gpt/ 目录
  - tokenizer/ 目录
- **获取方式**: 从ChatTTS官方仓库下载

### 3. rknn_libs/ 目录
- **用途**: 存放RKNN运行时库和工具
- **位置**: 项目根目录下的 `rknn_libs/` 文件夹
- **包含文件**:
  - librkllmrt.so (RKLLM运行时库)
  - librknnrt.so (RKNN运行时库)
  - rkllm.h (头文件)
  - drivers/ 目录
  - rknn-llm-main/ 目录
- **获取方式**: 从瑞芯微官方网站下载RKNN SDK

## 配置说明

### 环境变量
- **RKNN_LIB_PATH**: 指向rknn_libs目录
- **MODEL_PATH**: 指向models目录
- **ASSET_PATH**: 指向asset目录

### 依赖安装
1. 运行 `install_dependencies.sh` 安装Python依赖
2. 运行 `setup_rknpu.sh` 配置RKNN环境

## 运行脚本
- **run_fast_tts.sh**: 快速TTS运行脚本
- **run_npu_only.sh**: NPU推理运行脚本

## 注意事项
- 这些目录包含大型模型文件，已在 `.gitignore` 中配置为忽略
- 首次运行时需要下载或转换模型文件到指定位置
- 确保有足够的磁盘空间存放模型文件（至少需要3GB空间）