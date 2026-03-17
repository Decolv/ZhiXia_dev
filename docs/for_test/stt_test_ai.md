# RDK STT 详细测试协议 (AI/开发者版)

> **Context**: 本文档提供给 AI 代理或开发者进行 RDK 终端 STT 通道的全路径校验。
> **Scope**: 覆盖从代理环境修复、PortAudio 延迟加载验证到模型推理链路的整体测试。

## 1. 代理与镜像源配置 (HuggingFace 修复方案)
若报错 `Unknown scheme for proxy (socks://)`, 必须按如下顺序重置代理环境变量：
```bash
# 环境变量重置 (httpx 不支持 socks)
unset all_proxy ALL_PROXY socks_proxy SOCKS_PROXY
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"

# 模型下载加速镜像 (如有必要)
export HF_ENDPOINT="https://hf-mirror.com"
```

## 2. 推理逻辑无感验证 (测试模式: --wav)
旨在脱离 PortAudio 依赖 (解决 `OSError: PortAudio library not found` 阻塞) 验证识别代码的核心部分：
- **执行路径**: `main/cli_stt_rdk.py`
- **验证命令**:
```bash
# 自动通过 wav 输入模式跳过 recorder 的加载
python3 main/cli_stt_rdk.py --wav tests/audio/test_800hz.wav
```
- **关键逻辑点**:
  - `main/services/audio_recorder.py` 的 `import sounddevice` 已在 `main/cli_stt_rdk.py` 内部被延迟加载（`lazy import`）。
  - 在指定 `--wav` 参数时，不会触发 `AudioRecorder` 初始化，从而避免硬件层面的 OSError。

## 3. RDK 实机硬件集成验证 (测试模式: CLI 默认)
用于验证本地麦克风权限及单声道 16kHz WAV 的录制能力：
- **前置依赖**: `sudo apt-get install libportaudio2`
- **验证命令**:
```bash
python3 main/cli_stt_rdk.py --config config-edge-rdk-stt.yaml
```
- **配置参考**: [config-edge-rdk-stt.yaml](config-edge-rdk-stt.yaml) (默认 tiny + int8 + cpu + zh)。

## 4. 测试输出基准
- **成功**: 终端显示 `🧠 正在识别...` 后返回文本或识别为空的警告。
- **失败 (硬件无麦克风)**: 初始化环节会截获并提供建议（1-2-3 步）。
- **失败 (代理/连接错误)**: 首次运行时无法下载模型，报错 `ValueError: Unknown scheme...`。

## 5. 组件健康检查清单
- [x] venv 已激活
- [x] `faster-whisper` 已安装
- [x] `tiny` 模型预热完毕
- [ ] `libportaudio2` 系统库检查

---
**Author**: GitHub Copilot (Gemini 3 Flash)
**Date**: 2026-03-17
