# 智能语音助手 - Piper TTS版本

## 简介

这是一个为QuarkPi (RK3588)优化的智能语音助手，使用：
- **ASR**: FunASR (INT8量化)
- **LLM**: RKLLM (NPU加速，快速响应模式)
- **TTS**: Piper (超高速语音合成)

## 性能特点

- **TTS速度**: 0.5-1秒（比ChatTTS快10-20倍）
- **LLM响应**: 快速模式（max_tokens=32）
- **总响应时间**: 3-5秒
- **模型大小**: Piper仅42MB
- **完全离线**: 无需网络连接

## 快速开始

### 1. 安装Piper TTS

```bash
bash install_fast_tts.sh
```

这将自动：
- 安装Piper TTS（Python包或二进制）
- 下载中文模型（华研音色，42MB）
- 安装音频播放器（aplay）

### 2. 运行语音助手

```bash
python3 asr_llm_tts_piper.py
```

## 文件说明

```
ZhiXia_dev/
├── asr_llm_tts_piper.py      # 主程序（仅Piper TTS）
├── install_fast_tts.sh        # 安装脚本
├── rkllm_inference.py         # RKLLM推理模块
├── models/
│   ├── Qwen3-1.7B-w8a8-rk3588.rkllm  # LLM模型
│   └── piper/
│       ├── zh_CN-huayan-medium.onnx       # Piper模型
│       └── zh_CN-huayan-medium.onnx.json  # 配置文件
└── output/
    └── llm_response_piper.wav  # 输出音频
```

## 配置说明

### 调整LLM响应长度

在 `asr_llm_tts_piper.py` 中修改：

```python
# 更快响应（默认）
llm_response = llm_inference_npu_stream(recognized_text, max_new_tokens=32)

# 更长回复
llm_response = llm_inference_npu_stream(recognized_text, max_new_tokens=64)
```

### 调整输入音频路径

```python
input_audio = "/home/quark/音乐/test.wav"  # 修改为你的音频路径
```

### 使用更小的Piper模型

如果内存不足，可以使用x_low版本（18MB）：

```bash
cd models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx.json
```

然后在代码中修改模型路径。

## 性能优化

### 1. LLM优化
- 使用快速响应模式（max_new_tokens=32）
- 减少上下文长度（max_context_len=512）
- 提高温度参数（temperature=0.8）

### 2. TTS优化
- Piper已经是最快的选择
- 使用ONNX优化，专为ARM设备设计
- 模型小，加载快

### 3. 内存优化
- 每个步骤后强制垃圾回收
- 及时释放模型资源
- 使用INT8量化的ASR模型

## 故障排除

### 问题1: Piper安装失败

```bash
# 方法1: 使用pip
pip3 install piper-tts

# 方法2: 从源码安装
pip3 install git+https://github.com/rhasspy/piper.git

# 方法3: 使用二进制（ARM64）
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar -xzf piper_arm64.tar.gz
sudo mv piper/piper /usr/local/bin/
```

### 问题2: 模型下载失败

手动下载模型：
1. 访问: https://huggingface.co/rhasspy/piper-voices/tree/main/zh/zh_CN/huayan/medium
2. 下载 `zh_CN-huayan-medium.onnx` 和 `zh_CN-huayan-medium.onnx.json`
3. 放到 `models/piper/` 目录

### 问题3: 音频播放失败

```bash
# 安装aplay
sudo apt-get install alsa-utils

# 测试音频设备
aplay -l

# 手动播放
aplay output/llm_response_piper.wav
```

### 问题4: RKLLM模型不存在

确保 `models/Qwen3-1.7B-w8a8-rk3588.rkllm` 存在。
参考 `README_RKLLM.md` 获取模型。

## 性能对比

| 组件 | 之前（ChatTTS） | 现在（Piper） | 提升 |
|------|----------------|--------------|------|
| TTS速度 | 5-10秒 | 0.5-1秒 | 10x |
| 模型大小 | 800MB+ | 42MB | 20x |
| 内存占用 | 高 | 低 | 显著 |
| 总响应时间 | 15-20秒 | 3-5秒 | 4x |

## 技术细节

### Piper TTS
- 基于VITS架构
- ONNX Runtime优化
- 支持多种中文音色
- 实时率: 10-20x（CPU）

### LLM快速模式
- max_new_tokens: 32（约1-2句话）
- max_context_len: 512（减少计算）
- temperature: 0.8（加快采样）
- system_prompt: "用一句话简短回答"

### ASR优化
- INT8量化模型
- 禁用VAD和标点模型
- 减少推理时间

## 参考资料

- [Piper TTS GitHub](https://github.com/rhasspy/piper)
- [Piper语音模型](https://huggingface.co/rhasspy/piper-voices)
- [RKLLM文档](README_RKLLM.md)
- [FunASR文档](https://github.com/alibaba-damo-academy/FunASR)

## 许可证

- Piper TTS: MIT License
- FunASR: Apache-2.0
- RKLLM: 参考瑞芯微官方许可

## 致谢

- Rhasspy团队 - Piper TTS
- 阿里巴巴达摩院 - FunASR
- 瑞芯微 - RKLLM
