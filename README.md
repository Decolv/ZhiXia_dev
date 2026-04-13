
# 智能语音助手 - QuarkPi版

## 项目简介

这是一个为QuarkPi (RK3588)优化的智能语音助手项目，实现完整的语音交互流程：

**语音输入 → 语音识别(ASR) → 大模型推理(LLM) → 语音合成(TTS) → 语音输出**

### 核心特性

- ✅ **完全离线**: 所有模型本地运行，无需网络
- ✅ **NPU加速**: 使用RKLLM在NPU上运行大模型
- ✅ **超快响应**: 总响应时间3-5秒
- ✅ **低资源占用**: 优化内存使用，适合嵌入式设备
- ✅ **高质量TTS**: Piper TTS，速度快音质好

## 技术栈

| 组件 | 技术方案 | 特点 |
|------|---------|------|
| ASR | FunASR (INT8量化) | 中文识别，速度快 |
| LLM | RKLLM (Qwen3-1.7B) | NPU加速，快速响应 |
| TTS | Piper | 超高速，模型小(42MB) |

## 快速开始

### 1. 安装依赖

```bash
# 安装Piper TTS和下载模型
bash install_fast_tts.sh
```

### 2. 准备RKLLM模型

确保 `models/Qwen3-1.7B-w8a8-rk3588.rkllm` 存在。
详见 [README_RKLLM.md](README_RKLLM.md)

### 3. 运行语音助手

```bash
# 方法1: 使用启动脚本
bash run.sh

# 方法2: 直接运行
python3 asr_llm_tts_piper.py
```

## 性能表现

在QuarkPi (RK3588)上的实测性能：

- **ASR识别**: 1-2秒
- **LLM推理**: 1-2秒（快速模式）
- **TTS合成**: 0.5-1秒
- **总响应**: 3-5秒

相比之前使用ChatTTS的15-20秒，速度提升约4倍！

## 项目结构

```
ZhiXia_dev/
├── asr_llm_tts_piper.py       # 主程序（推荐使用）
├── run.sh                      # 快速启动脚本
├── install_fast_tts.sh         # 安装脚本
├── rkllm_inference.py          # RKLLM推理模块
├── convert_to_rkllm.py         # 模型转换工具
├── README.md                   # 本文档
├── README_PIPER.md             # Piper详细说明
├── README_RKLLM.md             # RKLLM详细说明
├── foragent.md                 # 开发文档
├── models/
│   ├── Qwen3-1.7B-w8a8-rk3588.rkllm  # LLM模型
│   └── piper/
│       ├── zh_CN-huayan-medium.onnx       # Piper模型
│       └── zh_CN-huayan-medium.onnx.json  # 配置
└── output/
    └── llm_response_piper.wav  # 输出音频
```

## 配置说明

### 调整输入音频

编辑 `asr_llm_tts_piper.py`:

```python
input_audio = "/home/quark/音乐/test.wav"  # 修改为你的音频路径
```

### 调整LLM响应长度

```python
# 快速响应（默认，1-2句话）
llm_response = llm_inference_npu_stream(recognized_text, max_new_tokens=32)

# 更长回复（3-4句话）
llm_response = llm_inference_npu_stream(recognized_text, max_new_tokens=64)
```

### 调整LLM系统提示

```python
llm.set_chat_template(
    system_prompt="你是AI助手，用一句话简短回答。",  # 修改这里
    prompt_prefix="",
    prompt_postfix=""
)
```

## 优化建议

### 1. 进一步提升速度

- 使用更小的Piper模型（x_low版本，18MB）
- 减少max_new_tokens到16-24
- 使用更激进的采样参数

### 2. 提升音质

- 使用Piper的high质量模型
- 调整TTS语速和音调

### 3. 降低内存占用

- 使用INT8量化的所有模型
- 及时释放不用的模型
- 减少上下文长度

## 故障排除

### Piper安装失败

```bash
pip3 install piper-tts
# 或使用二进制版本（见 README_PIPER.md）
```

### 模型下载失败

手动下载：
- Piper模型: https://huggingface.co/rhasspy/piper-voices
- RKLLM模型: 参考 README_RKLLM.md

### 音频播放失败

```bash
sudo apt-get install alsa-utils
aplay output/llm_response_piper.wav
```

## 详细文档

- [Piper TTS详细说明](README_PIPER.md)
- [RKLLM详细说明](README_RKLLM.md)
- [开发者文档](foragent.md)

## 应用场景

- 🏠 智能家居语音控制
- 🤖 机器人语音交互
- 📚 语音问答助手
- 🎓 教育辅助工具
- ♿ 无障碍辅助设备

## 技术特点

### Piper TTS优势

- 速度极快（10-20x实时率）
- 模型小（42MB）
- 音质好（VITS架构）
- ARM优化（专为嵌入式设计）

### RKLLM优势

- NPU加速（RK3588专用）
- 低延迟推理
- 支持量化模型
- 内存占用低

### FunASR优势

- 中文识别准确
- INT8量化支持
- 离线运行
- 阿里达摩院出品

## 许可证

- Piper TTS: MIT License
- FunASR: Apache-2.0
- RKLLM: 参考瑞芯微官方许可

## 致谢

- Rhasspy团队 - Piper TTS
- 阿里巴巴达摩院 - FunASR
- 瑞芯微 - RKLLM SDK
- ModelScope社区

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交Issue或查看详细文档。
