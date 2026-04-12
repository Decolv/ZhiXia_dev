# FunASR + ChatTTS 中文语音处理项目

## 项目简介

本项目实现了完整的中文语音处理流程，包括：
- **ASR (自动语音识别)**: 使用FunASR将语音转换为文本
- **TTS (文本转语音)**: 使用ChatTTS将文本转换为语音
- **完整闭环**: 语音 → 文本 → 语音

## 环境配置

本项目已自动完成以下配置：
- ✅ Python 3.9
- ✅ FunASR 1.3.1 (语音识别)
- ✅ ChatTTS (语音合成)
- ✅ PyTorch 2.8.0 (CPU版本)
- ✅ ModelScope 模型仓库
- ✅ 所有依赖库

## 快速开始

### 1. 语音识别 (ASR)

将音频文件转换为文本：

```bash
bash run_asr.sh
```

**识别结果示例**：
```
识别结果：创建启动器，创建文件夹hello hello.
```

### 2. 语音合成 (TTS)

将文本转换为语音：

```bash
bash run_tts.sh
```

**输出文件**: `output/tts_output.wav`

### 3. 完整流程 (ASR + TTS)

语音 → 文本 → 语音 完整转换：

```bash
bash run_asr_to_tts.sh
```

**流程**：
1. 识别输入音频 `/home/quark/音乐/test.wav`
2. 将识别结果转换为新的语音
3. 输出到 `output/asr_to_tts_output.wav`

## 文件说明

```
/home/quark/code/
├── test_basic.py              # ASR测试脚本
├── test_tts.py                # TTS测试脚本
├── asr_to_tts.py              # ASR+TTS完整流程脚本
├── run_asr.sh                 # ASR启动脚本
├── run_tts.sh                 # TTS启动脚本
├── run_asr_to_tts.sh          # 完整流程启动脚本
├── output/                    # 输出目录
│   ├── tts_output.wav         # TTS输出文件
│   └── asr_to_tts_output.wav  # 完整流程输出文件
├── .local/                    # Python依赖库
└── .cache/                    # 模型缓存
    ├── modelscope/            # ASR模型
    └── chattts/               # TTS模型
```

## 模型信息

### ASR模型 (FunASR)
- **主模型**: `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
  - 参数量：约944MB
  - 支持：中文语音识别
  
- **VAD模型**: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
  - 参数量：约1.64MB
  - 功能：语音活动检测
  
- **标点模型**: `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`
  - 参数量：约278MB
  - 功能：自动添加标点符号

### TTS模型 (ChatTTS)
- **模型**: ChatTTS 开源模型
  - 参数量：约813MB
  - 支持：中文语音合成
  - 特点：自然流畅，支持情感控制

## 使用方法

### 自定义ASR输入

修改 `test_basic.py` 中的音频路径：

```python
audio_path = "/path/to/your/audio.wav"
```

### 自定义TTS文本

修改 `test_tts.py` 中的文本内容：

```python
asr_text = "你想要合成的文本内容"
```

### 批量处理

使用 `test_batch.py` 进行批量ASR识别：

```bash
# 将音频文件放入 ./audio 目录
python3 test_batch.py
```

## 性能优化

### GPU加速

如需GPU加速，安装CUDA版本的PyTorch：

```bash
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

然后在代码中指定设备：

```python
# ASR
device="cuda"

# TTS
chat.load(compile=True)  # GPU模式下可以启用编译
```

### 参数调优

**ASR优化**：
- 调整batch_size
- 使用更轻量的模型
- 关闭标点恢复（移除punc_model参数）

**TTS优化**：
- 调整采样率（默认24000Hz）
- 使用不同的音色
- 调整情感参数

## 常见问题

### 1. 模型下载失败

**解决方案**：
- 检查网络连接
- 使用国内镜像源
- 手动下载模型到缓存目录

### 2. 内存不足

**解决方案**：
- 使用CPU模式
- 减小batch_size
- 关闭模型编译（compile=False）

### 3. 识别速度慢

**解决方案**：
- 使用GPU加速
- 选择轻量级模型
- 调整音频采样率为16kHz

### 4. 音质不佳

**解决方案**：
- 调整TTS采样率
- 尝试不同的音色
- 使用更好的音频输入

## 技术栈

- **Python**: 3.9
- **ASR**: FunASR (阿里巴巴开源)
- **TTS**: ChatTTS (开源中文TTS)
- **深度学习**: PyTorch 2.8.0
- **模型仓库**: ModelScope, HuggingFace
- **音频处理**: torchaudio

## 应用场景

1. **语音助手**: 语音输入 → 文本理解 → 语音回复
2. **有声读物**: 文本 → 语音合成
3. **语音翻译**: 语音 → 文本 → 翻译 → 语音
4. **智能客服**: 自动语音识别与回复
5. **教育辅助**: 语音教材制作

## 后续扩展

1. **实时对话系统**: 流式ASR + 流式TTS
2. **语音克隆**: 使用少量样本克隆特定音色
3. **多语言支持**: 扩展至英语、日语等
4. **WebUI界面**: 图形化操作界面
5. **Docker部署**: 容器化部署方案

## 参考资料

- [FunASR官方文档](https://github.com/alibaba-damo-academy/FunASR)
- [ChatTTS GitHub](https://github.com/2noise/ChatTTS)
- [ModelScope模型仓库](https://modelscope.cn/models)
- [HuggingFace模型仓库](https://huggingface.co/models)

## 许可证

本项目使用的开源模型：
- FunASR: Apache-2.0
- ChatTTS: 开源许可（请查看官方仓库）

## 致谢

感谢以下开源项目：
- 阿里巴巴达摩院 FunASR团队
- ChatTTS开发团队
- ModelScope社区
