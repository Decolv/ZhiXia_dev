# ZhiXia 语音助手 — 使用说明

## 快速开始

### 1. 配置输入音频

编辑 `localconfig/localconfig.json`，设置输入音频路径：

```json
{
  "asr": {
    "input_audio": "/path/to/your/audio.wav"
  }
}
```

### 2. 运行

```bash
python -m zhixia
```

或使用兼容入口：

```bash
python asr_llm_tts_piper.py
```

### 3. 输出

- 识别文本 + AI 回复 + 情绪标签（可选）
- 语音播放
- 输出音频保存到 `output/` 目录

---

## 配置选项

### ASR（语音识别）

```json
{
  "asr": {
    "engine": "funasr",           // 或 "whisper"
    "model": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1",
    "input_audio": "/path/to/audio.wav"
  }
}
```

### LLM（大语言模型）

```json
{
  "llm": {
    "model_path": "models/Qwen3-1.7B-w8a8-rk3588.rkllm",
    "max_new_tokens": 32,
    "temperature": 0.8,
    "top_p": 0.95,
    "system_prompt": "你是AI助手，用一句话简短回答。/no_think",
    "enable_structured_output": false  // 改为 true 启用 emotion 识别
  }
}
```

### TTS（语音合成）

```json
{
  "tts": {
    "engine": "piper",
    "model_path": "models/piper/zh_CN-huayan-medium.onnx"
  }
}
```

### 日志级别

```json
{
  "log_level": "INFO"  // 改为 "DEBUG" 查看详细日志
}
```

---

## Debug 方式

### 方式 1：启用 DEBUG 日志

编辑 `localconfig/localconfig.json`：

```json
{
  "log_level": "DEBUG"
}
```

运行时会输出：
- 每个 token 的流式输出
- TTS 合成耗时
- 模型加载信息
- 内存使用情况

### 方式 2：逐步测试各模块

**只测试 ASR：**

```python
from zhixia.asr.funasr_engine import FunASREngine
from zhixia.config.settings import ASRConfig
from pathlib import Path

config = ASRConfig(input_audio="/path/to/audio.wav")
asr = FunASREngine(config, Path("."))
result = asr.transcribe(Path("/path/to/audio.wav"))
print(f"识别结果: {result.text}")
```

**只测试 LLM：**

```python
from zhixia.llm.rkllm_engine import RKLLMEngine
from zhixia.llm.base import LLMMessage
from zhixia.config.settings import LLMConfig

config = LLMConfig(model_path="models/Qwen3-1.7B-w8a8-rk3588.rkllm")
llm = RKLLMEngine(config)
messages = [LLMMessage(role="user", content="你好")]
response = llm.chat(messages, max_new_tokens=32)
print(f"LLM 回复: {response}")
```

**只测试 TTS：**

```python
from zhixia.tts.piper_engine import PiperTTSEngine
from pathlib import Path

config_obj = type('obj', (object,), {'model_path': 'models/piper/zh_CN-huayan-medium.onnx'})()
tts = PiperTTSEngine(config_obj, Path("."))
wav = tts.synthesize_to_bytes("你好，世界")
print(f"合成音频: {len(wav)} bytes")
```

### 方式 3：查看流水线详情

运行时观察输出：

```
🎙️ 语音助手处理开始（流水线模式）
[1/3] 语音识别中...
✅ 识别文本: '你好'  (0.45s)

[2/3] AI 思考 + 语音合成（流水线）...
🔊 首句播放开始
✅ AI 完整回复: '你好，很高兴见到你'

⏱️  总耗时: 2.34s
```

关键指标：
- **识别耗时**：ASR 性能
- **首句播放**：首字延迟（越短越好）
- **总耗时**：端到端延迟

### 方式 4：检查模型文件

```bash
# 检查 ASR 模型
ls -lh .cache/modelscope/

# 检查 LLM 模型
ls -lh models/Qwen3-1.7B-w8a8-rk3588.rkllm

# 检查 TTS 模型
ls -lh models/piper/zh_CN-huayan-medium.onnx*
```

### 方式 5：查看完整日志

```bash
# 运行并保存日志
python -m zhixia 2>&1 | tee debug.log

# 查看特定模块的日志
grep "LLM\|TTS\|ASR" debug.log
```

---

## 常见问题

### Q: 首字延迟太长？

A: 检查以下几点：
1. 模型是否已预热（第一次运行会加载模型）
2. 日志中 "首句播放开始" 的时间点
3. 是否启用了 thinking 模式（system_prompt 加 `/no_think` 关闭）

### Q: TTS 合成失败？

A: 检查模型文件：
```bash
ls -lh models/piper/zh_CN-huayan-medium.onnx*
```

如果缺少，会自动从 HuggingFace 下载。

### Q: 内存占用过高？

A: 编辑 `localconfig.json`：
```json
{
  "device": {
    "memory_optimization": true
  }
}
```

### Q: 想要 emotion 识别？

A: 启用结构化输出：
```json
{
  "llm": {
    "enable_structured_output": true
  }
}
```

emotion 会在 LLM 输出时立即显示。

---

## 性能指标

| 模块 | 耗时 | 备注 |
|------|------|------|
| ASR | 0.3-0.5s | 取决于音频长度 |
| LLM 首句 | 0.2-0.4s | 流式输出 |
| TTS 首句 | 0.1-0.2s | 内存合成 |
| **首字延迟** | **0.6-1.1s** | ASR + LLM首句 + TTS首句 |
| 总耗时 | 1.5-3.0s | 取决于回复长度 |

---

## 架构

```
输入音频
   ↓
[ASR] 语音识别 → 文本
   ↓
[LLM] 流式推理 → token 流
   ↓
   ├─→ [分句] → 完整句子
   │      ↓
   │   [TTS] 内存合成 → WAV
   │      ↓
   │   [播放] 边合成边播放
   │
   └─→ [解析] 完整 JSON → emotion + metadata
          ↓
       [显示] 更新 UI
```

三线程并发：LLM 流式 → TTS 合成 → 播放，实现首字延迟最小化。
