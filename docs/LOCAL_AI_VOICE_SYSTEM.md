# 本地完整 AI 对话系统规划（STT + LLM + TTS）

## 项目概述

创建一个 **100% 离线、本地 NPU 驱动的语音对话系统**，无需云服务、无隐私泄露、无网络依赖。

核心流程：
```
🎤 麦克风 → 本地 Whisper → 文本
   ↓
🧠 本地 NPU (Ollama/vLLM/GGML) → 回复文本
   ↓
🔊 本地 TTS (Kokoro/FastPitch) + 切片流式 → 音频输出
```

---

## 一、技术栈选择规划

### 1.1 STT (语音识别)

#### 方案对比：

| 方案 | 优点 | 缺点 | 适配度 |
|------|------|------|--------|
| **Whisper (OpenAI)** | 准确率高、多语言支持、模型开源 | 需要 GPU/CPU，慢速 | ✅ **推荐** |
| **Whisper.cpp** | C++ 优化，速度快、支持 NPU | 需要编译、依赖复杂 | ⭐ **次优** |
| **FastWhisper** | 比官方快 4 倍 | 内存占用大 | ✅ **推荐** |
| **Silero VAD + Wav2Vec** | 轻量级组合 | 准确度不如 Whisper | ⚠️ 降级方案 |
| **Paraformer (阿里)** | 中文优化 | 模型小，准确度一般 | 🇨🇳 中文优选 |

#### 推荐方案：
```
主方案：Whisper (faster-whisper 库)
  ├─ 模型大小：base (~140MB) / small (~500MB)
  ├─ 速度：base 2-3x实时，CPU 可用
  └─ 多语言：中文/英文均可
```

#### 实现位置：
```python
# src/services/stt_service.py
from faster_whisper import WhisperModel

class LocalSTTService:
    def __init__(self, model_name="base"):
        # 初始化本地 Whisper 模型
        self.model = WhisperModel(
            model_name,
            device="auto",  # 自动选择 CUDA/CPU/NPU
            compute_type="int8"  # 量化以节省显存
        )
    
    async def transcribe(self, audio_path: str) -> str:
        segments, info = self.model.transcribe(audio_path, language="zh")
        return "".join([segment.text for segment in segments])
```

### 1.2 LLM (大语言模型)

#### 本地 LLM 推理框架：

| 框架 | 特点 | NPU 支持 | 推荐度 |
|------|------|---------|--------|
| **Ollama** | 傻瓜式部署，模型多 | ❌ 无（仅 CPU/GPU） | ✅ 简单 |
| **vLLM** | 高性能，生产级 | ✅ 部分支持 | ⭐ **推荐** |
| **GGML/llama.cpp** | 内存高效，支持量化 | ✅ NPU/CPU | ⭐ **推荐** |
| **TVM** | 通用优化编译 | ✅ 强力支持 | 📊 学习曲线陡 |
| **ONNCharAt** | 微软方案，通用 | ✅ NPU 友好 | 📊 需要转换 |
| **高通 AI Engine** | 移动端 NPU (骁龙) | ✅ 专属 | 📱 移动限定 |
| **英特尔 OpenVINO** | x86 优化 | ✅ 核显/集显 | 💻 PC 优选 |

#### 推荐方案组合：

##### 🎯 **首选：vLLM + GGML 混合**
```
# PC/服务器环境（RTX/RTX/NPU）
vLLM (快速推理)
  ├─ 模型：Llama-2-7B / Mistral-7B / Qwen-7B
  ├─ 量化：AWQ / GPTQ (节省显存)
  └─ 速度：100+ tokens/sec (GPU)

# 轻量化方案（CPU/集显）
GGML (llama.cpp)
  ├─ 模型：Llama-2-7B-GGML / Phi-2-GGML
  ├─ 量化：Q4_K_M (4-bit)
  └─ 速度：10-30 tokens/sec (CPU)
```

##### 🇨🇳 **中文优化**：
```
Qwen-7B (阿里)
  ├─ 中文性能最佳
  ├─ 支持 vLLM 和 GGML
  └─ 量化友好

或 Llama-2-Chinese-7B
  ├─ 微调对齐
  └─ LLaMA 生态支持
```

#### 实现位置：
```python
# src/services/llm_service.py
from vllm import LLM, SamplingParams

class LocalLLMService:
    def __init__(self, model_path: str):
        # vLLM 推理后端
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # 单张卡
            dtype="float16",  # 半精度
            gpu_memory_utilization=0.8
        )
    
    async def chat(self, prompt: str) -> str:
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

# GGML 备选
# from llama_cpp import Llama
# self.llm = Llama(model_path, n_gpu_layers=-1)  # -1 转移全部到 GPU
```

### 1.3 TTS (文本转语音)

#### 本地 TTS 方案：

| 方案 | 模型大小 | 质量 | 速度 | NPU 支持 | 推荐度 |
|------|---------|------|------|---------|--------|
| **Kokoro** | ~100MB | 中等 | 快 | ⭐ 原生 | ✅ **推荐** |
| **FastPitch** | ~30MB | 中等 | 快 | ⭐ 转换中 | ✅ **推荐** |
| **Tacotron2** | ~100MB | 高 | 慢 | 🔄 可能 | ⚠️ 次选 |
| **VITS** | ~50MB | 高 | 中等 | ✅ 支持 | ✅ **推荐** |
| **Glow-TTS** | ~40MB | 中等 | 快 | ✅ 支持 | ✅ **推荐** |
| **Edge TTS** | 云端 | 高 | 快 | ❌ 需网络 | ❌ 弃用 |
| **Coqui TTS** | 40-200MB | 中-高 | 中等 | ⭐ 支持 | ✅ **推荐** |

#### 推荐方案：**Kokoro + Vocoder**

##### 🎯 **首选组合**：
```
FastPitch/Glow-TTS (文本→梅尔谱)
  ├─ 快速生成梅尔频谱
  ├─ 模型轻量（40-100MB ONNX）
  └─ 易于量化到 NPU
     ↓
HiFi-GAN (梅尔谱→音频)
  ├─ 高保真 Vocoder
  ├─ 实时推理（<100ms per chunk）
  └─ ONNX 原生支持
```

#### 采用方案对比：

```python
# 方案 A：VITS (单模型端到端)
from TTS.api import TTS
tts = TTS(model_name="tts_models/en/ljspeech/vits", gpu=True)
text2wav = tts.tts(text)

# 方案 B：FastPitch + GlowTTS (可定制化)
from fastpitch import FastPitch
from hifigan import HifiGAN

fastpitch = FastPitch.load_from_checkpoint(...)
hifigan = HifiGAN.load_from_checkpoint(...)

mel = fastpitch.forward(text_tokens)
wav = hifigan(mel)

# 方案 C：Coqui TTS (社区活跃)
from TTS.tts.layers.xtts.xtts import Xtts
model = Xtts()
model.load_checkpoint(...)
gpt_cond_latent, speaker_embedding = model.speaker_encoder(...)
```

#### 推荐选择（综合考虑）：
```
✅ 首选：FastPitch + HiFi-GAN
   - 性能平衡好
   - ONNX 模型质量稳定
   - 易于转移到 NPU
   
✅ 次选：Coqui XTTS
   - 多语言能力强
   - 声纹克隆选项
   - 社区支持好

⭐ 本地 Kokoro (Web 已有)
   - 直接复用 AIRI 的 WASM 模型
   - 快速验证原型
```

#### 实现位置：
```python
# src/services/tts_service.py
import torch
from model import FastPitch, HiFiGAN

class LocalTTSService:
    def __init__(self, fastpitch_path: str, hifigan_path: str):
        self.fastpitch = FastPitch.load_from_checkpoint(fastpitch_path)
        self.hifigan = HiFiGAN.load_from_checkpoint(hifigan_path)
        
        self.fastpitch.eval()
        self.hifigan.eval()
    
    async def synthesize(self, text: str) -> np.ndarray:
        # 文本 → token
        tokens = encode_text(text)
        
        # FastPitch: token → mel spectrogram
        with torch.no_grad():
            mel = self.fastpitch(tokens)
        
        # HiFi-GAN: mel → 音频
        with torch.no_grad():
            audio = self.hifigan(mel)
        
        return audio.cpu().numpy()

    # 分块流式版本
    async def synthesize_streaming(self, text_chunks: AsyncIterator[str]):
        """
        流式处理：
        "你好" → [audio_chunk_1]
        "我是" → [audio_chunk_2]
        "AIRI" → [audio_chunk_3]
        """
        for chunk in text_chunks:
            audio = await self.synthesize(chunk)
            yield audio
```

---

## 二、架构设计

### 2.1 系统架构图

```
┌────────────────────────────────────────────────────────────────┐
│                       用户交互层                                │
├────────────────────────────────────────────────────────────────┤
│
│  桌面应用 (Electron/PyQt)  或  Web 应用 (Streamlit/Gradio)
│
├────────────────────────────────────────────────────────────────┤
│                     核心业务逻辑层                              │
├────────────────────────────────────────────────────────────────┤
│
│  ┌──────────────────────────────────────────────────────────┐
│  │           Voice Interaction Pipeline                        │
│  ├──────────────────────────────────────────────────────────┤
│  │
│  │  [1️⃣ 录音管理]  →  [2️⃣ VAD检测]  →  [3️⃣ STT转文本]
│  │       ↓                ↓                  ↓
│  │   Pyaudio        Silero VAD        Whisper (CPU)
│  │   (async)         (async)          faster-whisper
│  │
│  │  ┌────────────────────────────────────────┐
│  │  │    Chat Queue (消息编排)                 │
│  │  │    - 输入文本存储                        │
│  │  │    - 上下文管理                         │
│  │  │    - 消息去重                           │
│  │  └────────────────────────────────────────┘
│  │
│  │  [4️⃣ LLM推理]  →  [5️⃣ TTS合成]  →  [6️⃣ 音频播放]
│  │       ↓                ↓                  ↓
│  │   vLLM/GGML      FastPitch+HiFi-GAN    PyAudio
│  │   (NPU/GPU/CPU)   (Async)            (Stream)
│  │
│  └──────────────────────────────────────────────────────────┘
│
├────────────────────────────────────────────────────────────────┤
│                     硬件加速层                                  │
├────────────────────────────────────────────────────────────────┤
│
│  NPU 选项：
│  ├─ 高通骁龙 (移动)
│  ├─ 英特尔 CoreUltra (核显 / 独显)
│  ├─ RTX/RTX (NVIDIA CUDA)
│  ├─ MacBook (Apple Silicon)
│  └─ 树莓派 (ARM CPU)
│
└────────────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

```
project/
├── src/
│   ├── services/
│   │   ├── stt_service.py       # Whisper 本地推理
│   │   ├── llm_service.py       # vLLM/GGML 推理
│   │   ├── tts_service.py       # FastPitch + HiFi-GAN
│   │   ├── vad_service.py       # Silero VAD 检测
│   │   └── audio_device.py      # PyAudio 设备管理
│   │
│   ├── pipeline/
│   │   ├── voice_pipeline.py    # 完整语音流程调度
│   │   ├── chat_orchestrator.py # 消息编排与上下文
│   │   ├── intent_manager.py    # 意图队列 (类 AIRI)
│   │   └── event_bus.py         # 事件驱动架构
│   │
│   ├── models/
│   │   ├── chat_context.py      # 对话历史、系统提示
│   │   └── voice_config.py      # 模型配置、路径
│   │
│   ├── utils/
│   │   ├── text_chunker.py      # TTS 文本分块（AIRI 风格）
│   │   ├── audio_processor.py   # 音频处理工具
│   │   └── logger.py            # 日志系统
│   │
│   └── ui/
│       ├── webui/               # Streamlit/Gradio Web UI
│       ├── desktop/             # PyQt/Electron 桌面应用
│       └── cli.py               # 命令行版本
│
├── models/                      # 本地模型存储
│   ├── stt/
│   │   └── whisper-base.pt
│   ├── llm/
│   │   ├── Qwen-7B-Chat-GGML/
│   │   └── Llama-2-7b-hf/
│   └── tts/
│       ├── fastpitch.pt
│       └── hifigan.pt
│
├── tests/
├── docker/                      # Docker 打包
├── requirements.txt
├── config.yaml                  # 统一配置
└── main.py                      # 应用入口
```

### 2.3 组件通信架构

#### 事件驱动 + 消息队列

```python
# src/pipeline/event_bus.py
from asyncio import Queue
from enum import Enum

class EventType(Enum):
    AUDIO_START = "audio:start"
    SPEECH_DETECTED = "speech:detected"
    SPEECH_READY = "speech:ready"
    TRANSCRIPTION_COMPLETE = "stt:complete"
    LLM_START = "llm:start"
    LLM_TOKEN = "llm:token"
    LLM_COMPLETE = "llm:complete"
    TTS_START = "tts:start"
    AUDIO_CHUNK = "audio:chunk"
    AUDIO_COMPLETE = "audio:complete"


class EventBus:
    """跨组件事件广播（类似 AIRI 的 Eventa）"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
    
    async def emit(self, event_type: EventType, data: Any):
        if event_type not in self.subscribers:
            return
        
        tasks = [
            handler(data) 
            for handler in self.subscribers[event_type]
        ]
        await asyncio.gather(*tasks)
    
    def on(self, event_type: EventType, handler: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)


# src/pipeline/intent_manager.py
class VoiceIntent:
    """单个语音交互对象（类似 AIRI 的 Intent）"""
    
    def __init__(self, intent_id: str):
        self.intent_id = intent_id
        self.stream_id = f"stream-{intent_id}"
        self.text_buffer = ""
        self.audio_buffer = []
        self.priority = "normal"
        self.created_at = time.time()


class IntentManager:
    """管理多个并发 Intent"""
    
    def __init__(self, event_bus: EventBus):
        self.intents: Dict[str, VoiceIntent] = {}
        self.queue: asyncio.PriorityQueue[VoiceIntent] = asyncio.PriorityQueue()
        self.event_bus = event_bus
    
    async def create_intent(self) -> VoiceIntent:
        intent = VoiceIntent(intent_id=nanoid())
        self.intents[intent.intent_id] = intent
        return intent
    
    async def write_token(self, intent_id: str, token: str):
        """添加文本令牌（流式）"""
        intent = self.intents[intent_id]
        intent.text_buffer += token
        await self.event_bus.emit(
            EventType.LLM_TOKEN,
            {"intent_id": intent_id, "token": token}
        )
```

---

## 三、完整数据流

### 3.1 请求-响应周期

```
┌─────────────────────────────────────────────────────────────────┐
│ 用户说话：" 你好，今天天气怎么样？"                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ [步骤 1️⃣] 音频捕获 & VAD 检测                                  │
├─────────────────────────────────────────────────────────────────┤
│
│  PyAudio → 音频流（16kHz, 16-bit PCM）
│      ↓
│  Silero VAD 检测 speech-start
│      ↓
│  缓冲帧数据，直到 speech-end
│      ↓
│  EmitEvent: SPEECH_READY { audio_buffer: [...] }
│
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ [步骤 2️⃣] STT 转录 (Whisper)                                  │
├─────────────────────────────────────────────────────────────────┤
│
│  STT Service 接收到 audio_buffer
│      ↓
│  faster-whisper.transcribe(audio)
│    • 处理时间：0.5-2s（取决于模型和 CPU/GPU）
│    • 输出："你好，今天天气怎么样？"
│      ↓
│  EmitEvent: TRANSCRIPTION_COMPLETE { text: "..." }
│
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ [步骤 3️⃣] 创建 Intent 并入队                                  │
├─────────────────────────────────────────────────────────────────┤
│
│  Chat Orchestrator 接收文本
│      ↓
│  创建 Intent:
│  {
│    intent_id: "intent-abc123",
│    stream_id: "stream-abc123",
│    text_buffer: "用户：你好，今天天气怎么样？\n",
│    priority: "normal"
│  }
│      ↓
│  加入消息队列（通常只有 1 个消息等待处理）
│      ↓
│  EmitEvent: LLM_START { intent_id: "..." }
│
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ [步骤 4️⃣] LLM 推理 (vLLM/GGML)                                │
├─────────────────────────────────────────────────────────────────┤
│
│  LLM Service 接收 Intent
│      ↓
│  构建 Prompt:
│    • 系统提示：[系统角色描述]
│    • 历史对话：[上下文]
│    • 当前输入：用户：你好，今天天气怎么样？
│      ↓
│  vLLM.generate(prompt, stream=True)
│    • 流式生成令牌：
│      token_1: "今"
│      token_2: "天"
│      token_3: "天"
│      token_4: "气"
│      ...
│    • 每个令牌回调：
│      write_token(intent_id, token)
│      ↓
│    EmitEvent: LLM_TOKEN { token: "...", token_id: 1 }
│      ↓
│  LLM 完成：text_buffer = "今天天气晴朗，气温 25℃"
│      ↓
│  EmitEvent: LLM_COMPLETE { text: "..." }
│
│  推理时间：500ms-5s（取决于 K、显存、模型大小）
│
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ [步骤 5️⃣] 文本分块 & TTS 流水线                               │
├─────────────────────────────────────────────────────────────────┤
│
│  TTS Service 接收完整文本
│      ↓
│  TextChunker.chunk_for_tts(text):
│    硬分块规则（类似 AIRI）：
│    - 按句号、问号、感叹号分割
│    - 保留逗号、顿号
│    
│    输入：  "今天天气晴朗，气温 25℃。明天可能有雨。"
│    输出：
│      Chunk 1: "今天天气晴朗，气温 25℃。"
│      Chunk 2: "明天可能有雨。"
│      ↓
│  并行 TTS 请求：
│    ┌─────────────────────────────────┐
│    │ 任务队列                        │
│    ├─────────────────────────────────┤
│    │ [1] Chunk 1 → TTS (FastPitch+HiFi-GAN)
│    │     ↓ 处理时间：300ms
│    │     ✓ 生成音频块 (pcm_chunk_1)
│    │
│    │ [2] Chunk 2 → TTS (FastPitch+HiFi-GAN)
│    │     ↓ 处理时间：200ms
│    │     ✓ 生成音频块 (pcm_chunk_2)
│    └─────────────────────────────────┘
│      ↓
│  EmitEvent: AUDIO_CHUNK { 
│    intent_id: "...",
│    chunk_id: 1,
│    audio_buffer: pcm_chunk_1,
│    duration_ms: 1500
│  }
│      ↓
│  EmitEvent: AUDIO_CHUNK { 
│    intent_id: "...",
│    chunk_id: 2,
│    audio_buffer: pcm_chunk_2,
│    duration_ms: 800
│  }
│
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ [步骤 6️⃣] 实时音频播放                                       │
├─────────────────────────────────────────────────────────────────┤
│
│  Audio Player（消费队列）
│      ↓
│  接收 AUDIO_CHUNK 事件
│      ↓
│  若不在播放中，立即启动播放
│  ┌──────────────────────┐
│  │ ♪ ♪ ♪ ♪ ♪ ♪ ♪ ♪  │ （实时 PCM 播放）
│  │ 今天天气晴朗...      │
│  └──────────────────────┘
│      ↓
│  等待音频播放完成
│      ↓
│  EmitEvent: AUDIO_COMPLETE { intent_id: "..." }
│
│  用户听到：【自然语音流式回复】
│    "今天天气晴朗，气...    （LLM 尚在生成）
│     （空隙补充）温 25℃。   （LLM 继续生成）
│     明天...可能有雨。"     （首个分块完成）
│
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 时间轴对比：同步 vs 异步流式

```
【同步模式（传统）】
────────────────────────────────────────────────────────
用户说话：        [...]  ← 用户等待
Whisper 处理：          [===========]
LLM 推理：                           [================]
TTS 合成：                                          [==]
播放音频：                                               [====]
────────────────────────────────────────────────────────
总延迟：        ~5-8 秒  ❌ 太慢


【异步流式模式（本项目）】
────────────────────────────────────────────────────────
用户说话：          [...]
               ↓
Whisper 处理：      [===]
              ↓ （交互开始）
LLM 推理：          [==token==token==token...]
              ↓ （分块准备）
TTS 分块队列：        [chunk1] [chunk2] [chunk3]
              ↓       ↓       ↓
TTS 异步合成：      [====]  [====]  [====]
              ↓       ↓       ↓
播放音频：            [====][====][====]  ♪♪♪ （无缝播放）
────────────────────────────────────────────────────────
总延迟：        ~1-2 秒  ✅ 接近实时
用户体验：      自然流畅、不间断
```

### 3.3 核心实现代码片段

```python
# src/pipeline/voice_pipeline.py
import asyncio
from typing import AsyncIterator

class VoicePipeline:
    """完整的语音对话流水线"""
    
    def __init__(
        self,
        stt_service,
        llm_service,
        tts_service,
        vad_service,
        audio_device,
        event_bus
    ):
        self.stt = stt_service
        self.llm = llm_service
        self.tts = tts_service
        self.vad = vad_service
        self.audio = audio_device
        self.event_bus = event_bus
        
        self.intent_manager = IntentManager(event_bus)
        self.text_chunker = TextChunker()
        self.chat_history = []
    
    async def run_voice_interaction(self):
        """主循环：连续监听音频并处理"""
        
        print("🎙️  开始监听...")
        await self.event_bus.emit(EventType.AUDIO_START, {})
        
        async for audio_chunk in self.audio.listen_stream():
            # 1️⃣ VAD 检测
            is_speech = await self.vad.detect(audio_chunk)
            
            if is_speech:
                # 缓冲并等待 speech-end
                audio_buffer = await self.vad.get_speech_segment()
                
                # 2️⃣ STT 转录
                await self.event_bus.emit(EventType.SPEECH_READY, {
                    "audio": audio_buffer
                })
                
                text = await self.stt.transcribe(audio_buffer)
                print(f"👤 用户：{text}")
                
                # 3️⃣ 创建 Intent
                intent = await self.intent_manager.create_intent()
                
                # 4️⃣ LLM 推理（流式）
                await self._process_llm_stream(intent, text)
    
    async def _process_llm_stream(self, intent: VoiceIntent, user_text: str):
        """LLM 流式处理"""
        
        await self.event_bus.emit(EventType.LLM_START, {
            "intent_id": intent.intent_id
        })
        
        # 构建提示
        prompt = self._build_prompt(user_text)
        
        # 流式生成
        async for token in self.llm.stream_chat(prompt):
            # 写入 token
            await self.intent_manager.write_token(intent.intent_id, token)
            
            # 当缓冲足够时触发 TTS
            if self._should_trigger_tts(intent.text_buffer):
                await self._trigger_tts_stream(intent)
        
        # 处理剩余文本
        if intent.text_buffer:
            await self._trigger_tts_stream(intent)
        
        await self.event_bus.emit(EventType.LLM_COMPLETE, {
            "intent_id": intent.intent_id,
            "text": intent.text_buffer
        })
    
    async def _trigger_tts_stream(self, intent: VoiceIntent):
        """触发 TTS 流式处理"""
        
        text_to_speak = intent.text_buffer
        intent.text_buffer = ""  # 清空缓冲
        
        # 分块
        chunks = self.text_chunker.chunk(text_to_speak)
        
        # 异步 TTS 合成任务
        async def synthesize_chunk(chunk_id: int, text: str):
            try:
                audio = await self.tts.synthesize(text)
                
                await self.event_bus.emit(EventType.AUDIO_CHUNK, {
                    "intent_id": intent.intent_id,
                    "chunk_id": chunk_id,
                    "audio": audio,
                    "text": text
                })
            except Exception as e:
                print(f"❌ TTS 错误: {e}")
        
        # 并行启动 TTS 任务（无需等待完成）
        tasks = [
            synthesize_chunk(i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        asyncio.create_task(asyncio.gather(*tasks))

    def _should_trigger_tts(self, text_buffer: str) -> bool:
        """判断是否应该触发 TTS（流式决策）"""
        # 规则：
        # 1. 收集到 N 个词
        # 2. 遇到标点符号
        # 3. 缓冲时间超过 T 毫秒
        
        word_count = len(text_buffer.split())
        has_punctuation = any(p in text_buffer for p in '。！，？')
        
        return word_count >= 10 or (has_punctuation and word_count >= 3)
```

---

## 四、关键实现细节

### 4.1 文本分块算法（TTS 优化）

```python
# src/utils/text_chunker.py
import re
from typing import List

class TextChunker:
    """智能文本分块，平衡实时性和完整性"""
    
    # 分块规则配置
    HARD_PUNCTUATION = set('。！？…；：')     # 硬分块
    SOFT_PUNCTUATION = set('，、–—')         # 软分块
    MIN_CHUNK_CHARS = 10
    MAX_CHUNK_CHARS = 100
    RESERVE_PUNCTUATION = set('?？!！')      # 保留符号
    
    @staticmethod
    def chunk(text: str, boost: int = 2, min_words: int = 3, max_words: int = 15) -> List[str]:
        """
        基于词数和分页符的智能分块
        
        参数：
            text: 要分块的文本
            boost: 分块权重调整（1-5）
            min_words: 最小词数
            max_words: 最大词数
        
        返回：
            分块列表
        
        示例：
            输入：  "你好，我是 AI 助手。今天是星期一，天气很好。"
            输出：
              ["你好，我是 AI 助手。",
               "今天是星期一，天气很好。"]
        """
        
        chunks = []
        current_chunk = ""
        
        # 按字符迭代
        i = 0
        while i < len(text):
            char = text[i]
            current_chunk += char
            
            # 检查硬分块符号
            if char in TextChunker.HARD_PUNCTUATION:
                # 直接切割
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # 检查软分块符号
            elif char in TextChunker.SOFT_PUNCTUATION:
                # 累积，直到后续硬符号
                pass
            
            # 超长自动分块
            elif len(current_chunk) >= TextChunker.MAX_CHUNK_CHARS:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = ""
            
            i += 1
        
        # 处理剩余
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def chunk_streaming(text_generator: AsyncIterator[str]) -> AsyncIterator[str]:
        """
        从令牌流中实时分块
        
        用途：LLM 流式生成 token 时，实时提取完整分块
        
        示例：
            LLM: "你" → "好" → "，" → "天" → "气" → "晴" → "。" → "明" → "天" → "..."
                                                              ↓
                                          yield "你好，天气晴。"
        """
        buffer = ""
        
        async for token in text_generator:
            buffer += token
            
            # 检查是否有完整分块
            if any(p in buffer for p in TextChunker.HARD_PUNCTUATION):
                # 提取最后一个完整句子
                parts = re.split(f"([{''.join(TextChunker.HARD_PUNCTUATION)}])", buffer)
                
                for i in range(0, len(parts) - 2, 2):
                    chunk = (parts[i] + parts[i+1]).strip()
                    if chunk:
                        yield chunk
                
                # 保留未完成的部分
                buffer = parts[-1]
        
        # 返回剩余缓冲
        if buffer.strip():
            yield buffer.strip()
```

### 4.2 NPU 推理加速

#### 4.2.1 模型量化策略

```python
# src/services/model_quantizer.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

class ModelQuantizer:
    """模型量化与加速"""
    
    @staticmethod
    def quantize_to_ggml(model_path: str, output_path: str, bits: int = 4):
        """转换为 GGML 格式（支持 CPU + NPU）"""
        # 使用 ctransformers 库
        from ctransformers import convert_ggml_llama_cpp_to_gguf
        
        convert_ggml_llama_cpp_to_gguf(
            model_path=model_path,
            output_path=output_path,
            quantization_bits=bits  # 4, 5, 8
        )
    
    @staticmethod
    def quantize_to_int8(model_path: str):
        """8-bit 量化（vLLM 使用）"""
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=True,  # bitsandbytes
            trust_remote_code=True
        )
    
    @staticmethod
    def quantize_to_awq(model_path: str):
        """AWQ 量化（高效推理）"""
        from awq import AutoAWQForCausalLM
        
        return AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True,
            safetensors=True
        )


# 选择策略
config = {
    "device": "cuda",  # 或 "cpu"
    "quantization": "awq",  # "int8", "gptq", "awq"
    "model_size": "7b",
    "batch_size": 1
}
```

#### 4.2.2 NPU 特定优化

```python
# src/services/npu_acceleration.py

class NPUAccelerator:
    """NPU 特定的推理加速"""
    
    @staticmethod
    def detect_device():
        """自动检测可用硬件"""
        if torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        
        try:
            import torch_npu
            return "npu"  # 华为昇腾
        except ImportError:
            pass
        
        try:
            import torch_xpu
            return "xpu"  # Intel GPU
        except ImportError:
            pass
        
        try:
            # 高通骁龙（需要 Qualcomm AI Engine）
            from qti.aisw.quantization_tensorflow import quantize
            return "qualcomm_npu"
        except ImportError:
            pass
        
        return "cpu"  # 降级到 CPU
    
    @staticmethod
    def optimize_for_device(model, device: str):
        """针对不同 NPU 优化"""
        
        if device == "cuda":
            # NVIDIA
            model = model.half()  # 混精度
            model.to("cuda")
        
        elif device == "npu":
            # 华为昇腾
            import torch_npu
            model.to("npu")
            # compile 优化
            model = torch.jit.script(model)
        
        elif device == "xpu":
            # Intel
            import torch_xpu
            model.to("xpu")
        
        elif device == "qualcomm_npu":
            # 高通骁龙（移动）
            # 需要转换为 SNPE / QNN
            pass
        
        return model
```

### 4.3 并发与队列管理

```python
# src/pipeline/async_queue_manager.py
import asyncio
from asyncio import Queue, PriorityQueue
from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: str  # "tts", "llm", "playback"
    data: dict
    priority: int = 0  # 0=高, 越大越低
    created_at: float = 0
    
    def __lt__(self, other):
        """优先级队列排序"""
        return (self.priority, self.created_at) < (other.priority, other.created_at)


class TaskManager:
    """异步任务管理器"""
    
    def __init__(self, max_concurrent_tasks: int = 3):
        self.task_queue: PriorityQueue[Task] = PriorityQueue()
        self.running_tasks = {}
        self.max_concurrent = max_concurrent_tasks
    
    async def submit_task(self, task: Task):
        """提交任务"""
        await self.task_queue.put(task)
        await self._process_queue()
    
    async def _process_queue(self):
        """处理队列中的任务"""
        while not self.task_queue.empty() and len(self.running_tasks) < self.max_concurrent:
            task = await self.task_queue.get()
            
            # 创建后台任务
            coroutine = self._execute_task(task)
            asyncio.create_task(coroutine)
    
    async def _execute_task(self, task: Task):
        """执行单个任务"""
        self.running_tasks[task.task_id] = True
        
        try:
            if task.task_type == "tts":
                await self._handle_tts_task(task)
            elif task.task_type == "playback":
                await self._handle_playback_task(task)
            # ... 其他任务类型
        finally:
            del self.running_tasks[task.task_id]
            await self._process_queue()  # 处理下一个


# 使用示例
task_manager = TaskManager(max_concurrent_tasks=3)

# LLM 完成后，并行提交多个 TTS 任务
for chunk_id, text in enumerate(chunks):
    task = Task(
        task_id=f"tts-{chunk_id}",
        task_type="tts",
        data={"text": text, "intent_id": intent_id},
        priority=0  # 所有 TTS 任务相同优先级
    )
    await task_manager.submit_task(task)
```

---

## 五、实现步骤与里程碑

### 5.1 阶段规划

#### **第一阶段：核心框架（2-3周）**
- [ ] 创建项目结构和配置系统
- [ ] 集成 faster-whisper（STT）
- [ ] 集成 vLLM / GGML（LLM）
- [ ] 集成 FastPitch + HiFi-GAN（TTS）
- [ ] 基础 CLI 测试集成

**交付物**：离线可用的命令行版本

#### **第二阶段：流式架构（2周）**
- [ ] 实现事件总线（EventBus）
- [ ] 实现 Intent 管理器
- [ ] 实现文本分块流式处理
- [ ] 异步队列与并发管理
- [ ] 测试端到端流水线

**交付物**：流式语音对话原型

#### **第三阶段：UI & 优化（2-3周）**
- [ ] Streamlit/Gradio Web UI
- [ ] 桌面 UI（PyQt5 或 Electron + Python 后端）
- [ ] 模型量化与性能优化
- [ ] 日志与监控系统
- [ ] 错误处理与重试机制

**交付物**：可用的生产原型

#### **第四阶段：部署与文档（1-2周）**
- [ ] Docker 容器化
- [ ] 模型预下载脚本
- [ ] 用户文档与配置指南
- [ ] 性能基准测试
- [ ] 开源发布

**交付物**：完整开源项目

### 5.2 关键文件创建清单

```
src/
✅ __init__.py
✅ config.py                    # 统一配置管理
✅ logger.py                    # 日志系统

services/
✅ stt_service.py               # Whisper 集成
✅ llm_service.py               # vLLM/GGML 集成
✅ tts_service.py               # FastPitch+HiFi-GAN
✅ vad_service.py               # Silero VAD
✅ audio_device.py              # PyAudio 设备管理

pipeline/
✅ event_bus.py                 # 事件驱动架构
✅ intent_manager.py            # Intent 队列管理
✅ voice_pipeline.py            # 核心流水线调度
✅ chat_orchestrator.py         # 消息编排
✅ async_queue_manager.py       # 异步任务队列

utils/
✅ text_chunker.py              # TTS 文本分块
✅ audio_processor.py           # 音频处理工具
✅ model_quantizer.py           # 模型量化
✅ npu_acceleration.py          # NPU 加速

models/
✅ chat_context.py              # 对话上下文模型
✅ voice_config.py              # 语音配置

ui/
✅ cli.py                       # CLI 版本
✅ webui/app.py                 # Streamlit 应用
✅ desktop/main.py              # PyQt 桌面应用

tests/
✅ test_stt.py
✅ test_llm.py
✅ test_tts.py
✅ test_pipeline.py
✅ test_integration.py

✅ main.py                      # 应用入口
✅ requirements.txt             # 依赖清单
✅ config.yaml                  # 运行时配置
✅ docker-compose.yml           # Docker 编排
```

---

## 六、模型与依赖

### 6.1 核心依赖包

```txt
# requirements.txt

# 音频处理
numpy>=1.24.0
scipy>=1.11.0
librosa>=0.10.0
soundfile>=0.12.0
pyaudio>=0.2.13

# STT
faster-whisper>=1.0.0
# 或
openai-whisper>=20240415

# LLM 推理
vllm>=0.4.0
# 或
llama-cpp-python>=0.2.0
ctransformers>=0.2.27

# TTS
torch>=2.0.0
torchaudio>=2.0.0
# FastPitch & HiFi-GAN
# vocoder>=0.1.0

# Coqui TTS 备选
TTS>=0.22.0

# 量化支持
bitsandbytes>=0.41.0
auto-gptq>=0.6.0
awq>=0.1.0

# Web UI
streamlit>=1.28.0
gradio>=4.0.0

# 桌面 UI
PyQt5>=5.15.0
# 或
wxPython>=4.2.0

# 工具库
python-dotenv>=1.0.0
pydantic>=2.0.0
click>=8.1.0

# 异步和并发
aiofiles>=23.0.0
asyncio-contextmanager>=1.0.0

# 监控和日志
structlog>=23.0.0
python-json-logger>=2.0.0

# 开发测试
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
ruff>=0.1.0
```

### 6.2 模型下载清单

```bash
# 下载脚本：scripts/download_models.sh

# STT 模型
faster-whisper-base-en      (~140MB)
faster-whisper-small-en     (~500MB)

# LLM 模型
# 选项 1：vLLM 格式
Qwen-7B-Chat-GPTQ           (~4GB)
Mistral-7B-Instruct-v0.3    (~4GB)

# 选项 2：GGML 格式
Llama-2-7B-GGML (Q4_K_M)    (~4GB)
Phi-2-GGML (Q4_K_M)         (~2GB)

# TTS 模型
fastpitch-ljspeech          (~200MB)
hifigan-ljspeech            (~100MB)

# VAD 模型（可选）
silero_vad                  (~40MB)
```

### 6.3 硬件需求

```
最低配置（CPU 推理）：
  • CPU: 4 核 2.4GHz
  • 内存: 8GB
  • 存储: 20GB（模型 15GB + 系统 5GB）
  • 网络: 初次下载需要（模型约 15-20GB）
  • 音频设备: 麦克风 + 扬声器

推荐配置（GPU 加速）：
  • GPU: RTX 3060 / RTX 4070 或等效
  • 显存: 8GB+（LLM），6GB+（TTS）
  • CPU: 8 核 + 16GB RAM
  • 存储: SSD 20GB
  • 推理速度: 200+ tokens/sec

高端配置（多模型并行）：
  • GPU: RTX 4090 或 H100
  • 显存: 24GB+
  • CPU: 16+ 核心
  • 内存: 64GB RAM
  • 推理速度: 500+ tokens/sec
```

---

## 七、边缘计算极限优化（4GB RAM / 0.9B-1.5B / 10TOPS NPU）

> **场景针对**：RDK（高通骁龙）、树莓派、Jetson Nano、Kunlun、Sophon 等边缘设备
> **约束条件**：总内存 4GB、NPU 10TOPS、模型 < 2GB

### 7.1 边缘设备硬件约束分析

#### 7.1.1 4GB RAM 内存预算分配

```
总内存: 4GB (4096MB)

分配策略：
┌──────────────────────────────────┐
│ 系统 OS + 基础服务      ~800MB    │
│ Python Runtime + 库      ~600MB    │  <-- 固定开销 ~1.4GB
│─────────────────────────────────  │
│ STT 模型加载           ~300MB     │
│ LLM 模型加载           ~600MB     │  <-- 模型加载 ~1.2GB
│ TTS 模型加载           ~300MB     │
│ VAD 模型加载            ~30MB     │
│─────────────────────────────────  │
│ 运行时缓冲              ~800MB     │  <-- 工作区 ~0.8GB
│  - 音频缓冲 (VAD/STT)   ~200MB
│  - LLM Context 缓冲     ~300MB
│  - 音频输出缓冲         ~200MB
│  - 其他队列缓冲         ~100MB
│─────────────────────────────────  │
│ 剩余（安全边际）        ~500MB     │
└──────────────────────────────────┘

关键：
✅ 模型权重总和 ~1.2GB（可接受）
✅ 工作缓冲 ~0.8GB（控制流式IO）
✅ 500MB 安全边际（应对 GC、临时分配）
❌ 不能同时加载多个完整模型
❌ 不能支持批处理
```

#### 7.1.2 0.9B vs 1.5B 模型对比

```
模型大小对比（GGML Q3_K_S 量化）：

标准对比：
┌──────────────┬────────────┬──────────┬────────────┬──────────┐
│ 模型大小     │ 内存占用   │ 首词延迟 │ 吞吐量     │ 中文能力 │
├──────────────┼────────────┼──────────┼────────────┼──────────┤
│ 0.9B (Q3_K_S)│ ~500MB     │ 800-1200ms│ 15-20 t/s │ 中等     │
│ 1.5B (Q3_K_M)│ ~750MB     │ 600-1000ms│ 10-15 t/s │ 较好     │
└──────────────┴────────────┴──────────┴────────────┴──────────┘

推荐选择：
🎯 首选 1.5B（中文SOTA）：
   • Qwen-1.5B-Chat
   • MiniCPM-1.5B
   • Phi-1.5B-v2
   
🔄 备选 0.9B（极速）：
   • Qwen-0.5B（实际~0.9B参数）
   • Phi-9B 量化版本
   
两者都支持 CN Hub 快速下载
```

#### 7.1.3 10TOPS NPU 推理优化

```python
# src/services/rdk_npu_accelerator.py

class RDKNPUAccelerator:
    """RDK 高通骁龙 NPU 特定优化"""
    
    @staticmethod
    def detect_qualcomm_npu():
        """检测高通 AI Engine"""
        import platform
        if "qcom" in platform.processor().lower():
            return "qualcomm_hexagon"
        return None
    
    @staticmethod
    def optimize_for_10tops():
        """
        10TOPS NPU 特性：
        • 理论吞吐：10,000,000,000 operations/sec
        • 对于 1.5B 模型：每个 token ~150B ops
          → 理论吞吐 ~66 tokens/sec（但实际 ~10-15 t/s）
        
        瓶颈分析：
        ❌ 内存带宽（PCIe 或总线）
        ❌ 量化精度（int8）
        ❌ 模型优化程度
        
        优化策略：
        ✅ 使用 ONNX Runtime Qualcomm QNN 后端
        ✅ 启用 NPU 上的权重缓冲
        ✅ 按需转移数据（避免反复 IO）
        """
        
        try:
            from onnxruntime.transformers.onnx_model_rdk import RDKModel
            return "qnn_backend"
        except ImportError:
            return "ggml_fallback"  # GGML CPU 降级
    
    @staticmethod
    def config_for_rdk() -> dict:
        """RDK 特定配置"""
        return {
            "llm": {
                "backend": "onnx_qnn",  # 使用 QNN
                "quantization": "uint8",
                "execution_provider": "QnnExecutionProvider",
                "num_threads": 4,  # RDK 通常是 4-8 核
                "cache_dir": "/tmp/rdk_model_cache"
            },
            "memory": {
                "gpu_memory_fraction": 0.8,  # NPU 不占主内存
                "max_context_tokens": 512,  # 限制上下文
                "stream_buffer_size_mb": 50
            },
            "inference": {
                "batch_size": 1,  # 严格单个请求
                "beam_search": False,
                "temperature": 0.7,
                "max_new_tokens": 256
            }
        }
```

### 7.2 0.9B / 1.5B 本地模型可选方案

#### 7.2.1 中文支持矩阵

```
推荐方案（支持中文 + GGML 量化）：

┌─────────────────────────────────────────────────────────┐
│ 🥇 最佳选择：Qwen-1.5B-Chat                             │
├─────────────────────────────────────────────────────────┤
│ 官网：https://huggingface.co/Qwen/Qwen-1.5B-Chat        │
│ 大小：1.5B 参数                                         │
│ 中文：一流（原生对齐）                                 │
│ 量化：支持 GGML (Q2_K, Q3_K), GPTQ                     │
│ 许可：Apache 2.0 商用友好                               │
│                                                         │
│ 下载：                                                  │
│ $ huggingface-cli download Qwen/Qwen-1.5B-Chat --local-dir ./models/qwen-1.5b
│                                                         │
│ 量化版本下载：                                           │
│ $ huggingface-cli download second-state/Qwen-1.5B-Chat-GGUF --local-dir ./models/
│                                                         │
│ GGML 模型：                                             │
│ - qwen-1.5b-chat-q3_k_m.gguf  (~750MB)                 │
│ - qwen-1.5b-chat-q2_k.gguf    (~500MB)  ✅ 首选        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 🥈 极速方案：MiniCPM-1.5B                               │
├─────────────────────────────────────────────────────────┤
│ 官网：https://huggingface.co/openbmb/MiniCPM-1.5B       │
│ 特点：特别针对移动端优化                                │
│ 速度：~20% 更快                                         │
│ 质量：接近 Qwen-1.5B                                    │
│ 大小：1.5B                                              │
│                                                         │
│ 推荐场景：优先速度                                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 🥉 备选：Phi-1.5B-v2 (微软)                             │
├─────────────────────────────────────────────────────────┤
│ 官网：https://huggingface.co/microsoft/phi-1.5                  │
│ 优点：推理超快，英文最优                                │
│ 缺点：中文支持较弱                                       │
│ 大小：1.5B                                              │
│                                                         │
│ 使用场景：英文优先或需要超速                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 🚀 极速版本：Qwen-0.5B                                  │
├─────────────────────────────────────────────────────────┤
│ 官网：https://huggingface.co/Qwen/Qwen-0.5B             │
│ 大小：500M 参数（比 Qwen-1.5B 小 3x）                  │
│ 内存：~300MB Q2_K 量化                                  │
│ 速度：~2x 快                                            │
│ 质量：可接受但明显下降                                  │
│                                                         │
│ 适用场景：4GB 内存紧张时的机降选择                      │
└─────────────────────────────────────────────────────────┘
```

#### 7.2.2 GGML 量化方案对比

```python
# 量化精度 vs 质量 vs 大小 权衡

量化方案对比表：

┌─────────┬──────────────┬──────────────┬─────────────┬────────────┐
│ 量化    │ 大小(1.5B)   │ 质量损失     │ 推理速度    │ 推荐度     │
├─────────┼──────────────┼──────────────┼─────────────┼────────────┤
│ Q2_K    │ ~500MB       │ 中等-严重    │ 最快        │ ⭐需网络  │
│ Q3_K_S  │ ~650MB       │ 轻微-中等    │ 快          │ ⭐推荐    │
│ Q3_K_M  │ ~750MB       │ 轻微         │ 中等        │ ⭐⭐首选  │
│ Q4_K_M  │ ~950MB       │ 很轻微       │ 中等-慢     │ ❌超预算  │
│ Q5_K_M  │ ~1.1GB       │ 基本无损     │ 慢          │ ❌内存不足│
│ F16     │ ~3GB         │ 无损         │ 最慢        │ ❌不可用  │
└─────────┴──────────────┴──────────────┴─────────────┴────────────┘

推荐方案：
🎯 边缘首选：Q3_K_M
   理由：质量与大小平衡、速度接受、内存预算充足
   
🚀 极端情况：Q2_K
   用途：内存非常紧张时
   质量：明显下降（考虑是否可用）
   
🔄 备选：Q3_K_S
   若 Q3_K_M 仍然超预算
```

#### 7.2.3 模型自动下载与量化脚本

```python
# scripts/download_edge_models.py
"""边缘设备模型下载与量化脚本"""

import os
import subprocess
from pathlib import Path

def download_qwen_models():
    """下载 Qwen 量化版本"""
    model_dir = Path("./models/llm")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 方案 1：下载已量化的 GGUF（推荐）
    print("📥 下载 Qwen-1.5B-Chat-GGUF...")
    subprocess.run([
        "huggingface-cli", "download",
        "second-state/Qwen-1.5B-Chat-GGUF",
        "qwen-1.5b-chat-q3_k_m.gguf",  # 750MB
        "--local-dir", str(model_dir),
        "--local-dir-use-symlinks", "False"
    ])
    
    # 方案 2：下载原始模型并自行量化（若需要）
    # print("📥 下载原始 Qwen-1.5B...")
    # subprocess.run([
    #     "huggingface-cli", "download",
    #     "Qwen/Qwen-1.5B-Chat",
    #     "--local-dir", str(model_dir),
    # ])
    
    print("✅ 模型下载完成")


def quantize_to_ggml(model_path: str, output_path: str, quant_type: str = "q3_k_m"):
    """使用 llama.cpp 进行 GGML 量化"""
    
    # 如果已经是 GGUF，跳过
    if output_path.endswith(".gguf"):
        print(f"✅ 已是 GGUF 格式：{output_path}")
        return
    
    print(f"⚙️  量化中... ({quant_type})")
    
    # 使用 llama.cpp 的量化工具
    # 需要编译 llama.cpp：https://github.com/ggerganov/llama.cpp
    subprocess.run([
        "./llama.cpp/quantize",
        model_path,
        output_path,
        quant_type  # q3_k_m, q2_k, 等
    ])
    
    print(f"✅ 量化完成：{output_path}")


def prepare_stt_model():
    """下载 STT 模型（更轻量的选项）"""
    model_dir = Path("./models/stt")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 方案 1：faster-whisper tiny（最轻）
    print("📥 下载 Whisper Tiny...")
    subprocess.run([
        "python", "-c",
        """
from faster_whisper import WhisperModel
model = WhisperModel("tiny", device="cpu", compute_type="int8")
"""
    ])
    
    # 备选方案 2：Paraformer 或其他轻量 STT
    # 支持更好的中文且模型更小
    

def prepare_tts_model():
    """TTS 模型（边缘设备最轻选项）"""
    model_dir = Path("./models/tts")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Kokoro（最轻，已在 AIRI 中验证）
    print("📥 复用 AIRI Kokoro TTS...")
    # 或 FastPitch + HiFi-GAN 的 ONNX 版本
    

if __name__ == "__main__":
    print("🚀 开始下载边缘设备模型")
    print(f"📊 目标内存预算：< 1.5GB 模型 + 0.8GB 运行时")
    
    download_qwen_models()
    prepare_stt_model()
    prepare_tts_model()
    
    print("\n✅ 所有模型准备完成！")
    print("📐 总大小估计：STT ~300MB + LLM ~750MB + TTS ~300MB = ~1.35GB")
```

### 7.3 内存受限的架构改造

#### 7.3.1 单线程顺序处理（不并发）

```python
# src/pipeline/edge_pipeline.py
"""边缘设备优化版本：严格顺序处理"""

import gc
from typing import AsyncIterator

class EdgeVoicePipeline:
    """
    内存受限的语音流水线
    
    核心改变：
    ❌ 停止并行 TTS 处理
    ❌ 停止缓存大型中间结果
    ✅ 严格序列处理：VAD → STT → LLM → TTS → 播放
    ✅ 立即释放不需要的对象
    ✅ 流式 IO（不一次性加载）
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.stt = None
        self.llm = None
        self.tts = None
        
        # 严格限制缓冲区大小
        self.audio_buffer_max_sec = 30  # 最多 30 秒
        self.text_buffer_max_tokens = 256
        self.max_context_tokens = 512  # 历史对话
    
    async def initialize_lazy(self):
        """延迟加载：接收音频时才加载模型（可选）"""
        # 推迟模型加载，减少初始内存
        pass
    
    async def run_single_turn(self):
        """单轮对话流程（内存无泄漏）"""
        
        try:
            # [1] 录音 & VAD
            print("🎤 监听中...")
            audio_segment = await self._record_until_speech_ends()
            print(f"✅ 捕获音频：{len(audio_segment)} bytes")
            
            # [2] STT（转录后立即释放原始音频）
            user_text = await self.stt.transcribe(audio_segment)
            del audio_segment  # 立即释放
            gc.collect()  # 强制垃圾回收
            
            print(f"👤 用户：{user_text}")
            
            # [3] LLM（流式生成，不缓存中间结果）
            response_text = ""
            print("🤖 AI 思考中...")
            
            async for token in self.llm.stream_chat(user_text):
                response_text += token
                
                # 每收集 50 个字符就触发 TTS
                if len(response_text) >= 50 or token in "。！？…":
                    await self._stream_tts(response_text)
                    response_text = ""  # 清空
            
            # 处理剩余文本
            if response_text:
                await self._stream_tts(response_text)
            
            gc.collect()  # 回收 LLM 推理临时对象
        
        except Exception as e:
            print(f"❌ 错误：{e}")
            gc.collect()
    
    async def _stream_tts(self, text: str):
        """流式 TTS 并立即播放"""
        
        print(f"🔊 TTS：{text[:20]}...")
        
        # TTS 立即合成并播放（不缓存）
        audio_chunk = await self.tts.synthesize(text)
        
        # 直接播放（无缓冲队列）
        await self._play_audio_streaming(audio_chunk)
        
        # 立即释放
        del audio_chunk
        gc.collect()
    
    async def _record_until_speech_ends(self) -> bytes:
        """VAD 录音直到检测到完整句子"""
        
        # 帧缓冲（最多 30 秒）
        max_frames = int(16000 * 30 / 1024)  # 16kHz, 1024 样本/帧
        frames = []
        speech_detected = False
        silence_frames = 0
        silence_threshold = int(0.5 * 16000 / 1024)  # 0.5 秒
        
        async for frame in self.audio.listen_stream():
            is_speech = await self.vad.detect(frame)
            
            if is_speech:
                speech_detected = True
                silence_frames = 0
            else:
                if speech_detected:
                    silence_frames += 1
            
            # 缓冲帧
            frames.append(frame)
            
            # 检查停止条件
            if speech_detected and silence_frames > silence_threshold:
                break  # 检测到完整句子
            
            if len(frames) > max_frames:
                break  # 超时（防止记忆溢出）
        
        # 转为字节流
        return b"".join(frames)
    
    async def _play_audio_streaming(self, audio_data: bytes):
        """流式播放（无缓冲）"""
        
        chunk_size = 8192
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            await self.audio.play(chunk)
            # 实时播放，不缓冲全部


class EdgeTextChunker:
    """边缘优化的文本分块"""
    
    @staticmethod
    def chunk_greedy(text: str, max_chars: int = 50) -> list[str]:
        """贪心分块：优先速度而非完美分割"""
        
        chunks = []
        current = ""
        
        for char in text:
            current += char
            
            # 简单规则：长度或标点符号
            if len(current) >= max_chars or char in "。！？":
                if current:
                    chunks.append(current)
                current = ""
        
        if current:
            chunks.append(current)
        
        return chunks
```

#### 7.3.2 GGML 推理后端集成（针对 NPU）

```python
# src/services/edge_llm_service.py
"""GGML 后端 LLM 服务（RDK NPU 优化）"""

import asyncio
from llama_cpp import Llama

class EdgeLLMService:
    """
    专为边缘计算优化的 LLM 服务
    """
    
    def __init__(self, model_path: str, config: dict):
        self.model_path = model_path
        self.config = config
        
        # GGML 模型初始化（关键参数调优）
        self.llm = Llama(
            model_path=model_path,
            
            # 线程配置（RDK 通常 4-8 核）
            n_threads=4,
            n_threads_batch=4,
            
            # NPU 推理配置
            n_gpu_layers=-1,  # -1 = 全部转移到 GPU/NPU
            
            # 上下文大小（受内存限制）
            n_ctx=512,  # 最多 512 tokens 上下文
            
            # 批大小（必须 = 1）
            n_batch=256,  # 预分配缓冲
            
            # 内存映射（减少 RAM）
            mmap=True,
            mlock=False,  # 不锁定内存（节省 RAM）
            
            # 量化感知（GGML 已处理）
            verbose=False
        )
        
        self.system_prompt = """你是一个有帮助的中文 AI 助手。
回答要简洁、准确、有帮助。
"""
        self.chat_history = []
        self.max_history_tokens = 256  # 限制历史长度
    
    async def stream_chat(self, user_input: str):
        """流式聊天生成"""
        
        # 构建 prompt
        messages = self._build_messages(user_input)
        
        # 使用 create_chat_completion_stream（支持流式）
        # 每个 token 作为一个事件返回
        
        loop = asyncio.get_event_loop()
        
        # 在线程池中运行（避免阻塞）
        async for token in loop.run_in_executor(
            None,
            self._stream_generator,
            messages
        ):
            yield token
    
    def _stream_generator(self, messages: list):
        """原生 GGML stream（在线程中执行）"""
        
        full_response = ""
        
        # Llama.cpp 的流式容器
        response = self.llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.7,
            top_p=0.95,
            max_tokens=256  # 严格限制输出
        )
        
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                full_response += token
                yield token
        
        # 更新历史
        self._update_history(full_response)
    
    def _build_messages(self, user_input: str) -> list:
        """构建消息（只保留最近对话）"""
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 限制历史到 max_history_tokens
        history_tokens = 0
        for msg in reversed(self.chat_history):
            msg_tokens = len(msg["content"].split())
            if history_tokens + msg_tokens > self.max_history_tokens:
                break
            messages.insert(1, msg)
            history_tokens += msg_tokens
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _update_history(self, response: str):
        """更新对话历史（限制长度）"""
        
        # 仅保留最近 10 条消息
        max_messages = 10
        
        if len(self.chat_history) >= max_messages:
            self.chat_history = self.chat_history[-max_messages+2:]
        
        self.chat_history.append({
            "role": "assistant",
            "content": response
        })
```

### 7.4 NPU 推理加速配置（RDK 高通骁龙）

#### 7.4.1 使用 Qualcomm QNN 后端

```python
# src/services/qualcomm_qnn_backend.py
"""高通 QNN (Qualcomm Neural Network) 后端"""

class QNNBackend:
    """
    Qualcomm QNN 是高通 Hexagon NPU 的推理框架
    
    支持设备：
    • RDK（机器人开发套件）
    • 骁龙 Drive / Ride
    • 其他高通 Snapdragon 8 系列
    
    性能：比 CPU 快 3-5 倍
    """
    
    @staticmethod
    def setup_qnn_environment():
        """设置 QNN 环境"""
        
        import os
        
        # QNN SDK 路径
        os.environ["QNN_SDK_PATH"] = "/opt/qcom/qnn"
        os.environ["QNN_MODELS_PATH"] = "./models"
        
        # 后端选择（Hexagon = NPU，Cpu = CPU 降级）
        os.environ["QNN_BACKEND"] = "hexagon"  # 或 "cpu"
    
    @staticmethod
    def convert_model_to_qnn(pytorch_model_path: str, onnx_path: str):
        """
        转换流程：PyTorch → ONNX → QNN Model
        
        步骤 1：导出 ONNX
            python export_to_onnx.py
        
        步骤 2：QNN 转换
        """
        
        import subprocess
        
        # 使用 QNN 工具链转换
        cmd = [
            "qnn-onnx-converter",
            "--input_network", onnx_path,
            "--output_path", "./models/qnn",
            "--quantization_config", "quantization.json"
        ]
        
        subprocess.run(cmd)
    
    @staticmethod
    def run_qnn_inference():
        """
        QNN 推理示例
        """
        
        try:
            from qnn_sdk import QNNModel
            
            model = QNNModel(
                model_path="./models/qnn/qwen-1.5b-q3_k_m.qnn",
                backend="hexagon"
            )
            
            # 前向推理
            output = model.infer(input_data)
            return output
        
        except ImportError:
            print("⚠️  QNN SDK 未安装，降级到 GGML CPU")
            return None
```

#### 7.4.2 ONNX Runtime 结合 QNN

```python
# src/services/onnxrt_qnn.py
"""使用 ONNX Runtime 的 QNN 执行提供者"""

import onnxruntime as rt

def create_rdk_session(model_path: str) -> rt.InferenceSession:
    """
    创建针对 RDK 的 ONNX 推理会话
    """
    
    # 执行提供者顺序（优先级）
    providers = [
        ('QnnExecutionProvider', {}),  # 高通 NPU（优先）
        ('CPUExecutionProvider', {}),  # CPU 降级
    ]
    
    session = rt.InferenceSession(
        model_path,
        providers=providers,
        sess_options=rt.SessionOptions()
    )
    
    print(f"✅ 使用执行提供者：{session.get_providers()}")
    
    return session
```

### 7.5 4GB 内存优化版配置示例

```yaml
# config-edge-rdk.yaml
# 专用于 RDK 4GB RAM 边缘设备的配置

app:
  name: "LocalAI Voice Edge"
  version: "0.1.0"
  log_level: "WARNING"  # 减少日志开销
  debug: false

# STT 配置（轻量版）
stt:
  engine: "faster-whisper"
  model_name: "tiny"  # ⚠️ tiny 不是 base，更轻量
  model_path: "models/stt/whisper-tiny.pt"
  language: "zh"
  device: "cpu"  # ⚠️ NPU 可能不支持 STT
  compute_type: "int8"
  batch_size: 1

# LLM 配置（GGML + RDK NPU）
llm:
  engine: "ggml"  # 改为 GGML 而非 vLLM
  model_name: "Qwen-1.5B-Chat-GGUF"
  model_path: "models/llm/qwen-1.5b-chat-q3_k_m.gguf"
  
  # GGML 参数
  n_threads: 4
  n_gpu_layers: -1  # 全部用 NPU
  n_ctx: 512  # ⚠️ 严格限制上下文
  n_batch: 256
  
  # 推理优化
  temperature: 0.7
  top_p: 0.95
  max_tokens: 256  # ⚠️ 限制生成长度
  
  # 内存
  mmap: true
  mlock: false  # ⚠️ 不锁定内存

# TTS 配置（轻量版）
tts:
  engine: "kokoro"  # Kokoro 最轻量
  model_path: "models/tts/kokoro.onnx"  # ONNX 格式
  device: "cpu"
  batch_size: 1

# VAD 配置
vad:
  enabled: true
  model_path: "models/vad/silero_vad.jit"
  threshold: 0.6
  speech_pad_ms: 80
  min_silence_duration_ms: 500  # ⚠️ 拉长以避免误检

# 音频
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  
  # RDK 音频设备
  input_device_index: -1  # 默认麦克风
  output_device_index: -1  # 默认扬声器
  backend: "pyaudio"

# 流水线（禁用并发）
pipeline:
  # 文本分块
  text_chunking:
    min_words: 5  # 提高阈值，减少分块数
    max_words: 20
    boost: 1
  
  # ⚠️ 严格单线程
  max_concurrent_tasks: 1  # 禁止并发
  sequential_only: true

# 系统提示
system_prompt: |
  你是一个有帮助的中文 AI 助手。
  请简洁回答用户的问题，不超过 200 字。

# 内存监控
monitoring:
  enabled: false  # 禁用以节省开销
  memory_warning_threshold_mb: 3500  # 接近 4GB 时警告
```

### 7.6 实际延迟基准与期望

```
基于 RDK + Qwen-1.5B 的预期性能：

┌──────────────────────────────────────┐
│ 组件         单轮耗时        (可调)  │
├──────────────────────────────────────┤
│ 1. VAD检测   50-200ms       实时    │
│ 2. STT转录   800-1200ms     (Whisper tiny)
│ 3. LLM首词   600-1000ms     (1.5B on 10TOPS)
│ 4. LLM生成   100毫秒/token   (10-15 t/s)
│ 5. TTS合成   300-500ms      (FastPitch)
│ 6. 音频播放  同播放实时      (流式)
├──────────────────────────────────────┤
│ 总计（完整回复）    4-8秒              │
│  └─首词延迟          1.8-2.4秒         │
│  └─首次播放          2.5-3.5秒         │
└──────────────────────────────────────┘

实际案例：
用户说："你好，天气怎么样？"
    VAD 检测:  0.1s ─┐
    STT:       1.0s ─┼─→ 1.1s (可接受等待)
    ────────────────
    LLM:       0.8s ─┬─→ 0.8s (思考)
    TTS 首块:  0.3s ─┼─→ 1.1s 开始播放 ✅
    ────────────────
    对话循环时间: ~1.5-2 秒一个新话题

可接受性检查：
✅ < 5秒应答时间✅ 实时流式播放（不等全部生成）
✅ 连续对话流畅性可接受
⚠️  不如 GPU 的毫秒级响应（但可用）

优化空间：
• 预热模型：节省 200-400ms（初次加载）
• 减少 context 窗口：加速 100-150ms
• lower max_tokens：更快但可能截断
```

### 7.7 部署检查清单（RDK 特定）

```
[ ] 硬件验证
  [ ] 确认 RDK 有 NPU（cat /proc/cpuinfo 检查）
  [ ] 验证内存：free -h | 应该显示 ~4GB
  [ ] 检查存储：df -h | 至少 10GB 空余
  [ ] 网络：初次下载模型需要网络（~1.5GB）

[ ] 软件环境
  [ ] 安装 Python 3.10+
  [ ] 安装 llama-cpp-python
    pip install llama-cpp-python
  [ ] 安装 faster-whisper
    pip install faster-whisper
  [ ] 可选：安装 Qualcomm QNN SDK

[ ] 模型准备
  [ ] 下载 Qwen-1.5B-Chat-GGUF 量化版本
  [ ] 下载 Whisper-tiny 模型
  [ ] 验证模型大小：< 1.5GB 总和
  [ ] 测试模型加载（可能较慢）

[ ] 初次运行
  [ ] 运行 main.py --test-stt （验证 STT）
  [ ] 运行 python -c "from llama_cpp import Llama; ..." (验证 GGML)
  [ ] 记录实际延迟基准

[ ] 性能调优
  [ ] 监控内存使用：free -h
  [ ] 监控 CPU：top -b
  [ ] 若内存超过 3.5GB，启用更激进的 GC

[ ] 故障排除
  [ ] OOM 错误 → 降级到 0.5B 或 Q2_K
  [ ] 推理缓慢 → 检查 NPU 是否正确启用
  [ ] STT 超时 → 增大 min_silence_duration_ms
```

---

## 八、配置与部署

### 7.1 配置文件结构

```yaml
# config.yaml

# 应用设置
app:
  name: "Local AI Voice System"
  version: "0.1.0"
  log_level: "INFO"
  debug: false

# STT 配置
stt:
  engine: "faster-whisper"
  model_name: "base"  # base, small, medium, large
  model_path: "models/stt/faster-whisper-base/"
  language: "zh"  # "en", "zh", "multi"
  device: "auto"  # "auto", "cuda", "cpu"
  compute_type: "int8"  # "float32", "float16", "int8"

# LLM 配置
llm:
  engine: "vllm"  # "vllm", "ggml", "ollama"
  model_name: "Qwen-7B-Chat-GPTQ"
  model_path: "models/llm/Qwen-7B-Chat-GPTQ/"
  
  # vLLM 特定
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  dtype: "float16"
  
  # 推理参数
  max_tokens: 512
  temperature: 0.7
  top_p: 0.95
  
  # GGML 特定（如果启用）
  n_gpu_layers: -1  # -1 转移全部到 GPU
  n_threads: 8

# TTS 配置
tts:
  engine: "fastpitch_hifigan"  # "fastpitch", "coqui", "kokoro"
  fastpitch_model: "models/tts/fastpitch.pt"
  hifigan_model: "models/tts/hifigan.pt"
  device: "auto"
  
  # 优化
  use_half_precision: true
  batch_size: 1

# VAD 配置
vad:
  enabled: true
  model_path: "models/vad/silero_vad.jit"
  threshold: 0.6
  speech_pad_ms: 80
  min_silence_duration_ms: 400

# 音频设备
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  
  # 输入设备
  input_device_index: -1  # -1 = 默认
  
  # 输出设备
  output_device_index: -1
  
  # 播放
  backend: "pyaudio"  # "pyaudio", "sounddevice"

# 流水线
pipeline:
  # 文本分块
  text_chunking:
    min_words: 3
    max_words: 15
    boost: 2
  
  # 并发控制
  max_concurrent_tts_tasks: 3
  tts_queue_timeout_sec: 30

# 系统提示
system_prompt: |
  你是一个有帮助的 AI 助手。你的回答应该简洁、准确且有帮助。
  
  # 例子
  用户：今天天气怎么样？
  助手：抱歉，我无法实时了解天气信息。建议您查看天气预报应用。

# 日志
logging:
  level: "INFO"
  format: "json"  # "text", "json"
  file: "logs/app.log"
  max_size_mb: 100
  backup_count: 5

# 性能监控
monitoring:
  enabled: true
  metrics_port: 8090  # Prometheus 指标
  profile_enabled: false  # CPU profiling
```

### 7.2 Docker 部署

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 创建模型目录
RUN mkdir -p models/{stt,llm,tts,vad}

# 预下载模型（可选）
# RUN python scripts/download_models.py

# 暴露端口
EXPOSE 8090 8091

# 启动应用
CMD ["python", "main.py", "--config", "config.yaml"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  local-ai-voice:
    build: .
    container_name: local-ai-voice-system
    
    # GPU 支持
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    # 挂载卷
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
      - ./logs:/app/logs
      - /etc/localtime:/etc/localtime:ro
      - /dev/snd:/dev/snd  # 音频设备
    
    # 端口暴露
    ports:
      - "8090:8090"  # Prometheus 指标
      - "8091:8091"  # Web UI
    
    # 环境变量
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - TORCH_HOME=/app/models
      - PYTHONUNBUFFERED=1
    
    # 资源限制
    mem_limit: 16g
    cpus: 8
    
    # 启动设置
    restart: unless-stopped
    stdin_open: true
    tty: true
```

---

## 九、性能优化策略

### 9.1 延迟优化目标

```
目标：从用户说话到开始播放回复 < 2 秒

当前延迟分解：
  VAD 检测: 100-300ms (实时进行)
  STT 转录: 500-2000ms (Whisper base)
  LLM 推理: 1000-3000ms (第一个 token)
  TTS 首块: 300-500ms (FastPitch + HiFi-GAN)
  ──────────────────────────────
  总计: 1.9-5.8 秒

优化方案：
  1. 预加载模型到 GPU (节省 200-500ms)
  2. STT faster-whisper (节省 40% 时间)
  3. 提前分块采集文本 (边生成边播放)
  4. TTS 并行处理 (减少等待时间)
  
优化后目标：< 1.5 秒
```

### 9.2 内存优化

```python
# src/utils/memory_optimizer.py

class MemoryOptimizer:
    """内存使用优化"""
    
    @staticmethod
    def enable_gradient_checkpointing(model):
        """梯度检查点（推理时减少内存）"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    @staticmethod
    def clear_cache():
        """定期清理缓存"""
        torch.cuda.empty_cache()
    
    @staticmethod
    def reduce_batch_size(current_size: int) -> int:
        """动态降低批大小以适应显存"""
        return max(1, current_size // 2)
```

### 9.3 CPU 优化

```python
# CPU 线程数优化
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# GGML 推理优化
llm = Llama(
    model_path,
    n_threads=8,
    n_gpu_layers=-1,
    n_batch=512
)
```

---

## 十、测试与验证

### 10.1 测试计划

```python
# tests/test_integration.py

@pytest.mark.asyncio
async def test_end_to_end_voice_interaction():
    """完整端到端测试"""
    
    # 1. 初始化组件
    services = await initialize_services(config)
    pipeline = VoicePipeline(**services)
    
    # 2. 模拟音频输入
    audio_file = "tests/fixtures/test_audio.wav"
    audio_data = load_audio(audio_file)
    
    # 3. 运行流水线
    start_time = time.time()
    
    intent = await pipeline.process_voice_input(audio_data)
    response_text = intent.text_buffer
    audio_output = intent.audio_buffer
    
    elapsed_time = time.time() - start_time
    
    # 4. 验证输出
    assert response_text, "应该生成回复"
    assert audio_output, "应该生成音频"
    assert elapsed_time < 5.0, f"延迟应该 < 5s，实际 {elapsed_time}s"
    
    print(f"✅ 测试通过: {elapsed_time:.2f}s")


@pytest.mark.asyncio
async def test_streaming_performance():
    """流式性能测试"""
    
    llm_service = await load_llm_service()
    text_generator = llm_service.stream_chat("你好")
    
    first_token_time = None
    token_count = 0
    times = []
    
    async for token in text_generator:
        current_time = time.time()
        
        if first_token_time is None:
            first_token_time = current_time
        
        times.append(current_time - (times[-1] if times else first_token_time))
        token_count += 1
    
    avg_token_time = sum(times) / len(times) * 1000  # ms
    tokens_per_sec = 1000 / avg_token_time
    
    print(f"⏱️  平均令牌延迟: {avg_token_time:.1f}ms")
    print(f"📊 吞吐量: {tokens_per_sec:.1f} tokens/sec")
    print(f"🚀 首个令牌延迟: {(first_token_time - times[0]) * 1000:.1f}ms")
```

---

## 十一、部署检查清单

### 11.1 发布前验证

```
□ 功能测试
  □ STT 多语言测试（中英日）
  □ LLM 上下文保留测试
  □ TTS 实时流式播放测试
  □ 错误恢复机制验证

□ 性能测试
  □ 端到端延迟 < 2s
  □ 内存占用 < 8GB（CPU）
  □ CPU 使用率 < 80%
  □ GPU 显存 < 显卡容量

□ 兼容性测试
  □ Windows 10/11
  □ macOS 12+
  □ Ubuntu 20.04+
  □ Docker 环境

□ 安全检查
  □ 无网络泄露检查
  □ 本地模型验证
  □ 敏感信息不记录

□ 文档完善
  □ README 部署指南
  □ 快速开始教程
  □ 配置参数说明
  □ 故障排除指南
  □ 性能调优建议

□ 代码质量
  □ 代码覆盖率 > 70%
  □ 类型检查 mypy 通过
  □ Lint ruff 通过
  □ 格式化 black 通过
```

---

## 十二、后期优化与扩展方向

### 12.1 短期优化（1-3 个月）

- [ ] 支持本地微调小模型
- [ ] 实现模型缓存预热
- [ ] 多语言自动切换
- [ ] 长对话上下文压缩
- [ ] 音频增强与去噪

### 12.2 中期扩展（3-6 个月）

- [ ] 支持多个 AI 角色
- [ ] 声纹识别与个性化
- [ ] 离线知识库集成
- [ ] 本地 RAG 系统
- [ ] 移动端部署（Android/iOS）

### 12.3 长期愿景（6-12 个月）

- [ ] 全语言 SOTA 模型
- [ ] 多模态输入（视频、文本、语音混合）
- [ ] 实时情感识别
- [ ] 自学习与在线微调
- [ ] 边缘设备优化

---

## 十三、参考与对标

### AIRI 项目学习点

✅ **采用的架构模式**：
  1. Event Bus 事件驱动
  2. Intent 单位管理
  3. 流式分块 TTS
  4. 异步队列处理
  5. 提供商抽象层

🔄 **本项目改进**：
  1. 完全本地化（无云服务）
  2. 单个统一二进制（无微服务）
  3. 轻量级框架（Python 而非 Node.js）
  4. 模型自主管理与量化
  5. 简化与聚焦

---

## 总结

这个规划涵盖了从基础架构、技术选型、数据流设计、到完整实现的所有方面。核心创新是**流式 TTS + 并行 LLM + 本地 NPU 推理**的组合，确保完整的离线对话体验。

**下一步**：选择首个阶段的任务并开始编码！

