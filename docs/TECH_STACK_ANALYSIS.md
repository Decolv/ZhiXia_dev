# AIRI 项目技术栈深度分析

## 项目概述
Project AIRI 是一个 **LLM 驱动的虚拟角色系统**，旨在重现 Neuro-sama，将 AI 虚拟形象引入现实世界。核心特点是完整的语音交互流水线：`语音输入 → LLM 处理 → 语音输出`。

---

## 一、语音输入技术栈

### 1.1 语音活动检测 (VAD - Voice Activity Detection)

#### 核心模型：**Silero VAD**
- **模型来源**：Hugging Face Community (onnx-community/silero-vad)
- **部署方式**：
  - **浏览器端**：ONNX 模型 + `@huggingface/transformers` 库（支持 WASM）
  - **桌面端（Electron/Tauri）**：
    - Rust ORT（ONNX Runtime）实现：`crates/tauri-plugin-ipc-audio-vad-ort`
    - 支持多个执行提供者：CPU、CUDA、CoreML、DirectML

#### 部署位置：
- 📁 **主实现**：[packages/stage-ui/src/workers/vad/vad.ts](packages/stage-ui/src/workers/vad/vad.ts)
- 📁 **Web 特定**：[apps/stage-web/src/workers/vad/vad.ts](apps/stage-web/src/workers/vad/vad.ts)
- 📁 **移动端**：[apps/stage-pocket/src/workers/vad/vad.ts](apps/stage-pocket/src/workers/vad/vad.ts)
- 📁 **Rust 原生**：[crates/tauri-plugin-ipc-audio-vad-ort/src/lib.rs](crates/tauri-plugin-ipc-audio-vad-ort/src/lib.rs)
- 📁 **Tauri 插件**：[crates/tauri-plugin-ipc-audio-vad-ort/](crates/tauri-plugin-ipc-audio-vad-ort/)

#### VAD 核心参数配置：
```typescript
interface BaseVADConfig {
  sampleRate: 16000,           // 采样率
  speechThreshold: 0.3,        // 语音判定阈值（可调）
  exitThreshold: 0.1,          // 退出阈值（=speechThreshold * 0.3）
  minSilenceDurationMs: 400,   // 最小静音时长
  speechPadMs: 80,             // 语音前后填充
  minSpeechDurationMs: 250,    // 最小语音时长
  maxBufferDuration: 30,       // 最大缓冲时长（秒）
  newBufferSize: 512,          // 新缓冲大小（样本数）
}
```

#### VAD 事件发出：
```typescript
interface VADEvents {
  'speech-start': void                                    // 语音检测开始
  'speech-end': void                                      // 语音检测结束
  'speech-ready': { buffer, duration }                   // 完整语音段准备就绪
  'status': { type, message }                            // 状态更新
  'debug': { message, data }                             // 调试信息
}
```

### 1.2 音频捕获与处理

#### 前端音频处理流水线：
```
麦克风 → MediaStream → Audio Worklet → VAD 处理器 → 状态事件
  ↓
Web Audio API ScriptProcessorNode / AudioWorkletNode
  ↓
PCM Float32Array 缓冲
  ↓
实时 ONNX 推理
```

#### 关键依赖：
- `@huggingface/transformers`: ONNX 模型加载与推理
- `@ricky0123/vad-web`: Web VAD 行为封装
- Web Audio API (MediaRecorder, AudioContext)
- Capacitor (移动端音频采集)

#### 采集代码示例：
📁 [packages/stage-ui/src/libs/audio/vad.ts - createVADStates](packages/stage-ui/src/libs/audio/vad.ts#L124)
```typescript
export function createVADStates(vad: BaseVAD, vadAudioWorkletUrl: string) {
  async function start(stream: MediaStream) {
    // 连接 Audio Context → Audio Worklet → VAD
    // 监听 speech-start, speech-end, speech-ready 事件
  }
}
```

### 1.3 语音转文本 (STT)

#### 支持的 STT 引擎：

| 引擎 | 类型 | 位置 | 特点 |
|------|------|------|------|
| **OpenAI Whisper** | 云端 API | `setupProviders()` | 高准确度，支持多语言 |
| **Google Cloud Speech-to-Text** | 云端 API | 提供商注册 | Google 官方 |
| **Hugging Face Transformers** | 本地 ONNX | `services/discord-bot/src/pipelines/tts.ts` | 离线支持 |
| **Ollama** | 本地模型 | 兼容 OpenAI 接口 | 自托管 |

#### STT 实现示例（Discord Bot）：
📁 [services/discord-bot/src/pipelines/tts.ts](services/discord-bot/src/pipelines/tts.ts)
```typescript
export class WhisperLargeV3Pipeline {
  static model = 'Xenova/whisper-medium.en'
  static async getInstance(progress_callback = null) {
    return await pipeline('automatic-speech-recognition', this.model)
  }
}

export async function transcribe(pcmBuffer: Buffer) {
  const whispering = await WhisperLargeV3Pipeline.getInstance()
  return whispering(pcmBuffer)  // 返回 { text: "..." }
}
```

---

## 二、LLM 调用技术栈

### 2.1 LLM 框架：**xsAI**

#### 项目特点：
- **设计理念**：OpenAI 兼容优先（类似 Vercel AI SDK，但更轻量）
- **作用**：
  - 统一的 LLM API 抽象层
  - 支持 30+ LLM 提供商
  - 流式文本生成 + 工具调用 + 结构化输出

#### 核心 API：
```typescript
import { streamText } from '@xsai/stream-text'
import { generateText } from '@xsai/generate-text'

// 流式生成（常用于 web UI）
streamText({
  baseURL: 'https://api.openai.com/v1',
  apiKey: '...',
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello' }],
  maxSteps: 10,           // 工具调用循环次数
  tools: [{ ... }],       // 支持工具调用
  onEvent: (event) => {   // 流式事件回调
    // { type: 'text-delta', text: '...' }
    // { type: 'finish', ... }
    // { type: 'tool-call', ... }
  }
})

// 非流式生成（用于 Minecraft 等）
const result = await generateText({
  ...config,
  messages,
  reasoning: { effort: 'low' | 'medium' | 'high' }  // 支持推理模型
})
```

### 2.2 支持的 LLM 提供商（40+）

#### 🌟 主流商业模型：
- **OpenAI**：GPT-4, GPT-4o, GPT-4o-mini
- **Anthropic Claude**：Claude 3.5 Sonnet, Haiku, Opus
- **Google Gemini**：Gemini 2.0, Pro, etc.
- **DeepSeek**：DeepSeek-V3, R1（推理）
- **xAI Grok**：Grok-3
- **Mistral**：Mistral Large, Small
- **Meta Llama**（通过 Together.ai, Groq）

#### 🇨🇳 中文优化：
- **Zhipu (智谱)**：GLM-4
- **通义千问 (Qwen)**：Qwen-Max, Qwen-Plus
- **讯飞星火 (Sparks)**：（有 PR，未完成）
- **SiliconFlow**：开源模型托管
- **Baichuan**：百川
- **Minimax**：文心
- **Moonshot AI (Kimi)**：多轮对话优化

#### 🔧 本地/自托管：
- **Ollama**：离线本地推理
- **vLLM**：开源 LLM 推理引擎
- **SGLang**：结构化生成
- **LM Studio**：桌面 LLM 管理器
- **OpenRouter / AIHubMix**：路由聚合

#### 提供商注册位置：
📁 [packages/stage-ui/src/libs/providers/providers](packages/stage-ui/src/libs/providers/providers)
- 每个提供商独立文件夹（e.g., `anthropic/`, `openai/`）
- 统一接口：`defineProvider<ConfigSchema>()`

### 2.3 LLM 交互流程

#### 核心 Store：LLM Store
📁 [packages/stage-ui/src/stores/llm.ts](packages/stage-ui/src/stores/llm.ts)

```typescript
export async function streamFrom(
  model: string,
  chatProvider: ChatProvider,
  messages: Message[],
  options?: StreamOptions
) {
  // 1️⃣ 消息验证 & 扁平化
  const sanitized = sanitizeMessages(messages)
  
  // 2️⃣ 工具兼容性检查
  const supportedTools = streamOptionsToolsCompatibilityOk(...)
  
  // 3️⃣ 工具收集（MCP 工具 + 调试工具 + 自定义工具）
  const tools = supportedTools ? [
    ...await mcp(),        // MCP 服务器工具
    ...await debug(),      // 调试工具
    ...await resolveTools()
  ] : undefined
  
  // 4️⃣ 流式调用
  streamText({
    ...chatConfig,
    maxSteps: 10,          // 自动工具调用重试
    messages: sanitized,
    tools,
    onEvent: (event) => {
      // 文本增量、工具调用、完成事件
    }
  })
}
```

#### Chat Orchestrator Store
📁 [packages/stage-ui/src/stores/chat.ts](packages/stage-ui/src/stores/chat.ts)

```typescript
export const useChatOrchestratorStore = defineStore('chat-orchestrator', () => {
  const sendQueue = createQueue<QueuedSend>({
    handlers: [async ({ data }) => {
      await performSend(
        sendingMessage,
        options,          // model, chatProvider, tools
        generation,
        sessionId
      )
    }]
  })
  
  async function performSend(
    text: string,
    options: SendOptions,
    generation: number,
    sessionId: string
  ) {
    // 1️⃣ 获取提供商实例
    const provider = await getProviderInstance(options.chatProvider)
    
    // 2️⃣ 构建消息队列
    const messages = await buildMessages(...)
    
    // 3️⃣ 流式生成
    await llmStore.streamFrom(
      options.model,
      provider,
      messages,
      {
        tools: options.tools,
        onStreamEvent: handleStreamEvent,
      }
    )
  }
})
```

### 2.4 LLM 上下文管理

#### Chat Context Store
📁 [packages/stage-ui/src/stores/chat/context-store.ts](packages/stage-ui/src/stores/chat/context-store.ts)

提供者层级：
```typescript
interface ContextProvider {
  priority: number
  fetch: (ctx: ChatContext) => Promise<ContextFragment>
}

// 内置提供者：
- DateTimeContext      // 当前日期时间
- CharacterContext     // 角色设定
- HistoryContext       // 聊天历史
- UserContext          // 用户信息
- CustomContext        // 自定义上下文
```

#### 上下文拼接流程：
```
消息 → [日期/时间] + [角色背景] + [历史对话] + [用户资料] → 最终消息队列 → LLM
```

---

## 三、文本转语音 (TTS) 技术栈

### 3.1 支持的 TTS 提供商

#### 🌟 云端 TTS：

| 提供商 | 支持语言 | 特点 | 关键参数 |
|--------|---------|------|---------|
| **ElevenLabs** | 29+语言 | 最高质量，支持 SSML | voice_id, stability, similarity_boost, use_speaker_boost |
| **OpenAI TTS** | 多语言 | 兼容 OpenAI | model (tts-1, tts-1-hd), voice (alloy, echo, etc) |
| **Google Cloud** | 多语言 | 高质量 | voice_name, pitch, speaking_rate |
| **Azure Speech** | 多语言 | 企业级 | - |
| **Deepgram TTS** | - | 实时低延迟 | model (aura-2-*), speaking_rate |
| **Volcengine (火山)** | 中文优化 | 字节跳动 | appId, voice_id |
| **Alibaba Cloud** | 中文优化 | 阿里云 | voice_id |
| **Moonshot (Kimi)** | 多语言 | - | - |

#### 🔧 本地 TTS：

| 引擎 | 技术 | 位置 |
|------|------|------|
| **Kokoro** | 轻量级 ONNX | [packages/stage-ui/src/workers/kokoro/worker.ts](packages/stage-ui/src/workers/kokoro/worker.ts) |
| **Index-TTS (vLLM)** | 本地服务 | `index-tts-vllm` 提供商 |
| **OpenAI 兼容** | 自定义端点 | `openai-compatible-audio-speech` |

### 3.2 Kokoro TTS：本地轻量级方案

#### 特点：
- **纯前端 ONNX 模型**：无需服务器
- **文件大小**：模型较小（MB 级）
- **部署**：Web Worker 隔离

#### Worker 实现：
📁 [packages/stage-ui/src/workers/kokoro/worker.ts](packages/stage-ui/src/workers/kokoro/worker.ts)
```typescript
async function generate(request: GenerateRequest) {
  const { text, voice } = request
  
  const result = await ttsModel.generate(text, { voice })
  const blob = await result.toBlob()
  const buffer = await blob.arrayBuffer()
  
  // 使用 transferList 避免复制
  globalThis.postMessage(
    { type: 'result', status: 'success', buffer },
    [buffer]  // 转移所有权
  )
}
```

### 3.3 TTS 处理流程：音频流水线

#### 核心组件：Speech Pipeline
📁 [packages/pipelines-audio/src/speech-pipeline.ts](packages/pipelines-audio/src/speech-pipeline.ts)

```typescript
export interface SpeechPipelineOptions<TAudio> {
  tts: (request: TtsRequest, signal?: AbortSignal) => Promise<TAudio>
  // TAudio 通常是 AudioBuffer（浏览器）或 ArrayBuffer（桌面）
}

export function createSpeechPipeline<TAudio>(options: SpeechPipelineOptions<TAudio>) {
  // 核心职责：
  // 1️⃣ 管理 Intent（流式对话单元）
  // 2️⃣ 分块文本 → TTS 请求
  // 3️⃣ 音频播放调度
  // 4️⃣ 优先级队列管理
}
```

#### TTS 分块引擎：
📁 [packages/pipelines-audio/src/processors/tts-chunker.ts](packages/pipelines-audio/src/processors/tts-chunker.ts)

```typescript
export const TTS_FLUSH_INSTRUCTION = '\u200B'  // 刷新符
export const TTS_SPECIAL_TOKEN = '\u2063'       // 特殊标记

// 分块规则：
const keptPunctuations = new Set('?？!！')      // 保留
const hardPunctuations = new Set('.。?？!！…')   // 强制分块
const softPunctuations = new Set(',，、–—:')    // 软分块

export async function* chunkTtsInput(
  input: string | ReaderLike,
  options?: TtsInputChunkOptions
) {
  // 参数：
  // - boost: 分块权重调整
  // - minimumWords: 最少词数
  // - maximumWords: 最多词数
  
  // 示例输出：
  // { text: "你好", duration: 800ms, special: false }
  // { text: "我是 AIRI", duration: 1200ms, special: false }
}
```

#### 示例：流式 TTS UI
📁 [packages/stage-ui/src/components/scenarios/providers/speech-streaming-playground.vue](packages/stage-ui/src/components/scenarios/providers/speech-streaming-playground.vue)

```vue
<script setup lang="ts">
async function handleSpeechGeneration(ctx: { data: string }) {
  const res = await props.generateSpeech(input, props.voice, false)
  const audioBuffer = await audioContext.decodeAudioData(res)
  audioQueue.enqueue({ audioBuffer, text: ctx.data })
}

// 流式处理：
async function testStreaming() {
  for await (const chunk of chunkTTSInput(props.text)) {
    ttsQueue.enqueue(chunk.text)  // 实时分块生成
  }
}

// 播放队列：
const audioQueue = createQueue({
  handlers: [
    (ctx) => {
      const source = audioContext.createBufferSource()
      source.buffer = ctx.data.audioBuffer
      source.connect(audioContext.destination)
      source.start(0)
    }
  ]
})
</script>
```

### 3.4 SSML 支持

#### SSML 生成：
📁 [packages/stage-ui/src/stores/modules/speech.ts - generateSSML](packages/stage-ui/src/stores/modules/speech.ts#L243)

```typescript
function generateSSML(
  text: string,
  voice: VoiceInfo,
  providerConfig?: {
    pitch?: number
    speed?: number
    volume?: number
  }
) {
  const prosody = {
    pitch: pitch != null ? toSignedPercent(pitch) : undefined,
    rate: speed,
    volume: volume != null ? toSignedPercent(volume) : undefined
  }
  
  // 生成 XML：
  // <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">
  //   <voice name="voice_id" gender="female">
  //     <prosody pitch="50%" rate="1.2">文本</prosody>
  //   </voice>
  // </speak>
}
```

---

## 四、底层核心架构

### 4.1 模块化设计

#### 包结构：
```
packages/
├── stage-ui/          ← 🎯 核心业务组件 (共享)
│   ├── composables/   ← Vue 可组合函数
│   ├── stores/        ← Pinia 状态管理
│   │   ├── chat.ts    ← 聊天编排
│   │   ├── llm.ts     ← LLM 流程
│   │   ├── speech-runtime.ts ← 语音流水线运行时
│   │   └── modules/   ← 业务模块
│   ├── services/      ← 核心服务
│   ├── workers/       ← Web Worker (VAD, TTS)
│   └── libs/          ← VAD, 提供商等算法库
├── pipelines-audio/   ← 📡 音频流水线编排
├── stage-pages/       ← UI 页面基座
├── stage-shared/      ← 跨平台共享逻辑
├── ui/                ← 原子组件库 (reka-ui)
├── server-sdk/        ← 服务端 SDK
└── i18n/              ← 国际化

apps/
├── stage-web/         ← Web 应用 (Vue + Vite)
├── stage-tamagotchi/  ← Electron 应用
├── stage-pocket/      ← 移动应用 (Capacitor)
└── server/            ← Node.js 后端
```

### 4.2 IPC 和事件系统：Eventa

#### 概念：
- **Transport-agnostic**：不依赖特定框架（Electron, WebSocket, Worker 等）
- **类型安全**：TypeScript 优先
- **流式 RPC**：支持双向流、工具调用循环

#### 关键用途：

##### VAD 事件系统：
```typescript
// 定义包装类
export interface BaseVAD {
  on<K extends keyof VADEvents>(event: K, callback: VADEventCallback<K>): void
  processAudio(inputBuffer: Float32Array): Promise<void>
}

// VAD 事件：
'speech-start'  → speech-ready → 'speech-end'
```

##### Speech Pipeline 事件：
📁 [packages/stage-ui/src/services/speech/bus.ts](packages/stage-ui/src/services/speech/bus.ts)

```typescript
// 事件定义：
export const speechIntentStartEvent = 'speech-intent:start'
export const speechIntentLiteralEvent = 'speech-intent:literal'
export const speechIntentSpecialEvent = 'speech-intent:special'
export const speechIntentEndEvent = 'speech-intent:end'

// 使用 Eventa 跨进程通信：
context.on(speechIntentLiteralEvent, (evt) => {
  const intent = remoteIntentMap.get(evt.body.intentId)
  intent?.writeLiteral(evt.body.value)
})
```

#### Speech Pipeline Runtime：
📁 [packages/stage-ui/src/services/speech/pipeline-runtime.ts](packages/stage-ui/src/services/speech/pipeline-runtime.ts)

```typescript
export interface SpeechPipelineRuntime {
  openIntent(options?: IntentOptions): IntentHandle
  registerHost(pipeline: ReturnType<typeof createSpeechPipeline>): Promise<void>
  isHost(): boolean
  dispose(): Promise<void>
}

// 实现跨上下文 Intent 管理
// 允许多个地方（Web Worker, 主线程等）发送文本到同一个 TTS 流水线
```

### 4.3 数据流完整链路

#### 完整交互流程图：
```
┌─────────────────────────────────────────────────────────────┐
│                    🎤 语音输入阶段                           │
├─────────────────────────────────────────────────────────────┤
│
│  麦克风 MediaStream
│      ↓
│  Audio Worklet Node (采样率: 16000Hz)
│      ↓
│  VAD Worker (Silero ONNX)
│      ↓
│  speech-start → speech-ready (PCM Buffer) → speech-end
│      ↓
│  语音 → STT API (Whisper/Google/etc)
│      ↓
└───────────────────────────────────────┬─────────────────────┘
                                        │ 文本输出
                                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    🧠 LLM 处理阶段                           │
├─────────────────────────────────────────────────────────────┤
│
│  Chat Orchestrator Store
│      ↓
│  1. 消息验证 & 扁平化
│  2. 收集上下文 (日期、角色、历史)
│  3. 工具兼容性检查
│  4. MCP 工具发现
│      ↓
│  xsAI streamText() → LLM Provider
│      ↓
│  OpenAI / Claude / DeepSeek / Qwen / ... → LLM API
│      ↓
│  流式事件处理：
│    - text-delta (文本增量)
│    - tool-call (工具调用)
│    - tool-result (工具结果)
│    - finish (完成)
│      ↓
│  文本生成完成
│      ↓
└───────────────────────────────────────┬─────────────────────┘
                                        │ LLM 输出文本
                                        ↓
┌─────────────────────────────────────────────────────────────┐
│                    🔊 语音输出阶段                           │
├─────────────────────────────────────────────────────────────┤
│
│  Speech Pipeline Runtime
│      ↓
│  Intent 开启 (streamId, priority)
│      ↓
│  文本分块 (TtsChunker):
│      ├─ 按句号/问号/感叹号分割
│      ├─ word count 限制
│      └─ 最小/最大单位约束
│      ↓
│  并行 TTS 生成队列：
│      ├─ "你好" → ElevenLabs API → 生成音频块
│      ├─ "我是" → 并行请求
│      └─ "AIRI" → 优先级排队
│      ↓
│  Audio Buffer 收集 → AudioContext 播放
│      ↓
│  实时流式播放 (无缓冲等待)
│      ↓
└───────────────────────────────────────────────────────────────┘
```

### 4.4 状态管理架构（Pinia）

#### 核心 Stores：
```typescript
// 1️⃣ Chat 层
useChatSessionStore          // 会话信息 (id, 创建时间)
useChatStreamStore          // 流式消息 (当前流, 事件队列)
useChatContextStore         // 上下文提供者层级
useChatOrchestratorStore    // 核心编排 (消息队列, 发送逻辑)

// 2️⃣ LLM 配置
useConsciousnessStore       // 选定模型/提供商
useProvidersStore           // 所有提供商配置

// 3️⃣ 语音处理
useSpeechRuntimeStore       // Speech Pipeline 运行时
useSpeechStore              // 活跃 TTS/STT 配置
useChatVoiceStore           // 角色语音设置

// 4️⃣ 其他
useCharacterStore           // 角色信息
useSettingsStore            // 用户偏好设置
```

#### Store 数据流示例：
```typescript
// 用户输入文本
const sendOptions = {
  model: 'gpt-4',
  chatProvider: activeChatProvider,
  tools: [...],
}

// Chat Orchestrator 处理
await useChatOrchestratorStore().performSend(
  userText,
  sendOptions,
  generation,
  sessionId
)
  ↓
// 消息入队
sendQueue.enqueue({
  sendingMessage: userText,
  options: sendOptions,
  generation,
  sessionId,
})
  ↓
// LLM Store 流式生成
await llmStore.streamFrom(model, provider, messages, {
  onStreamEvent: (event) => {
    // 更新 useChatStreamStore
  }
})
  ↓
// Speech Pipeline 接收输出
speechRuntime.registerHost(speechPipeline)
speechPipeline.addIntent({
  text: llmOutput,
  priority: 'normal'
})
```

### 4.5 依赖注入：injeca (仅 Electron 应用)

#### Desktop App 引导：
📁 [apps/stage-tamagotchi/src/main/index.ts](apps/stage-tamagotchi/src/main/index.ts)

```typescript
// Tauri + Electron 应用中
// 使用 injeca 管理服务生命周期

// 示例：
container.provide(AudioService)
container.provide(TauriPlugin)
container.provide(...)

// 在页面中注入
const audioService = container.get(AudioService)
```

---

## 五、关键技术决策

### 5.1 为什么选择 xsAI？

| 决策 | 原因 |
|------|------|
| **轻量级** | 比 Vercel AI SDK 小，依赖少 |
| **多提供商** | 支持 40+ LLM 提供商无缝切换 |
| **流式优先** | 天生支持流式文本 + 工具调用 |
| **TypeScript** | 类型安全，IDE 友好 |
| **OpenAI 兼容** | 降低迁移成本，支持本地 Ollama |

### 5.2 为什么选择 Silero VAD？

| 决策 | 原因 |
|------|------|
| **ONNX** | 跨平台、跨运行时（Web/Native） |
| **准确性** | 相比 WebRTC VAD，误正率低 |
| **轻量** | 模型大小 ~40MB（ONNX） |
| **双平台** | WASM（浏览器）+ ORT（Rust） |
| **可定制** | 支持 threshold 动态调整 |

### 5.3 为什么采用 Streaming Pipeline？

| 特点 | 意义 |
|------|------|
| **分块 TTS** | 避免等待完整响应，提高实时性 |
| **优先级队列** | 确保重要消息优先播放 |
| **异步处理** | LLM 和 TTS 并行运行 |
| **Intent 管理** | 支持多个来源并发发送文本 |

---

## 六、部署架构变体

### 6.1 Web 应用 (stage-web)
```
浏览器 ──[WebSocket]──→ Backend 服务器
  ↓
  ├─ 前端 VAD (Silero ONNX + WASM)
  ├─ STT (API 调用: Whisper/Google)
  ├─ LLM 调用 (xsAI + OpenAI/etc)
  ├─ TTS (API: ElevenLabs/OpenAI) 流式生成
  └─ 音频播放 (Web Audio API)
```

### 6.2 Desktop 应用 (stage-tamagotchi / Electron)
```
Electron 主进程 ────────────────────┐
  ├─ Tauri 插件 (Rust)             │
  │  └─ Silero VAD (ORT)            │ ← 原生性能
  │  └─ 自定义 IPC 命令             │
  └──────────────────────────────┐  │
Electron 渲染进程                │  │
  ├─ STT (API 或 本地 Whisper)    │  │
  ├─ LLM (xsAI API)              ├──┤ 同步 via Eventa
  ├─ TTS (API 或本地 Kokoro)     │  │
  └─ Web Audio 播放              │  │
        ↓                         │  │
  Electron 主进程                 ↓  │
  ├─ 窗口管理                     ├──┘
  └─ 系统集成 (托盘, 快捷键)     ↑
```

### 6.3 移动应用 (stage-pocket / Capacitor + Vue)
```
iOS/Android 设备
  ├─ Capacitor 原生 API
  │  └─ 麦克风权限 + 音频输入
  ├─ Web 层 (Vue 3)
  │  ├─ 前端 VAD (Silero WASM)
  │  ├─ STT (API)
  │  ├─ LLM (xsAI)
  │  ├─ TTS (API)
  │  └─ WAV 文件处理
  └─ 后台播放控制 (Capacitor Media)
```

---

## 七、性能优化技术

### 7.1 VAD 优化
- **Worker 隔离**：不阻塞主线程
- **缓冲策略**：预先分配 Float32Array，避免 GC
- **Threshold 自适应**：用户可调
- **快速路径**：静音检测快速返回

### 7.2 TTS 优化
- **分块并行**：文本分块后并行请求 TTS API
- **缓存**：相同文本使用缓存音频
- **流式播放**：无需等待全部生成完毕
- **本地 Fallback**：本地 Kokoro 降级方案

### 7.3 LLM 优化
- **流式生成**：逐 token 发送到 UI，无缓冲延迟
- **工具调用循环**：maxSteps 限制重试次数
- **消息扁平化**：避免嵌套数组破坏兼容性
- **工具兼容性缓存**：减少重复检测

### 7.4 IPC 优化
- **Zero-Copy Transfer**：Audio Buffer 使用 Transferable
- **Event Batching**：汇总小事件为大包
- **Context 隔离**：Worker 中不涉及 DOM

---

## 八、关键代码文件索引

### 核心流程
| 功能 | 文件 |
|------|------|
| VAD 主实现 | [packages/stage-ui/src/workers/vad/vad.ts](packages/stage-ui/src/workers/vad/vad.ts) |
| LLM 流处理 | [packages/stage-ui/src/stores/llm.ts](packages/stage-ui/src/stores/llm.ts) |
| Chat 编排 | [packages/stage-ui/src/stores/chat.ts](packages/stage-ui/src/stores/chat.ts) |
| TTS 分块 | [packages/pipelines-audio/src/processors/tts-chunker.ts](packages/pipelines-audio/src/processors/tts-chunker.ts) |
| Speech Pipeline | [packages/pipelines-audio/src/speech-pipeline.ts](packages/pipelines-audio/src/speech-pipeline.ts) |
| Speech Runtime | [packages/stage-ui/src/services/speech/pipeline-runtime.ts](packages/stage-ui/src/services/speech/pipeline-runtime.ts) |

### 提供商配置
| 位置 | 内容 |
|------|------|
| [packages/stage-ui/src/libs/providers/](packages/stage-ui/src/libs/providers/) | 所有 LLM 提供商定义 |
| [packages/stage-ui/src/stores/providers.ts](packages/stage-ui/src/stores/providers.ts) | TTS 提供商注册 |

### UI 组件
| 功能 | 文件 |
|------|------|
| TTS 播放器 | [packages/stage-ui/src/components/scenarios/providers/speech-playground.vue](packages/stage-ui/src/components/scenarios/providers/speech-playground.vue) |
| 流式 TTS | [packages/stage-ui/src/components/scenarios/providers/speech-streaming-playground.vue](packages/stage-ui/src/components/scenarios/providers/speech-streaming-playground.vue) |
| 语音设置 | [packages/stage-pages/src/pages/settings/modules/speech.vue](packages/stage-pages/src/pages/settings/modules/speech.vue) |

---

## 九、总结与特色

### 🎯 创新点
1. **完整 AI 虚拟形象系统**：语音输入 → LLM 推理 → 实时语音输出
2. **多提供商支持**：40+ LLM + 15+ TTS 提供商，快速切换
3. **流式处理架构**：减少用户感知延迟
4. **跨平台一致**：Web / Desktop / Mobile 同一代码库
5. **离线支持**：本地 Silero VAD + Kokoro TTS + Ollama

### 🛠️ 技术栈胶水
- **xsAI**：LLM 通信层的"USB 接口"
- **Eventa**：跨进程事件通信的"神经网络"
- **Pinia**：状态管理的"中央大脑"
- **Web Audio API + Worklets**：音频处理的"声卡驱动"

### 📊 架构层次
```
应用层    ← VS Code / Web UI / Electron
  ↓
状态层    ← Pinia Stores (Chat, LLM, Providers)
  ↓
业务层    ← Composables (useVAD, useSpeech, useChat)
  ↓
服务层    ← xsAI, Eventa, Pipeline Runtime
  ↓
算法层    ← Silero VAD, Kokoro TTS, 文本分块
  ↓
运行时    ← ONNX, Web Audio, IPC
```

### 🚀 下一步优化方向
- [ ] 本地量化 LLM (Ollama + GGUF)
- [ ] 更多 TTS 本地引擎
- [ ] 智能缓存系统
- [ ] 多角色并行对话
- [ ] 端到端加密通信

