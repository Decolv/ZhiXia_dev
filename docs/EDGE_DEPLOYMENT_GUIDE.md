# 🚀 边缘计算快速开始指南 (4GB RAM / RDK)

> **目标场景**：RDK（基于高通骁龙）、树莓派、Jetson Nano 等 4GB RAM 边缘设备  
> **模型选择**：Qwen-1.5B (Q3_K_M) GGML 量化  
> **预期性能**：< 5 秒端到端响应，实时流式播放

---

## 📋 快速检查清单

### 硬件验证（5分钟）

```bash
# 1. 检查系统信息
  uname -a                    # 确认 Linux 系统
  free -h                     # 应显示 ~4GB 可用内存
  df -h                       # 至少 10GB 剩余存储
  cat /proc/cpuinfo          # 检查 CPU 核心数（通常 4-8 核）

# 2. 检查 NPU（可选，如果可用）
  ls /dev/hexagon*           # 高通 Hexagon NPU（RDK 特定）
  pip list | grep qnn        # 检查 QNN SDK 是否安装
```

### 软件环境（10分钟）

```bash
# 1. 安装 Python 3.10+
  python3 --version          # 必须 >= 3.10

# 2. 创建虚拟环境（推荐）
  python3 -m venv voice-env
  source voice-env/bin/activate

# 3. 安装依赖
  pip install --upgrade pip
  pip install -r requirements-edge.txt
    # 包含：llama-cpp-python, faster-whisper, pyaudio, etc.
```

### 模型准备（30-60分钟取决于网络）

```bash
# 1. 自动下载脚本
  python download_edge_models.py
    # 交互式选择模型
    # 首选：Qwen-1.5B-Chat (Q3_K_M)

# 或 2. 手动下载（如果脚本失败）
  mkdir -p models/{stt,llm,tts,vad}
  
  # STT（自动下载，第一次运行时）
  python -c "from faster_whisper import WhisperModel; \
             WhisperModel('tiny', device='cpu')"
  
  # LLM（GGUF 量化版，推荐）
  huggingface-cli download second-state/Qwen-1.5B-Chat-GGUF \
    qwen-1.5b-chat-q3_k_m.gguf --local-dir models/llm
  
  # TTS 和 VAD（可选，可以使用备选方案）
```

### 配置设置（5分钟）

```bash
# 1. 复制 RDK 专用配置
  cp config-edge-rdk.yaml config.yaml

# 2. 编辑配置（如需要）
  # 检查以下参数：
  # - stt.model_name           # 通常 "tiny"
  # - llm.model_path           # 确保指向下载的模型
  # - llm.n_threads            # 设置为 CPU 核心数
  # - llm.n_ctx                # 限制为 512（内存约束）
  # - tts.engine               # Kokoro 或备选
  
  nano config.yaml  # 或用你喜欢的编辑器
```

---

## 🎯 三步启动

### 步骤 1：验证模型加载（5分钟）

```bash
# 测试 LLM 模型是否能正常加载
python -c "
from llama_cpp import Llama

print('📥 加载 Qwen LLM...')
llm = Llama(
    model_path='models/llm/qwen-1.5b-chat-q3_k_m.gguf',
    n_threads=4,
    n_gpu_layers=-1
)
print('✅ 模型加载成功！')

# 测试推理
response = llm('你好', max_tokens=50)
print(f'测试输出：{response}')
"

# 或测试 STT
python -c "
from faster_whisper import WhisperModel
print('📥 加载 Whisper...')
model = WhisperModel('tiny', device='cpu', compute_type='int8')
print('✅ STT 模型加载成功！')
"
```

### 步骤 2：运行应用

```bash
# 基础启动
python main.py --config config.yaml

# 或带 debug 输出
python main.py --config config.yaml --debug

# 或指定特定端口（如果有 Web UI）
python main.py --config config.yaml --port 8080
```

### 步骤 3：开始对话

```
🎤 监听中...
👤 用户：你好，今天天气怎么样？
（等待 1-2 秒推理...）
🤖 AI 思考中...
🔊 TTS：我不太清楚实时天气...
（实时播放回复）
```

---

## ⚡ 性能基准

### 单轮对话延迟分解

| 阶段 | 耗时 | 说明 |
|------|------|------|
| VAD 检测 | 50-200ms | 实时进行（用户说话时） |
| STT 转录 | 800-1200ms | Whisper-tiny（CPU） |
| LLM 首词 | 600-1000ms | 1.5B 模型思考时间 |
| LLM 生成 | ~100ms/token | 10-15 tokens/sec |
| TTS 合成 | 300-500ms | Kokoro ONNX |
| 播放延迟 | 0-100ms | 流式（无缓冲） |
| **总计** | **1.8-5 秒** | **首词到播放** |

示例时间线：

```
0.0s ───┐ 用户说话："你好"
        │
0.8s ───┤ STT 完成，识别文本
        │
1.4s ───┤ LLM 首词生成，TTS 开始
        │
2.0s ───┤ 🔊 用户开始听到回复 ✅
        │
2.5s ───┤ LLM 继续生成，TTS 并行处理
        │
5.0s ───┘ 完整回复结束
```

### 内存占用

```
启动时：
  - OS/基础: ~400MB
  - Python: ~200MB
  - STT loaded: +150MB
  - LLM loaded: +700MB (Q3_K_M)
  - TTS loaded: +160MB
  ─────────────────────────
  总计: ~1.6GB

运行时峰值（推理中）：
  - 上述基础: 1.6GB
  - LLM 工作缓冲: +200MB
  - 音频缓冲: +100MB
  ─────────────────────────
  总计: ~2.0GB ✅ 远低于 4GB
```

---

## 🔧 调优指南

### 情景 1：内存不足 (OOM) 错误

```bash
# ✋ 立即停止，执行以下操作：

# 方案 A：降级量化版本（最快）
  # 改用 Q2_K GGUF 而非 Q3_K_M
  huggingface-cli download second-state/Qwen-1.5B-Chat-GGUF \
    qwen-1.5b-chat-q2_k.gguf --local-dir models/llm
  
  # 更新 config.yaml
  # llm.model_path: "models/llm/qwen-1.5b-chat-q2_k.gguf"

# 方案 B：更小的模型
  # 改用 Qwen-0.5B（内存减半）
  huggingface-cli download Qwen/Qwen-0.5B-Chat --local-dir models/llm

# 方案 C：减少上下文窗口
  # 在 config.yaml 中修改
  # llm:
  #   n_ctx: 256           # 从 512 降低

# 方案 D：减少生成长度
  # llm:
  #   max_tokens: 128      # 从 256 降低
```

### 情景 2：推理缓慢 (< 3 tokens/sec)

```bash
# 检查 NPU 是否启用
cat /proc/1/environ | tr '\\0' '\\n' | grep QNN

# 方案 A：增加线程数
  # config.yaml
  # llm:
  #   n_threads: 8        # 从 4 增加（如果 RDK 有 8+ 核）

# 方案 B：确认 GPU/NPU 使用
  # 查看实际执行提供者
  python -c "
from llama_cpp import Llama
llm = Llama('models/qwen-1.5b-chat-q3_k_m.gguf')
print(f'GPU 层数激活：{llm.n_gpu_layers}')
  "

# 方案 C：减少批大小或预分配
  # config.yaml 已设为最优值，通常无需改动
```

### 情景 3：STT 识别错误或超时

```bash
# 方案 A：调整 VAD 敏感度
  # config.yaml
  # vad:
  #   threshold: 0.5      # 更敏感（从 0.6 降低）
  #   min_silence_duration_ms: 600  # 拉长，避免误检

# 方案 B：增加音频超时
  # pipeline:
  #   max_recording_seconds: 60  # 允许更长录音

# 方案 C：检查麦克风
  python -m sounddevice  # 列出可用音频设备
  # 确认 input_device_index 指向正确的麦克风
```

---

## 📂 文件结构参考

```
项目根目录/
├── main.py                         # 应用入口
├── config-edge-rdk.yaml           # RDK 专用配置（推荐）
├── download_edge_models.py         # 模型下载脚本
├── requirements-edge.txt           # 轻量依赖（<100 packages）
│
├── src/
│   ├── services/
│   │   ├── edge_llm_service.py    # GGML LLM（关键）
│   │   ├── stt_service.py         # faster-whisper
│   │   ├── tts_service.py         # Kokoro/FastPitch
│   │   └── vad_service.py         # Silero VAD
│   │
│   ├── pipeline/
│   │   ├── edge_pipeline.py       # 单线程顺序处理（关键）
│   │   └── event_bus.py           # 事件驱动
│   │
│   └── utils/
│       ├── text_chunker.py        # 文本分块
│       └── memory_optimizer.py    # 内存优化
│
├── models/                         # 模型存储
│   ├── stt/
│   │   └── whisper-tiny.pt
│   ├── llm/
│   │   └── qwen-1.5b-chat-q3_k_m.gguf  # ⭐ 核心
│   ├── tts/
│   │   └── kokoro.onnx
│   └── vad/
│       └── silero_vad.jit
│
├── logs/                           # 日志目录
└── LOCAL_AI_VOICE_SYSTEM.md       # 完整文档（含第七章边缘优化）
```

---

## 🚨 故障排除

### 问题：`RuntimeError: out of memory`

```bash
# 诊断
free -h                              # 检查可用内存
ps aux | grep python                 # 查看进程占用

# 解决方案
# 1. 关闭其他应用（释放内存）
# 2. 使用 Q2_K 量化（较快修复）
# 3. 等待 5 分钟让 OS 回收内存，重启应用
```

### 问题：`ModuleNotFoundError: No module named 'llama_cpp'`

```bash
# 重新安装
pip uninstall llama-cpp-python
pip install llama-cpp-python --no-cache-dir

# 或指定版本
pip install 'llama-cpp-python==0.2.0'
```

### 问题：推理完全无法运行（GGUF 模型加载失败）

```bash
# 验证 GGUF 模型完整性
file models/llm/*.gguf               # 检查文件类型
ls -lh models/llm/*.gguf             # 检查大小（应 ~750MB）

# 重新下载
rm models/llm/*.gguf
huggingface-cli download second-state/Qwen-1.5B-Chat-GGUF \
  qwen-1.5b-chat-q3_k_m.gguf --local-dir models/llm
```

### 问题：声音播放没反应

```bash
# 检查音频设备
python -m sounddevice
# 输出本地音频设备列表

# 检查配置
grep -A 5 'audio:' config.yaml
# 确保 output_device_index 指向正确设备

# 测试音频输出
python -c "
import pyaudio
import numpy as np

# 生成测试音频（440Hz 正弦波，1 秒）
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, output=True)

t = np.linspace(0, 1, 16000)
sine = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

stream.write(sine.tobytes())
stream.close()
p.terminate()

print('✅ 音频输出成功！')
"
```

---

## 📚 深入文档

- **完整架构文档**：[LOCAL_AI_VOICE_SYSTEM.md](LOCAL_AI_VOICE_SYSTEM.md#七边缘计算极限优化)
  - 第七章：边缘计算极限优化（本指南的扩展）
  - 详细的内存预算、模型选择、NPU 配置

---

## 🧪 最简 STT CLI（RDK 适配）

目标：仅验证「录音 → STT → 终端打印文字」，不接入 LLM/TTS。

### 1) 安装最小依赖

```bash
pip install --upgrade pip
pip install -r requirements-edge-stt.txt
```

### 2) 预热 Whisper tiny 模型（首次运行会自动下载）

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"
```

### 3) 运行最简 CLI

```bash
python main/cli_stt_rdk.py --config config-edge-rdk-stt.yaml
```

可选参数：

```bash
# 每轮录音 5 秒
python main/cli_stt_rdk.py --config config-edge-rdk-stt.yaml --seconds 5

# 循环录音模式（每轮按 Enter）
python main/cli_stt_rdk.py --config config-edge-rdk-stt.yaml --loop
```

### 4) 预期输出

```text
🚀 启动 RDK 最简 STT CLI
按 Enter 开始录音...
🎙️ 开始录音：4.0 秒...
✅ 录音完成：recordings/record_*.wav
🧠 正在识别...
📝 识别结果：...
```
  
- **模型选择对比**：[LOCAL_AI_VOICE_SYSTEM.md#722-中文支持矩阵](LOCAL_AI_VOICE_SYSTEM.md#722)
  
- **RDK NPU 推理**：[LOCAL_AI_VOICE_SYSTEM.md#74-npu-推理加速配置rdk-高通骁龙](LOCAL_AI_VOICE_SYSTEM.md#74)

---

## 💡 最佳实践

### ✅ 推荐做法

```python
# 1. 延迟初始化模型
#    不要在启动时加载所有模型
#    只在首次使用时加载

# 2. 流式处理（不缓冲全部输出）
#    LLM token → 立即分块 → 立即 TTS → 实时播放

# 3. 严格垃圾回收
#    每个阶段完成后立即 gc.collect()

# 4. 限制上下文
#    max_context_tokens = 256  # 不要存储完整历史
```

### ❌ 避免做法

```python
# 1. 并发 TTS 任务
#    4GB 内存无法支持多个 TTS 推理同时进行

# 2. 缓存大型中间结果
#    每次使用完立即释放内存

# 3. 不受限的 LLM 输出
#    max_tokens 必须设置，防止 OOM

# 4. 在主线程中进行重 I/O
#    使用 asyncio 或线程池
```

---

## 🆘 获取帮助

| 问题类型 | 建议 |
|---------|------|
| **模型选择** | 参考第 7.2 章节：0.9B/1.5B 对比 |
| **内存优化** | 参考第 7.3 章节：单线程设计 |
| **NPU 配置** | 参考第 7.4 章节：RDK/QNN 配置 |
| **延迟调优** | 参考第 7.6 章节：延迟基准与优化策略 |
| **部署失败** | 查看本指南的"故障排除"部分 |

---

**祝部署顺利！🚀**

生成时间：2024年 | 针对 RDK + 4GB RAM + 10TOPS NPU
