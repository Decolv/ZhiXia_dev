# ZhiXia 综合测试 Notebook 使用指南

这个Notebook整合了ZhiXia项目的所有核心功能测试，采用 **模块化设计** — 复杂逻辑封装在helper模块中，Notebook只保留关键代码。

## 📋 文件结构

```
ZhiXia_dev/tests/
├── ZhiXia_Test.ipynb          ← 主Notebook（仅6.5KB，核心代码）
├── notebook_helpers.py         ← 辅助模块（17KB，全部逻辑封装）
├── README.md                   ← 本文件
└── __init__.py                 ← (可选) Python包标记
```

## 🚀 快速开始

### 1. 安装依赖

在 RK3588 开发板上运行：

```bash
cd ~/ZhiXia_dev/tests
pip install -r requirements.txt
```

**注意**：以下依赖需要在项目根目录安装（已在主项目中配置）：
- Python 3.9
- PyTorch 2.8.0
- FunASR 1.3.1
- ChatTTS、MeloTTS、PaddleSpeech
- pyttsx3、faster-whisper
- sounddevice、modelscope

### 2. 打开Notebook

```bash
cd ~/ZhiXia_dev/tests
jupyter notebook ZhiXia_Test.ipynb
```

### 3. 运行Cell（按顺序）

| Cell | 说明 | 依赖 |
|------|------|------|
| 0 | 环境配置（必须第一个运行） | - |
| 1 | Part 1: ASR 标题 | - |
| 2 | 加载 ASR 模型 | FunASR |
| 3 | ASR 识别测试 UI | ASR 模型 |
| 4 | Part 2: LLM 标题 | - |
| 5 | 加载 LLM 模型 UI | RKLLM |
| 6 | LLM 推理测试 UI | LLM 模型 |
| 7 | Part 3: TTS 标题 | - |
| 8 | TTS 合成 UI | MeloTTS/PaddleSpeech |
| 9 | TTS 性能对比 UI | TTS 函数 |
| 10 | 完整流程 UI | 所有模块 |

## 💡 核心特性

### 1. 代码极简
- ✅ Notebook只有 **~100行关键代码**
- ✅ 每个功能Cell只需 **3-5行代码调用**
- ✅ 所有复杂逻辑封装在 `notebook_helpers.py`

### 2. 模块独立
- ✅ ASR、LLM、TTS可单独运行
- ✅ 无相互依赖
- ✅ 可跳过不需要的模块

### 3. 交互友好
- ✅ ipywidgets: Dropdown、Slider、Button、FileUpload
- ✅ 文件上传、参数调节完全可视化
- ✅ 实时音频播放

### 4. 性能对比
- ✅ 快速版 vs 离线版 **顺序执行**（非并发）
- ✅ 详细展示合成时间、内存占用
- ✅ pandas DataFrame 格式化输出

## 📁 使用流程

### 仅测试单个模块

**测试ASR**：
```
运行 Cell 0 → Cell 1a → Cell 1b
```

**测试LLM**：
```
运行 Cell 0 → Cell 2a → Cell 2b
```

**测试TTS**：
```
运行 Cell 0 → Cell 3a (或 Cell 3b for 对比)
```

### 体验完整流程

```
运行 Cell 0 → Cell 1a → Cell 2a → Cell 4
```

## ⚙️ 环境配置

Cell 0 自动处理：
- ✅ 环境变量（LD_LIBRARY_PATH, PYTHONPATH等）
- ✅ 创建必要目录（models/, output/, .cache/）
- ✅ 导入所有依赖模块

## 🎯 各模块说明

### ASR 语音识别
**模型**: FunASR Paraformer INT8

**Cell 1a - 加载**:
```python
asr_model = AutoModel(...)
```

**Cell 1b - 测试**:
```python
create_asr_ui(project_root, asr_model)
```

交互式选择：
- 指定本地WAV路径
- 上传WAV文件

输出：识别文本

### LLM NPU推理
**模型**: RKLLM (Qwen2/3)

**Cell 2a - 加载**:
```python
create_llm_ui(project_root, llm_model)
```

**Cell 2b - 测试**:
```python
create_llm_inference_ui(llm_model)
```

交互式控制：
- 模型选择
- 参数调节（temperature/top_p/max_tokens）
- 模式选择（Generate/Chat）

输出：AI回复 + 性能统计

### TTS 语音合成
**版本**:
- 快速版: MeloTTS（离线）
- 离线版: PaddleSpeech（离线）

**Cell 3a - 单版本**:
```python
create_tts_synthesis_ui(project_root, tts_synthesis_fast, tts_synthesis_offline)
```

**Cell 3b - 性能对比**:
```python
create_tts_comparison_ui(project_root, tts_synthesis_fast, tts_synthesis_offline)
```

输出：音频 + 合成时间/内存统计

### 完整流程
**Cell 4**:
```python
create_pipeline_ui(project_root, asr_model, llm_model, tts_synthesis_fast, tts_synthesis_offline)
```

流程：ASR → LLM → TTS

输出：
- 中间结果（识别文本、AI回复）
- 最终音频
- 各阶段耗时

## 📊 notebook_helpers.py 模块说明

所有UI和逻辑都封装在这个模块中：

| 函数 | 功能 |
|------|------|
| `create_asr_ui()` | ASR识别界面 + 回调逻辑 |
| `create_llm_ui()` | LLM模型加载界面 |
| `create_llm_inference_ui()` | LLM推理界面 + 参数控制 |
| `create_tts_synthesis_ui()` | TTS单版本合成 |
| `create_tts_comparison_ui()` | TTS性能对比 |
| `create_pipeline_ui()` | 完整流程界面 |

### 特点
- ✅ 所有ipywidgets配置集中管理
- ✅ 回调函数内聚
- ✅ 容易维护和扩展
- ✅ Notebook保持极简

## 🔧 常见问题

**Q: Cell中变量找不到？**
A: 确保按顺序运行。全局变量在Cell 0中定义，后续Cell修改这些变量。

**Q: 模型加载失败？**
A: 检查models/目录是否有.rkllm文件，或网络连接。

**Q: helpers导入失败？**
A: 确保notebook_helpers.py与ZhiXia_Test.ipynb在同一目录。

**Q: 内存不足？**
A: 分别运行各模块而非完整流程，或减少max_tokens。

**Q: 依赖安装失败？**
A: 确保已在项目根目录安装主依赖，然后在tests/目录安装Notebook特定依赖。

## 💾 输出文件

所有生成的音频保存在 `../output/`:
- `tts_output.wav` - Cell 3a 输出
- `tts_fast.wav` / `tts_offline.wav` - Cell 3b 对比
- `pipeline.wav` - Cell 4 输出

## 📈 优化建议

想要进一步优化？可以在notebook_helpers.py中：
- 添加更多TTS版本的支持
- 增加模型自动下载逻辑
- 实现批量测试
- 添加性能基准测试

---

**提示**: 这个设计遵循"Notebook即故事"的理念：
- Notebook是**用户交互的故事**
- helpers.py是**实现细节**

用户只需专注故事流程，无需关心技术细节。🎉
